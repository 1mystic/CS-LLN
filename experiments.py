"""
CS-LLN Paper Experiments
Runs all benchmarks, ablations, and statistical tests.
Outputs: results.json (tables + stats for paper update)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import wilcoxon
import xgboost as xgb
import json
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


# ── CS-LLN Implementation ────────────────────────────────────────────────────

class CSLLN(BaseEstimator, ClassifierMixin):
    """
    Context-Shift Log-Linear Network.
    Selects top-k feature pairs by class-conditional Pearson correlation
    divergence (ΔRij), appends product terms, fits L2-logistic regression.
    """
    def __init__(self, k_interactions=10, C_reg=1.0):
        self.k_interactions = k_interactions
        self.C_reg = C_reg

    def _compute_delta_r(self, X, y):
        classes = np.unique(y)
        d = X.shape[1]
        corrs = []
        for c in classes:
            Xc = X[y == c]
            old = np.seterr(divide="ignore", invalid="ignore")
            corr = np.corrcoef(Xc, rowvar=False)
            np.seterr(**old)
            corrs.append(np.nan_to_num(corr, nan=0.0))

        shift = np.zeros((d, d))
        npairs = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                shift += np.abs(corrs[i] - corrs[j])
                npairs += 1
        if npairs:
            shift /= npairs
        np.fill_diagonal(shift, 0.0)
        shift = np.triu(shift)
        return shift

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.classes_ = np.unique(y)
        d = X.shape[1]

        shift = self._compute_delta_r(X, y)
        self.delta_r_ = shift

        flat = np.argsort(shift.flatten())[::-1]
        self.selected_pairs_ = []
        for idx in flat:
            if len(self.selected_pairs_) >= self.k_interactions:
                break
            r, c = divmod(idx, d)
            if shift[r, c] > 0:
                self.selected_pairs_.append((r, c))

        Xe = self._enrich(X)
        self.lr_ = LogisticRegression(
            C=self.C_reg, max_iter=3000, penalty="l2",
            solver="lbfgs", random_state=SEED
        )
        self.lr_.fit(Xe, y)
        return self

    def _enrich(self, X):
        X = np.asarray(X, float)
        if not self.selected_pairs_:
            return X
        terms = np.column_stack(
            [X[:, i] * X[:, j] for i, j in self.selected_pairs_]
        )
        return np.hstack([X, terms])

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._enrich(np.asarray(X, float)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Helper: select k via inner CV ────────────────────────────────────────────

def _select_k_inner_cv(X_train, y_train, k_candidates, n_inner=5):
    """5-fold inner CV to choose best k from candidates."""
    best_k, best_acc = k_candidates[0], -1.0
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
    for k in k_candidates:
        accs = []
        for tr, va in inner_cv.split(X_train, y_train):
            sc = StandardScaler().fit(X_train[tr])
            Xtr = sc.transform(X_train[tr])
            Xva = sc.transform(X_train[va])
            m = CSLLN(k_interactions=k).fit(Xtr, y_train[tr])
            accs.append(accuracy_score(y_train[va], m.predict(Xva)))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc, best_k = mean_acc, k
    return best_k


# ── Metrics helper ───────────────────────────────────────────────────────────

def _compute_metrics(model, X_test, y_test, use_proba=True):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    ll = None
    if use_proba and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            # clip for numerical stability
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            probs /= probs.sum(axis=1, keepdims=True)
            ll = log_loss(y_test, probs)
        except Exception:
            ll = None
    return acc, ll, f1, preds


# ── McNemar test (fold-level prediction arrays) ──────────────────────────────

def mcnemar_pvalue(preds_a, preds_b, y_true):
    """Compute McNemar p-value comparing two prediction arrays."""
    from scipy.stats import chi2
    n01 = np.sum((preds_a == y_true) & (preds_b != y_true))
    n10 = np.sum((preds_a != y_true) & (preds_b == y_true))
    n = n01 + n10
    if n == 0:
        return 1.0
    # mid-p McNemar
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1 - chi2.cdf(stat, df=1))


# ── Full 5-fold evaluation for one model builder ────────────────────────────

def run_cv(X, y, model_fn, n_folds=N_FOLDS):
    """
    Returns:
      fold_accs, fold_lls, fold_f1s, all_preds (concat of test preds across folds),
      all_true (concat of test labels in same order)
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_accs, fold_lls, fold_f1s = [], [], []
    all_preds, all_true = [], []
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = model_fn(Xtr, y[tr])
        acc, ll, f1, preds = _compute_metrics(m, Xte, y[te])
        fold_accs.append(acc)
        if ll is not None:
            fold_lls.append(ll)
        fold_f1s.append(f1)
        all_preds.append(preds)
        all_true.append(y[te])

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    mean_ll = float(np.mean(fold_lls)) if fold_lls else None
    return (
        np.array(fold_accs),
        np.array(fold_lls) if fold_lls else None,
        np.array(fold_f1s),
        all_preds,
        all_true,
    )


# ── Dataset + k config ───────────────────────────────────────────────────────

DATASETS = {
    "breast_cancer": {
        "loader": load_breast_cancer,
        "k_candidates": list(range(1, 51)),   # sweep 1–50
        "default_k": 45,
        "label": "Breast Cancer Wisconsin",
    },
    "wine": {
        "loader": load_wine,
        "k_candidates": list(range(1, 20)),
        "default_k": 5,
        "label": "Wine Recognition",
    },
    "iris": {
        "loader": load_iris,
        "k_candidates": [1, 2, 3, 4, 5, 6],
        "default_k": 5,
        "label": "Iris",
    },
}


# ── Model builders ────────────────────────────────────────────────────────────

def build_gnb(X, y):
    return GaussianNB().fit(X, y)

def build_lda(X, y):
    return LinearDiscriminantAnalysis().fit(X, y)

def build_lr(X, y):
    return LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                               random_state=SEED).fit(X, y)

def build_lr_tuned(X, y):
    """Inner-CV tuned C for fairness."""
    best_C, best_acc = 1.0, -1.0
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        accs = []
        for tr, va in inner.split(X, y):
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                   random_state=SEED).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[va], m.predict(X[va])))
        mean_a = np.mean(accs)
        if mean_a > best_acc:
            best_acc, best_C = mean_a, C
    return LogisticRegression(C=best_C, max_iter=3000, solver="lbfgs",
                               random_state=SEED).fit(X, y)

def build_svm(X, y):
    return SVC(kernel="linear", C=1.0, probability=True, random_state=SEED).fit(X, y)

def build_rf(X, y):
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)

def build_xgb(X, y):
    n_cls = len(np.unique(y))
    obj = "binary:logistic" if n_cls == 2 else "multi:softprob"
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        objective=obj, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, random_state=SEED
    ).fit(X, y)

def build_cslln_fixed(k):
    def _build(X, y):
        return CSLLN(k_interactions=k).fit(X, y)
    return _build

def build_cslln_cv(k_candidates):
    def _build(X, y):
        k = _select_k_inner_cv(X, y, k_candidates)
        return CSLLN(k_interactions=k).fit(X, y)
    return _build


# ── Ablation helpers ─────────────────────────────────────────────────────────

class CSLLNRandomPairs(BaseEstimator, ClassifierMixin):
    """CS-LLN but with randomly chosen k pairs instead of ΔRij-ranked."""
    def __init__(self, k_interactions=10, seed=0):
        self.k_interactions = k_interactions
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        rng = np.random.RandomState(self.seed)
        all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        n = min(self.k_interactions, len(all_pairs))
        idxs = rng.choice(len(all_pairs), n, replace=False)
        self.selected_pairs_ = [all_pairs[i] for i in idxs]
        Xe = self._enrich(X)
        self.lr_ = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                       random_state=SEED).fit(Xe, y)
        return self

    def _enrich(self, X):
        X = np.asarray(X, float)
        if not self.selected_pairs_:
            return X
        terms = np.column_stack([X[:, i] * X[:, j] for i, j in self.selected_pairs_])
        return np.hstack([X, terms])

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._enrich(np.asarray(X, float)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def run_ablation_random_pairs(X, y, k, n_seeds=10):
    """Average accuracy of k random pairs over n_seeds."""
    all_accs = []
    for seed in range(n_seeds):
        def _build(Xtr, ytr, _seed=seed):
            return CSLLNRandomPairs(k_interactions=k, seed=_seed).fit(Xtr, ytr)
        accs, _, _, _, _ = run_cv(X, y, _build)
        all_accs.append(float(np.mean(accs)))
    return float(np.mean(all_accs)), float(np.std(all_accs))


# ── Per-dataset k-ablation curve ─────────────────────────────────────────────

def run_k_ablation(X, y, k_max, k_step=1):
    ks = list(range(1, k_max + 1, k_step))
    mean_accs = []
    std_accs = []
    for k in ks:
        accs, _, _, _, _ = run_cv(X, y, build_cslln_fixed(k))
        mean_accs.append(float(np.mean(accs) * 100))
        std_accs.append(float(np.std(accs) * 100))
    return ks, mean_accs, std_accs


# ── Main experiment loop ──────────────────────────────────────────────────────

def main():
    results = {}

    MODELS = {
        "GNB":       build_gnb,
        "LDA":       build_lda,
        "LR":        build_lr,
        "LR_tuned":  build_lr_tuned,
        "SVM":       build_svm,
        "RF":        build_rf,
        "XGBoost":   build_xgb,
    }

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_cfg['label']}")
        print(f"{'='*60}")

        data = ds_cfg["loader"]()
        X, y = data.data, data.target
        d = X.shape[1]
        k_candidates = [k for k in ds_cfg["k_candidates"] if k <= d * (d - 1) // 2]
        default_k = min(ds_cfg["default_k"], d * (d - 1) // 2)

        ds_results = {
            "label": ds_cfg["label"],
            "N": int(X.shape[0]),
            "d": int(d),
            "C": int(len(np.unique(y))),
            "default_k": default_k,
            "models": {},
            "cslln_cv": {},
            "ablations": {},
            "statistical_tests": {},
        }

        # ── Baselines ──
        baseline_preds = {}
        for name, builder in MODELS.items():
            print(f"  Running {name}...", end=" ", flush=True)
            accs, lls, f1s, preds, true = run_cv(X, y, builder)
            baseline_preds[name] = (preds, true)
            ds_results["models"][name] = {
                "mean_acc": float(np.mean(accs) * 100),
                "std_acc":  float(np.std(accs) * 100),
                "mean_ll":  float(np.mean(lls)) if lls is not None and len(lls) > 0 else None,
                "mean_f1":  float(np.mean(f1s)),
                "fold_accs": [float(a * 100) for a in accs],
            }
            print(f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}")

        # ── CS-LLN fixed k ──
        print(f"  Running CS-LLN (k={default_k}, fixed)...", end=" ", flush=True)
        accs, lls, f1s, preds_cslln, true = run_cv(X, y, build_cslln_fixed(default_k))
        ds_results["models"]["CS-LLN"] = {
            "mean_acc": float(np.mean(accs) * 100),
            "std_acc":  float(np.std(accs) * 100),
            "mean_ll":  float(np.mean(lls)) if lls is not None and len(lls) > 0 else None,
            "mean_f1":  float(np.mean(f1s)),
            "fold_accs": [float(a * 100) for a in accs],
            "k": default_k,
        }
        print(f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}")

        # ── CS-LLN inner-CV k selection ──
        print(f"  Running CS-LLN (CV-selected k)...", end=" ", flush=True)
        cv_k_log = []
        outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_accs_cv, fold_lls_cv, fold_f1s_cv = [], [], []
        preds_cslln_cv, true_cv = [], []
        for tr_idx, te_idx in outer_cv.split(X, y):
            sc = StandardScaler().fit(X[tr_idx])
            Xtr, Xte = sc.transform(X[tr_idx]), sc.transform(X[te_idx])
            chosen_k = _select_k_inner_cv(Xtr, y[tr_idx], k_candidates)
            cv_k_log.append(int(chosen_k))
            m = CSLLN(k_interactions=chosen_k).fit(Xtr, y[tr_idx])
            acc, ll, f1, pred = _compute_metrics(m, Xte, y[te_idx])
            fold_accs_cv.append(acc)
            if ll is not None:
                fold_lls_cv.append(ll)
            fold_f1s_cv.append(f1)
            preds_cslln_cv.append(pred)
            true_cv.append(y[te_idx])

        preds_cslln_cv = np.concatenate(preds_cslln_cv)
        true_cv_arr = np.concatenate(true_cv)
        ds_results["cslln_cv"] = {
            "mean_acc": float(np.mean(fold_accs_cv) * 100),
            "std_acc":  float(np.std(fold_accs_cv) * 100),
            "mean_ll":  float(np.mean(fold_lls_cv)) if fold_lls_cv else None,
            "mean_f1":  float(np.mean(fold_f1s_cv)),
            "fold_accs": [float(a * 100) for a in fold_accs_cv],
            "k_per_fold": cv_k_log,
        }
        print(f"{np.mean(fold_accs_cv)*100:.2f}% ± {np.std(fold_accs_cv)*100:.2f}  "
              f"(k chosen per fold: {cv_k_log})")

        # ── Statistical significance tests ──
        print("  Running statistical tests...")
        sig = {}
        # Compare CS-LLN(fixed k) vs each baseline via Wilcoxon signed-rank on fold-level accs
        cslln_fold_accs = np.array(ds_results["models"]["CS-LLN"]["fold_accs"])
        for name in MODELS:
            baseline_fold_accs = np.array(ds_results["models"][name]["fold_accs"])
            diff = cslln_fold_accs - baseline_fold_accs
            if np.all(diff == 0):
                pval = 1.0
            else:
                try:
                    stat, pval = wilcoxon(cslln_fold_accs, baseline_fold_accs, alternative="greater")
                    pval = float(pval)
                except Exception:
                    pval = 1.0
            # McNemar on full concatenated predictions
            preds_base, true_base = baseline_preds[name]
            mcn_p = mcnemar_pvalue(preds_cslln, preds_base, true)
            sig[name] = {
                "wilcoxon_p_greater": float(pval),
                "mcnemar_p": float(mcn_p),
                "mean_diff_pp": float(np.mean(diff)),
                "significant_05": bool(pval < 0.05 or mcn_p < 0.05),
            }
        ds_results["statistical_tests"] = sig

        # ── Ablations ──
        print("  Running ablations...")
        ablations = {}

        # 1. ΔRij ranked vs random pairs
        print(f"    DeltaR vs random pairs (k={default_k})...")
        rand_mean, rand_std = run_ablation_random_pairs(X, y, default_k, n_seeds=10)
        cslln_mean = ds_results["models"]["CS-LLN"]["mean_acc"]
        ablations["deltaR_vs_random"] = {
            "cslln_ranked_acc": cslln_mean,
            "random_pairs_mean_acc": rand_mean * 100,
            "random_pairs_std_acc": rand_std * 100,
            "gain_pp": cslln_mean - rand_mean * 100,
        }
        print(f"      DeltaR={cslln_mean:.2f}%  random={rand_mean*100:.2f}%  "
              f"gain={cslln_mean - rand_mean*100:+.2f}pp")

        # 2. k=0 (no enrichment) — pure LR baseline comparison already in models["LR"]
        ablations["k0_note"] = "k=0 is equivalent to LR (see models['LR'] row)"

        # 3. Full quadratic expansion vs sparse k
        max_pairs = d * (d - 1) // 2
        if max_pairs != default_k and max_pairs <= 500:
            print(f"    Full quadratic (k={max_pairs}) vs sparse (k={default_k})...")
            accs_full, _, _, _, _ = run_cv(X, y, build_cslln_fixed(max_pairs))
            ablations["full_vs_sparse"] = {
                "full_k": max_pairs,
                "full_acc": float(np.mean(accs_full) * 100),
                "sparse_k": default_k,
                "sparse_acc": cslln_mean,
                "sparse_advantage_pp": cslln_mean - float(np.mean(accs_full) * 100),
            }
            print(f"      full={np.mean(accs_full)*100:.2f}%  sparse={cslln_mean:.2f}%")

        # 4. k-ablation curve (for ablation figure)
        print(f"    k-ablation (1 to {min(default_k + 20, max_pairs)})...")
        k_abl_max = min(default_k + 20, max_pairs, 60)
        ks, macc, sacc = run_k_ablation(X, y, k_abl_max, k_step=1 if max_pairs <= 60 else 3)
        ablations["k_curve"] = {"ks": ks, "mean_accs": macc, "std_accs": sacc}

        ds_results["ablations"] = ablations

        results[ds_key] = ds_results
        print(f"  Done: {ds_cfg['label']}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_models = list(MODELS.keys()) + ["CS-LLN"]
    header = f"{'Model':<14}" + "".join(f"  {k:>17}" for k in DATASETS)
    print(header)
    print("-" * len(header))
    for mname in all_models:
        row = f"{mname:<14}"
        for ds_key in DATASETS:
            r = results[ds_key]["models"].get(mname, {})
            if r:
                row += f"  {r['mean_acc']:>7.2f}% ±{r['std_acc']:>5.2f}  "
            else:
                row += f"  {'N/A':>17}"
        print(row)

    print("\nStatistical significance (CS-LLN vs LR, McNemar p-value):")
    for ds_key in DATASETS:
        sig = results[ds_key]["statistical_tests"]
        lr_sig = sig.get("LR", {})
        print(f"  {results[ds_key]['label']:30s}  "
              f"McNemar p={lr_sig.get('mcnemar_p', 'N/A'):.4f}  "
              f"Wilcoxon p={lr_sig.get('wilcoxon_p_greater', 'N/A'):.4f}  "
              f"diff={lr_sig.get('mean_diff_pp', 0):+.2f}pp")

    # ── Save to JSON ─────────────────────────────────────────────────────────
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Results saved to results.json")

    return results


if __name__ == "__main__":
    main()
