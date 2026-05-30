# Understanding CS-LLN — A Student's Companion

*A from-scratch guide to the maths, the model, the website, and your viva.*

You wrote this project a while ago and have since forgotten how it works. This
document rebuilds your understanding from the ground up. Read it top to bottom
once; then use **Part 6 (Viva Prep)** to rehearse.

It assumes almost nothing. If you already know probability and Naive Bayes,
skim Parts 1–2 and slow down at Part 3 (the actual contribution).

---

## Table of Contents

1. [The fundamentals you need](#part-1--the-fundamentals-you-need)
2. [Naive Bayes and why it breaks](#part-2--naive-bayes-and-why-it-breaks)
3. [CS-LLN: the idea, the design, the algorithm](#part-3--cs-lln-the-idea-the-design-the-algorithm)
4. [How the website implements it](#part-4--how-the-website-implements-it)
5. [Limitations — say these before the examiner does](#part-5--limitations--say-these-before-the-examiner-does)
6. [Viva preparation — questions and model answers](#part-6--viva-preparation)
7. [A 30-second and a 3-minute pitch](#part-7--two-pitches)

---

## Part 1 — The fundamentals you need

### 1.1 What is classification?

You have examples, each described by **features** (numbers) and a **label**
(a category). Example: an iris flower described by 4 measurements (features),
labelled as one of 3 species (label). The job of a **classifier** is to learn
a rule from labelled examples (the *training set*) that correctly labels new,
unseen examples (the *test set*).

- **Feature vector** `x = [x₁, x₂, …, x_d]` — `d` numbers describing one example.
- **Label** `y` — which of `C` classes the example belongs to.
- `N` = number of examples, `d` = number of features, `C` = number of classes.

### 1.2 Probability in one page

- **Probability** `P(A)` — how likely event `A` is, between 0 and 1.
- **Conditional probability** `P(A | B)` — probability of `A` *given that* `B`
  happened. Read “|” as “given”.
- **Joint probability** `P(A, B)` — probability `A` *and* `B` both happen.
- **Independence**: `A` and `B` are independent if knowing one tells you
  nothing about the other: `P(A, B) = P(A) · P(B)`.
- **Conditional independence**: `A` and `B` are independent *once you know* `C`:
  `P(A, B | C) = P(A | C) · P(B | C)`. This last one is the heart of Naive Bayes
  — remember it.

### 1.3 Bayes' theorem

We want `P(class | features)` — given the measurements, which class? Bayes'
theorem flips a hard question into easier pieces:

```
P(y | x)  =   P(x | y) · P(y)   /   P(x)
              ╰──────╮──────╯       ╰─╮─╯
              likelihood × prior     evidence (same for all classes)
```

- **Prior** `P(y)` — how common each class is before seeing features.
- **Likelihood** `P(x | y)` — how likely these features are *if* the class is `y`.
- The denominator `P(x)` is the same for every class, so to **pick** the best
  class we can ignore it and just compare `P(x | y) · P(y)` across classes.

To **classify**, compute that product for every class and choose the largest.
We usually take logs (sums are nicer than products numerically):

```
score(y) = ln P(y) + ln P(x | y)
```

### 1.4 Statistics you must be able to define

- **Mean** `x̄` — the average.
- **Variance / standard deviation** — how spread out the values are. Std is the
  square root of variance.
- **Standardisation (z-score)**: `z = (x − mean) / std`. After this every
  feature has mean 0 and std 1, so features measured in different units
  (cm vs mg) become comparable. *The paper uses this in Step 1.*
- **Covariance** — do two features move together? Positive = they rise together.
- **Pearson correlation** `r` — covariance *normalised* to the range `[−1, +1]`:

  ```
  r(x, y) =  Σ (xᵢ − x̄)(yᵢ − ȳ)  /  sqrt( Σ(xᵢ − x̄)²  ·  Σ(yᵢ − ȳ)² )
  ```

  - `r = +1`: perfect straight-line increase. `r = −1`: perfect decrease.
  - `r = 0`: **no _linear_ relationship** (there could still be a curved one!).
  - Crucial fact for this project: **Pearson `r` only sees straight-line
    (linear) relationships.** It is blind to curved ones. Remember this — it is
    a key limitation.

> **One mental model to keep:** a classifier draws a *boundary* in feature
> space. Points on one side get one label, the other side another. Simple models
> draw straight boundaries; richer models draw curved ones. CS-LLN's whole
> trick is to draw a curved boundary cheaply.

---

## Part 2 — Naive Bayes and why it breaks

### 2.1 The Naive Bayes (NB) classifier

NB applies Bayes' theorem with one big simplifying assumption: **all features
are conditionally independent given the class.** That turns the hard joint
likelihood into a product of easy one-feature likelihoods:

```
P(x | y) = P(x₁ | y) · P(x₂ | y) · … · P(x_d | y)
```

So the score becomes:

```
score(y) = ln P(y) + Σᵢ ln P(xᵢ | y)
```

For **Gaussian** Naive Bayes (GNB), each `P(xᵢ | y)` is modelled as a bell
curve (Gaussian) — you just estimate a mean and std per feature *per class*.
That is fast, needs little data, and is a great first baseline.

### 2.2 Why the assumption is “naive”

In real data, features are usually **correlated**. Three measurements of a
tumour's size (radius, perimeter, area) all rise together — they are not
independent. NB pretends they are.

### 2.3 The subtle, important failure (this is the paper's hook)

Here is the insight the whole paper is built on. There are two ways NB can be
“wrong” about a correlated pair:

1. **Wrong but consistently** — the pair is correlated `r = 0.6` in *every*
   class. NB ignores the correlation everywhere equally. This shifts every
   class's score by roughly the same amount, so the *boundary between classes*
   barely moves. NB is wrong, but harmlessly.

2. **Wrong in a class-specific way** — the pair is correlated `r = 0.9` in
   class A but `r = 0.0` in class B. Now NB's error is *different in each
   class*, which pushes the classes' scores by *different* amounts and
   **systematically bends the decision boundary the wrong way.** This is the
   damaging case.

So the pairs worth fixing are exactly the ones whose **correlation changes
across classes**. That change is what the paper names a **context shift**.

> ⚠️ Precision point (examiners love this): GNB does not literally state
> “`r = 0`”. It assumes the class-conditional density *factorises*. *Under a
> Gaussian model* that factorisation is equivalent to zero correlation, which is
> why the paper frames it as “GNB implicitly assumes `R⁽ᶜ⁾ᵢⱼ = 0`”. Zero
> correlation is **necessary but not sufficient** for independence in general.
> The corrected paper now says exactly this.

---

## Part 3 — CS-LLN: the idea, the design, the algorithm

**CS-LLN** = **C**ontext-**S**hifting **L**og-**L**inear **N**etwork.
(“Log-linear” because logistic regression is a log-linear model. There is no
neural network — the name is about the *log-linear decision form*, not an
architecture. Be ready to say this in the viva.)

### 3.1 The one-sentence idea

> Find the feature pairs whose correlation changes most between classes, build
> new “product” features only from those pairs, and feed the enlarged feature
> set to ordinary logistic regression.

It is a **feature-engineering / screening** method, *not* a new probabilistic
model. The final classifier is plain L2 logistic regression. The novelty is
**which** interaction features to add and **how to choose them cheaply**.

### 3.2 How the idea was reached (the ideation story for your viva)

1. Start from a known weakness: NB ignores feature correlations.
2. Notice that not all ignored correlations matter — only the ones that
   *change across classes* bend the boundary (Part 2.3).
3. Borrow a trick from bioinformatics: **differential co-expression** studies
   already look at gene pairs whose correlation differs between healthy and
   diseased tissue. Apply the *same principle* to general classification.
4. Turn “correlation that changes across classes” into a single number you can
   rank pairs by — the **ΔR (delta-R) score**.
5. For the top-ranked pairs, add the product `xᵢ·xⱼ` as a new feature. A product
   term lets a *linear* model carve a *curved* boundary.
6. Train logistic regression on the enriched features. Done.

### 3.3 The ΔR score (the core formula)

For each class `c`, compute the `d×d` Pearson correlation matrix `R⁽ᶜ⁾` (one
number per feature pair). Then the **Mean Absolute Context-Shift** of pair
`(i, j)` is the average, over all pairs of classes, of how much that pair's
correlation differs:

```
ΔRᵢⱼ  =   (2 / (C(C−1)))  ·  Σ_{m<n} | R⁽ᵐ⁾ᵢⱼ  −  R⁽ⁿ⁾ᵢⱼ |
```

- If a pair has the same correlation in every class → `ΔR = 0` (NB handles it
  fine, skip it).
- If a pair is strongly correlated in one class and not another → `ΔR` is large
  (this is the damaging pair, target it).
- Range: `ΔR ∈ [0, 2]` (because each `r ∈ [−1, 1]`, so the gap is at most 2).
- **It is closed-form**: one pass over the data, no gradient descent, no tuning.

### 3.4 The five steps (Algorithm 1)

| Step | Name | What happens | Cost |
|---|---|---|---|
| 1 | **Standardise** | z-score each feature (mean 0, std 1) using *training* stats only | O(Nd) |
| 2 | **Class correlations** | For each class, compute `R⁽ᶜ⁾` (Pearson matrix) | O(Nd²) |
| 3 | **Context-shift score** | Compute `ΔRᵢⱼ` for every pair | O(Cd²) |
| 4 | **Sparse selection Ωₖ** | Rank pairs by `ΔR`, keep the top **k** | O(d² log d) |
| 5 | **Enrich + classify** | Append `xᵢ·xⱼ` for chosen pairs, train L2 logistic regression | O(N(d+k)·iters) |

The enriched vector is:

```
x̃ = [ x₁, …, x_d ,  {xᵢ·xⱼ for each (i,j) in Ωₖ} ]   ∈ ℝ^(d+k)
```

The decision boundary is **linear in `x̃`** but **quadratic (curved) in the
original `x`**, because the product terms are nonlinear. That is how a linear
classifier gets a curved boundary — cheaply, and only where it helps.

### 3.5 Why “sparse” matters

A full quadratic expansion adds *all* `C(d,2)` products. For Breast Cancer
(`d = 30`) that is 435 extra features → 465 total, which causes
**multicollinearity** (redundant, overlapping features that destabilise the
model). CS-LLN adds only `k` of them (e.g. `k = 45`), giving 75 features — a
6.2× reduction — while keeping the *useful* interactions. “Sparse” = “few, but
the right few.”

### 3.6 Why logistic regression as the final step?

- It produces a **probability** via the softmax/sigmoid, not just a label.
- L2 regularisation keeps weights small and stable (handles leftover
  collinearity).
- It is convex → one global optimum, reliable training.
- The selected pairs `Ωₖ` are stored explicitly, so the model stays
  **interpretable**: you can read off *which* interactions mattered.

### 3.7 The headline results (know these numbers)

| Dataset | d | C | CS-LLN | Best baseline | Takeaway |
|---|---|---|---|---|---|
| Breast Cancer | 30 | 2 | **97.72%** | LR 97.37% | 63% fewer errors than GNB (92.97%); log-loss ~10× better *vs GNB* |
| Wine | 13 | 3 | **98.89%** | LR 98.33% | a single pair (k=1) already hits 99.44% |
| Iris | 4 | 3 | 96.67% | LDA 97.33% | LDA wins — d=4 is too small for interactions to help |

**Be honest in the viva:** CS-LLN wins 2 of 3, and the margins over LR/SVM are
small (within one std). The big, unambiguous win is over *Naive Bayes*, which
is the method it is designed to fix. The ~10× log-loss claim is *versus GNB*;
against LR the calibration is about equal.

---

## Part 4 — How the website implements it

The site (`index.html` + `style.css` + `data.js` + `app.js`) is a no-server,
vanilla-JS recreation so users can *feel* the algorithm. Load order matters:
`data.js` before `app.js`.

### 4.1 Where each concept lives in the code (`app.js`)

| Concept | Function |
|---|---|
| Pearson `r` | `pearson(x, y)` |
| Per-class matrices `R⁽ᶜ⁾` | `classCorrelations(X, y, d)` |
| `ΔRᵢⱼ` | `computeDeltaR(corrs)` |
| Top-k selection `Ωₖ` | `topK(dr, k, d)` |
| Feature enrichment | `enrich(X, pairs)` |
| Standardisation (z-score) | `fitNormalizer` + `applyNorm` |
| Logistic regression | `trainLR` (gradient descent) |
| Multi-class | `trainOvR` + `predictProbaOvR` (one-vs-rest) |
| Full pipeline on dataset load | `loadDataset(key)` |
| Live prediction from sliders | `runDemo()` |

### 4.2 Honesty notes built into the site (so you can defend it)

- **Iris is the real Fisher 1936 data.** **Wine and Breast Cancer are *seeded
  synthetic proxies*** — generated to reproduce the paper's correlation
  structure so the demo is meaningful, but they are *not* the full UCI sets.
  The site now says this clearly under the dataset picker and in each dataset's
  description. The **headline accuracies come from the full UCI datasets** and
  live on the Benchmark tab (they are taken from the paper, not computed live).
- **Standardisation matches the paper.** The site originally used min-max
  scaling; it now uses **z-score (StandardScaler)** to match Step 1. *Useful
  fact:* Pearson correlation — and therefore the whole ΔR matrix — is
  **unchanged** by this choice (correlation is invariant to positive linear
  rescaling), but the product terms and the logistic-regression boundary do
  change, so matching the paper keeps the live classifier faithful.
- **Multi-class probabilities** are a softmax over one-vs-rest logits — a
  reasonable illustration but *not* a perfectly calibrated posterior. The UI
  now says so.

### 4.3 Design

The interface uses a dark “PACEOS” theme. The palette was refreshed to a
**warm amber accent + teal secondary** (replacing the old lime/blue), with
larger type and bigger components for readability. All colours are driven by CSS
variables in `style.css` (`--accent`, `--blue` = teal, `--accent-rgb`, etc.), so
re-theming is a one-place change.

---

## Part 5 — Limitations — say these before the examiner does

Owning your limitations is the single biggest viva-score multiplier.

1. **Linear only.** ΔR is built on Pearson correlation, so it only detects
   *linear* correlation shifts. A pair whose relationship changes *nonlinearly*
   across classes (e.g. circular in one class, linear in another) has `ΔR ≈ 0`
   and is missed. Fix: a distance-correlation or rank-correlation version.
2. **Continuous features only.** Pearson is not meaningful for binary/categorical
   features. The planned student-survey study (mostly Likert/binary) will need
   **polychoric / point-biserial** correlations instead.
3. **Needs enough samples per class.** Correlation estimates from few samples
   are noisy. Rule of thumb: at least `2d` samples per class.
4. **No significance control.** With `C(d,2)` pairs ranked from finite data,
   some high ΔR values are flukes. There is no permutation test / FDR control,
   so spurious pairs can be picked on small datasets.
5. **Small margins, no significance test.** Over LR and SVM the accuracy gains
   are tiny and within one standard deviation; no paired test (McNemar /
   Wilcoxon) was run, so call them “competitive”, not “better”.
6. **`k` must be tuned** by cross-validation (though results are robust across a
   wide range of `k`).
7. **Not closed-form end-to-end.** The *screening* (Steps 1–4) is closed-form;
   the final logistic regression is still solved iteratively. The title's
   “closed-form” refers to the screening criterion.
8. **Low-dimensional data gains little.** On Iris (`d=4`, only 6 pairs) there is
   no room for *sparse* screening — `k=5` is almost the full expansion, and LDA
   wins.

---

## Part 6 — Viva preparation

Rehearse these out loud. Each answer is short on purpose; expand if pushed.

**Q1. In one sentence, what is your contribution?**
A closed-form criterion (ΔR) that ranks feature pairs by how much their
correlation changes across classes, used to add only the most useful product
features before logistic regression — fixing Naive Bayes's blind spot cheaply
and interpretably.

**Q2. What exactly does Naive Bayes assume, precisely?**
That features are *conditionally independent given the class*: the
class-conditional joint density factorises into per-feature densities. For
Gaussian NB that is equivalent to assuming zero class-conditional correlation.
Zero correlation is necessary but not sufficient for independence in general.

**Q3. Why correlation *difference* across classes, not just correlation?**
A correlation that is the same in every class shifts all class scores equally
and doesn't move the decision boundary — NB is harmlessly wrong about it. Only a
correlation that *differs* across classes distorts the boundary. ΔR targets
exactly those.

**Q4. Write the ΔR formula and state its range.**
`ΔRᵢⱼ = 2/(C(C−1)) · Σ_{m<n} |R⁽ᵐ⁾ᵢⱼ − R⁽ⁿ⁾ᵢⱼ|`, range `[0, 2]`. It is the mean
absolute difference of the pair's class-conditional Pearson correlations over
all class pairs.

**Q5. How does a linear model get a curved boundary?**
By adding product features `xᵢ·xⱼ`. The boundary is linear in the enriched space
`x̃` but quadratic in the original space, because the products are nonlinear in
the original features.

**Q6. Why not just add all pairwise products (full quadratic expansion)?**
Too many features (`C(d,2)`), causing multicollinearity and overfitting. For
Breast Cancer that's 435 extra features vs CS-LLN's ~45. ΔR keeps the few that
matter — “sparse” selection.

**Q7. Is it really “closed-form”?**
The *screening* (correlations + ΔR + top-k) is closed-form: one pass, no
gradients. The final logistic regression is iterative. So “closed-form” scopes
to the pair-selection criterion, which is the novel part.

**Q8. Why is the name “…Network” if there's no neural network?**
“Log-linear” describes logistic regression's decision form (a log-linear model).
There is no network architecture; the name refers to the log-linear structure,
not layers. (Fair criticism; an alternative name like “Log-Linear Classifier”
would be defensible.)

**Q9. Why does LDA beat you on Iris?**
Iris has only 4 features → 6 possible pairs, so sparse interaction screening has
no room to help; `k=5` is almost the full expansion. LDA's single pooled-Gaussian
covariance fits this clean, low-dimensional data well. We include Iris to show
CS-LLN doesn't *degrade* there, not to win.

**Q10. Your accuracy is only 0.35 pp above logistic regression — is that real?**
Honestly, the margin over LR/SVM is within one standard deviation and we did not
run a paired significance test, so I'd call it *competitive* with LR/SVM. The
clear, large, designed win is over Naive Bayes (4.75 pp, 63% fewer errors,
~10× better log-loss).

**Q11. The “~10× log-loss improvement” — over what?**
Over Gaussian NB (0.78 → 0.078). Against LR the calibration is about equal
(~0.077 vs 0.076). The gain is over the independence assumption, not over
discriminative baselines.

**Q12. Biggest weakness of ΔR?**
It only sees *linear* dependence (Pearson). Class-specific *nonlinear*
dependence has ΔR ≈ 0 and is missed. A distance/rank-correlation version would
fix this.

**Q13. How would you make pair selection statistically rigorous?**
Permute the class labels many times to build a null distribution for ΔR, keep
only pairs above (say) the 95th percentile, and apply false-discovery-rate
control across the `C(d,2)` pairs.

**Q14. How do you avoid data leakage?**
All fitting — standardisation, correlations, ΔR, and `k` selection — happens on
the *training fold only*, inside each cross-validation split. Test statistics
never touch training.

**Q15. Complexity?**
Correlations O(Nd²); sort O(d² log d); LR training O(N(d+k)·iters); inference
O((d+k)·C). The expensive screening part is non-iterative.

**Q16. How is this different from Factorization Machines / TAN / AODE?**
FMs model *all* pairs via learned embeddings, need SGD, and are hard to
interpret. TAN allows one parent per feature (a tree). AODE averages many
one-parent models (memory-heavy). CS-LLN instead *explicitly ranks* a *sparse*
set of pairs by a closed-form score and adds them as readable product terms.

**Q17. Where did the ΔR idea come from?**
Differential co-expression analysis in bioinformatics, which flags gene pairs
whose correlation differs between healthy and diseased tissue. CS-LLN
generalises that principle into an interaction-screening filter for supervised
classification.

**Q18. What's next?**
Polychoric correlations for ordinal/binary features (for the student-performance
study), a nonlinear (distance-correlation) ΔR, significance/FDR control, and
generalisation bounds for the enriched space.

---

## Part 7 — Two pitches

**30-second version:**
“Naive Bayes ignores feature correlations. But only correlations that *change
across classes* hurt the decision boundary. I score every feature pair by that
change — a one-pass statistic called ΔR — add product features for just the
top-k pairs, and train logistic regression. It fixes Naive Bayes's blind spot
cheaply and stays interpretable: it beats NB by 63% fewer errors on Breast
Cancer and wins 2 of 3 UCI benchmarks.”

**3-minute version:** walk through Part 2.3 (the two ways NB is wrong), then the
five steps in Part 3.4, then the results and the honest limitations in Part 5.

---

### Glossary (quick reference)

- **Conditional independence** — independent once the class is known.
- **Likelihood** `P(x|y)` — chance of the features given a class.
- **Pearson `r`** — linear correlation, in `[−1, 1]`.
- **ΔR (context shift)** — mean absolute change of a pair's correlation across
  classes; the paper's ranking score.
- **Enrichment** — appending product terms `xᵢ·xⱼ` to the feature vector.
- **Ωₖ** — the set of top-k selected pairs.
- **L2 regularisation** — penalty on large weights that improves stability.
- **Multicollinearity** — redundant, overlapping features that destabilise a
  linear model.
- **Log-loss** — penalises confident wrong predictions; lower = better
  calibrated probabilities.
- **Stratified k-fold CV** — split data into k parts keeping class ratios,
  train on k−1, test on 1, rotate, average.
