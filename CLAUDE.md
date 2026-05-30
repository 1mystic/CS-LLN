# CS-LLN Inspector — Project Context

## What This Project Is

A standalone, single-file-per-concern web app (`index.html` + `style.css` + `app.js` + `data.js`) that acts as an interactive research companion for the paper:

> **"Context-Shift Interaction Screening for Feature-Enriched Classification: A Sparse, Closed-Form Approach Beyond the Naive Independence Assumption"**
> — Atharv Khare, CSE, LNCT Bhopal, India

The site lets users select one of three UCI datasets, drag feature sliders, and watch CS-LLN classify in real time — with the ΔRij heatmap, top interaction pairs, and enriched feature space updating live. Everything runs in the browser with no server.

---

## The Paper — CS-LLN Algorithm

### Core Problem

Naive Bayes assumes all features are conditionally independent given the class. In practice, feature pairs can have **class-specific correlations** — high in one class, near zero in another. NB systematically misestimates posteriors for these pairs.

### The CS-LLN Solution (5 steps, Algorithm 1)

**Step 1 — Standardise**
Zero-mean, unit-variance each feature using training-fold statistics only. Prevents leakage.

**Step 2 — Class-Conditional Pearson Correlations**
For each class `c`, compute a `d×d` correlation matrix `R(c)` where `R(c)_ij = Pearson(X_c,i , X_c,j)`.

**Step 3 — Context-Shift Scoring ΔRij**
```
ΔRij = (2 / C(C-1)) × Σ_{m<n} |R(m)_ij − R(n)_ij|
```
A high ΔR means the pair's relationship is class-specific — exactly what NB fails to see.

**Step 4 — Sparse Pair Selection Ωk**
Rank all `C(d,2)` pairs by ΔR descending, select top-k. For BC: k=45 of 435 pairs used — 6.2× fewer than full quadratic expansion.

**Step 5 — Feature Enrichment + L2-LR**
Append product terms `xi·xj` for each selected pair to get enriched vector `x̃ ∈ ℝ^(d+k)`. Train L2-regularised logistic regression on `x̃`. Decision boundary is linear in `x̃` but quadratic in original space `X`.

### Key Insight

A pair with `ΔRij = 0` is one NB is wrong about *consistently* — no bias. A pair with `ΔRij >> 0` is one NB is wrong about in a *class-specific* way — systematic distortion. CS-LLN targets precisely those pairs.

### Benchmark Results (5-fold stratified CV)

| Dataset | N | d | C | k | CS-LLN Acc | Best Baseline | Notes |
|---|---|---|---|---|---|---|---|
| Breast Cancer Wisconsin | 569 | 30 | 2 | 45 | **97.72% ± 1.6** | LR 97.37% | 63% error ↓ vs GNB; log-loss ~10× better (0.777→0.078) |
| Wine Recognition | 178 | 13 | 3 | 5 | **98.89% ± 1.4** | LR 98.33% | k=1 alone achieves 99.44% |
| Iris | 150 | 4 | 3 | 5 | 96.67% ± 4.2 | LDA 97.33% | d=4 limits interaction benefit |

CS-LLN wins on 2/3 UCI benchmarks. LDA wins Iris because d is too small for sparse interactions to add value.

### Complexity

| Step | Big-O |
|---|---|
| Class correlations | O(Nd²) |
| Sort ΔR | O(d² log d) |
| LR training | O(N(d+k)·iter) |
| Inference / sample | O((d+k)·C) |

No gradient computation needed for the screening phase — it is entirely closed-form.

### Limitations (paper-stated)

- Pearson correlation requires continuous features (no categorical)
- Needs `Nc ≥ 2d` samples per class for stable correlation estimates
- Hyperparameter k requires cross-validation
- Future work: polychoric correlations for ordinal/Likert data
- Planned extension: student academic performance dataset (20 features, 3 CGPA-band classes, 200–400 survey responses from LNCT Bhopal)

---

## Site Architecture

### Files

| File | Role |
|---|---|
| `index.html` | 5-page SPA shell, all static HTML |
| `style.css` | PACEOS design system, all visual tokens |
| `data.js` | Three UCI datasets + `window.DATASET_REGISTRY` |
| `app.js` | Full CS-LLN pipeline + all rendering logic |

**Load order matters:** `data.js` must load before `app.js`.

```html
<script src="data.js"></script>
<script src="app.js"></script>
```

### 5 Pages (SPA — JS switches `.active` class)

| Nav label | `data-page` id | Content |
|---|---|---|
| LIVE INSPECTOR | `inspector` | Dataset selector + live sliders + CS-LLN inference |
| ΔR MATRIX | `matrix` | Pre-baked 12×12 BC heatmap, top-10 pairs, k-ablation canvas |
| BENCHMARK | `benchmark` | Tables I-III from paper, accuracy bar charts |
| PIPELINE | `pipeline` | 5-step algorithm cards, complexity table, comparisons |
| ABOUT | `about` | Paper metadata, abstract, key results, limitations |

Page switching: `setPage(id)` in `app.js` — no routing library.

---

## data.js — Datasets

### IRIS (exact Fisher 1936, 150×4)
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width (all cm)
- Classes: 0=Setosa, 1=Versicolor, 2=Virginica (50 each)
- `IRIS_X` is a hardcoded literal array (all 150 exact rows)
- `IRIS_Y = [...Array(50).fill(0), ...Array(50).fill(1), ...Array(50).fill(2)]`

### WINE (seeded-deterministic, 180×13)
- Uses Mulberry32 PRNG (`mkrng(0xDEADBEEF)`) + Box-Muller Gaussian (`gauss()`)
- 60 samples per class, matches UCI per-class means/stds
- Engineered within-class correlations for meaningful ΔRij:
  - Class 1: `flavanoids × total_phenols` r≈0.87 (high co-variance)
  - Class 3: `malic_acid × alcalinity` r≈0.73, `color_intensity × malic_acid` r≈0.70
  - Class 2: weak correlations (independent-ish)
- `window.WINE_X`, `window.WINE_Y` (built by IIFE)

### BREAST CANCER (seeded-deterministic, 120×10 top features)
- Uses Mulberry32 PRNG (`mkrng(0xCAFEBABE)`)
- Top 10 features from paper: Worst Radius, Worst Area, Worst Perimeter, Mean Area, Mean Radius, Mean Perimeter, Radius Error, Perimeter Error, Worst Concavity, Mean Concavity
- 60 Malignant + 60 Benign samples
- Engineered geometric co-elevation (malignant): shared `geoFactor` drives worst_rad, worst_area, worst_perim together (r≈0.88-0.92)
- Benign: weaker shared factor (r≈0.47-0.58)
- This produces real ΔRij signal matching paper's Fig. 2
- `window.BC_X`, `window.BC_Y` (built by IIFE)

### DATASET_REGISTRY (`window.DATASET_REGISTRY`)
```javascript
{
  iris:         { key, label, subLabel, description, features[4], units[4], classes[3], classColors[3], X, y, defaultK:3, bestResult },
  wine:         { key, label, subLabel, description, features[13], units[13], classes[3], classColors[3], X (getter), y (getter), defaultK:5, bestResult },
  breast_cancer:{ key, label, subLabel, description, features[10], units[10], classes[2], classColors[2], X (getter), y (getter), defaultK:4, bestResult }
}
```
`classColors` uses CSS vars: `var(--accent)`, `var(--blue)`, `var(--purple)` / `var(--danger)`.

---

## app.js — Algorithm Implementation

### Math Functions

```javascript
mean(arr)                           // arithmetic mean
pearson(x, y)                       // Pearson r, returns 0 if zero-variance
classCorrelations(X, y, d)          // returns C matrices of shape d×d
computeDeltaR(corrs)                // returns d×d ΔRij matrix (upper=lower, diag=0)
topK(dr, k, d)                      // returns sorted [{i,j,score}] array length k
enrich(X, pairs)                    // appends xi·xj products to each row
sigmoid(z)
trainLR(X, y, lambda=0.5, lr=0.08, iters=1500)   // returns {w, b}
```

### Normalisation
```javascript
fitNormalizer(X)          // returns {mins, maxs} — min-max from full dataset
applyNorm(row, norm)      // maps each feature to [0,1]
normalizeX(X, norm)       // applies applyNorm to every row
```

### Multi-class (One-vs-Rest)
```javascript
trainOvR(Xe, y)           // trains C binary models, returns [{w,b}, ...]
predictProbaOvR(x, models)// softmax over raw OvR logits → probability vector length C
```
Works for both binary (BC) and 3-class (Iris, Wine).

### Inspector State Globals
```javascript
let currentDatasetKey = 'iris';
let currentDataset    = null;   // ref to DATASET_REGISTRY entry
let normalizer        = null;   // {mins, maxs}
let DEMO_K            = 3;      // current k (updated by k-slider)
let CORRS, DELTA_R, SELECTED_PAIRS, LR_MODELS;
```

### Key Functions

```javascript
loadDataset(key)         // full pipeline: normalize → correlations → ΔR → topK → OvR train → render UI
updateDatasetUI()        // updates all static DOM text/badges for current dataset
renderDatasetSelector()  // renders 3 clickable dataset buttons
renderDynamicSliders()   // generates slider HTML per feature, wires input events
runDemo()                // reads slider values → normalise → enrich → predict → render all 4 panels
```

### Rendering Functions

| Function | Target element | Description |
|---|---|---|
| `renderMiniHeatmap(dr)` | `#mini-heatmap` | Top-8 features by ΔR sum, accent=selected pairs |
| `renderTopPairs(pairs)` | `#top-pairs-live` | Top-6 pairs with dot-bar and SELECTED badge |
| `renderEnrichedFeatures(raw, norm, pairs)` | `#enrich-display` | Original chips + interaction term chips |
| `renderResult(probs, raw, norm)` | `#class-result` | Class label, confidence %, probability bars, key signal |
| `renderHeatmap()` | `#bc-heatmap` | Pre-baked 12×12 BC ΔRij heatmap (matrix page) |
| `renderTopPairsBC()` | `#top-pairs-bc` | Top-10 paper pairs with dot-bars |
| `renderKChart()` | `#k-chart` (canvas) | k-ablation accuracy curve (Fig. 4 from paper) |
| `renderBenchmark()` | `#bench-0/1/2` | Tables I-III with best-row highlighting |
| `renderAccuracyBars()` | `#acc-bars` | 3-column horizontal bar chart |

### Pre-baked Data (Matrix + Benchmark pages)

These are hardcoded in `app.js` from the paper figures (no computation needed):
- `BC_FEATURES[12]` — 12 feature labels for the BC heatmap
- `BC_DELTA_R[12][12]` — ΔRij values from paper Fig. 2
- `TOP_PAIRS_BC[10]` — top-10 pairs from paper Fig. 5
- `BENCHMARK` — Tables I, II, III with all model accuracies
- `K_BREAST[12]` — k-ablation curve from paper Fig. 4

### Init Flow (DOMContentLoaded)

1. Wire nav buttons → `setPage()`
2. Wire k-slider → retrain OvR + `runDemo()`
3. `loadDataset('iris')` — trains on full Iris data, renders all inspector panels
4. `setPage('inspector')` — activates inspector tab

---

## style.css — PACEOS Design System

### Design tokens (CSS variables)
```css
--bg:           #0F0F0F      /* page background */
--bg-card:      #161616      /* sidebar, card background */
--bg-card-2:    #1D1D1D      /* secondary card / hover state */
--border:       #2A2A2A
--border-light: #333333
--text-primary: #FFFFFF
--text-secondary:#D0D0D0
--text-muted:   #888888
--accent:       #E0FF53      /* yellow-green — CS-LLN brand color */
--danger:       #F14336      /* red */
--blue:         #3B82F6
--purple:       #A855F7
--font-mono:    'Space Mono', monospace
--font-ui:      'DM Sans', sans-serif
```

### Layout
```
.shell { display: flex; height: 100vh; overflow: hidden; }
  └── .sidebar (220px fixed, flex column)
  └── .main (flex 1, overflow-y: auto, padding 32px 28px)
```

### Key Component Classes

| Class | Description |
|---|---|
| `.page` | `display:none` by default |
| `.page.active` | `display:flex; flex-direction:column; gap:16px;` with `fadeUp` animation |
| `.card` | Dark card with border, padding 20px |
| `.card--flush` | Card with no padding (content fills to edge) |
| `.stat-row`, `.stat-row--3`, `.stat-row--4` | Grid of stat cards |
| `.two-col` | 2-column layout (55/45 split) |
| `.three-col` | 3-column equal layout |
| `.panel-header` | Flex row with title + badge, border-bottom |
| `.panel-title` | Mono, 10px, letter-spacing, muted |
| `.k-row` | Horizontal row: label + value display + range slider + info |
| `.slider-row` | Label/value header + `.slider-wrap` (custom track + thumb) |
| `.heatmap-cell` | Sized div with hover tooltip support |
| `.heatmap-label` | Mono, tiny, muted label for heatmap axis |
| `.pair-row` | Flex row: pair names + dot-bar + score + badge |
| `.enrich-chip` | Pill: feature name + value |
| `.enrich-chip--new` | Accent-bordered pill for interaction terms |
| `.result-class` | Large bold classification label |
| `.result-bar-row` | Label + track + percentage for probability bar |
| `.cmp-table` | Benchmark comparison table, `tr.best` row highlighted |
| `.pipeline-step` | Numbered card with step formula |
| `.about-row` | Key/value row with optional accent value |
| `.badge--accent` | Yellow-green pill badge |
| `.badge--muted` | Grey pill badge |
| `.badge--blue` | Blue pill badge |
| `.badge--purple` | Purple pill badge |
| `.dot`, `.dot--on`, `.dot--blue` | Dot-bar building blocks |
| `.hm-tooltip` | Fixed tooltip for BC heatmap hover |

---

## Dynamic Inspector Page — HTML IDs

All these IDs are updated by JS when the dataset changes:

| ID | Updated by | Content |
|---|---|---|
| `#ds-badge` | `updateDatasetUI` | Dataset label (e.g. "IRIS") |
| `#ds-desc` | `updateDatasetUI` | Dataset description string |
| `#dataset-selector` | `renderDatasetSelector` | 3 dataset picker buttons |
| `#stat-features` | `updateDatasetUI` | Feature count d |
| `#stat-features-sub` | `updateDatasetUI` | Abbreviated feature names |
| `#stat-pairs` | `updateDatasetUI` / k-slider | Current k |
| `#stat-dim` | `updateDatasetUI` / k-slider | d + k |
| `#k-val`, `#k-val-2` | `updateDatasetUI` / k-slider | Current k display |
| `#k-max-pairs` | `updateDatasetUI` | C(d,2) max pairs |
| `#panel-features-title` | `updateDatasetUI` | "{DATASET} FEATURES" |
| `#mini-hm-size-badge` | `updateDatasetUI` | "N×N" heatmap size |
| `#feature-inputs` | `renderDynamicSliders` | Slider HTML (re-generated on dataset switch) |
| `#enrich-display` | `renderEnrichedFeatures` | Original + interaction chip panels |
| `#class-result` | `renderResult` | Classification output |
| `#mini-heatmap` | `renderMiniHeatmap` | Top-8 feature ΔRij grid |
| `#top-pairs-live` | `renderTopPairs` | Top-6 pair rows |

Dynamic slider IDs per feature `i`: `#sl-feat-i`, `#val-feat-i`, `#track-feat-i`, `#thumb-feat-i`

---

## Known Constraints & Design Decisions

- **No build step** — vanilla JS/HTML/CSS only. No modules, no bundler.
- **No CDN dependencies** — Google Fonts only. Everything else is inline.
- **file:// protocol warning** is harmless — browsers block same-origin XHR from `file://` but this app makes no network requests at runtime.
- **Training on full dataset** — for demo purposes, normalizer and models are fit on all available samples (no train/test split). This is intentional.
- **OvR multi-class** — logistic regression uses one-vs-rest with softmax over raw logits. Same code path for binary (BC) and 3-class (Iris, Wine).
- **Mini-heatmap limits to top-8 features** — computed by summing each feature's row in ΔRij, prevents overcrowding for Wine (d=13).
- **k-slider max updates dynamically** — `loadDataset` sets `kSlider.max = d*(d-1)/2`.
- **Retraining on k-change** — when user moves k-slider, OvR models are retrained synchronously (fast enough at these dataset sizes).
- **Iris defaultK=3**, Wine defaultK=5, BC defaultK=4.

---

## Planned / Future Work

- Student academic performance dataset (LNCT Bhopal survey, 20 features, 3 classes) — mentioned in paper, data not yet collected
- Polychoric correlations for ordinal features
- Mobile responsive layout (currently desktop-only, sidebar is fixed 220px)
- Per-dataset k-ablation charts (currently only BC's Fig. 4 is shown)
- Export / share classification result as URL params
