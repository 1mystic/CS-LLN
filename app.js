/* ─────────────────────────────────────────────
   CS-LLN Inspector — app.js
   ───────────────────────────────────────────── */

// ── Nav ──────────────────────────────────────
const pages = ['inspector', 'matrix', 'benchmark', 'pipeline', 'about'];
let currentPage = 'inspector';

function setPage(id) {
  currentPage = id;
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + id));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.page === id));
  if (id === 'matrix') renderHeatmap();
  if (id === 'benchmark') renderBenchmark();
  if (id === 'inspector') runDemo();
}

// ── Pre-baked Breast Cancer ΔRij data (Fig. 2) ──
const BC_FEATURES = [
  'radius err', 'smooth err', 'sym err', 'perim err',
  'mean perim', 'mean conc', 'mean area', 'mean radius',
  'worst perim', 'worst area', 'conc pts err', 'worst radius'
];

const BC_DELTA_R = [
  [0,    0.05, 0.20, 0.06, 0.66, 0.22, 0.70, 0.67, 0.65, 0.64, 0.02, 0.63],
  [0.05, 0,    0.19, 0.02, 0.60, 0.16, 0.60, 0.59, 0.51, 0.49, 0.26, 0.49],
  [0.20, 0.19, 0,    0.11, 0.48, 0.21, 0.45, 0.46, 0.43, 0.37, 0.21, 0.39],
  [0.06, 0.02, 0.11, 0,    0.62, 0.17, 0.66, 0.62, 0.56, 0.58, 0.04, 0.56],
  [0.66, 0.60, 0.48, 0.62, 0,    0.41, 0.00, 0.00, 0.05, 0.08, 0.20, 0.06],
  [0.22, 0.16, 0.21, 0.17, 0.41, 0,    0.40, 0.40, 0.35, 0.33, 0.33, 0.34],
  [0.70, 0.60, 0.45, 0.66, 0.00, 0.40, 0,    0.00, 0.04, 0.07, 0.21, 0.05],
  [0.67, 0.59, 0.46, 0.62, 0.00, 0.40, 0.00, 0,    0.05, 0.08, 0.21, 0.06],
  [0.65, 0.51, 0.43, 0.56, 0.05, 0.35, 0.04, 0.05, 0,    0.01, 0.14, 0.00],
  [0.64, 0.49, 0.37, 0.58, 0.08, 0.33, 0.07, 0.08, 0.01, 0,    0.13, 0.01],
  [0.02, 0.26, 0.21, 0.04, 0.20, 0.33, 0.21, 0.21, 0.14, 0.13, 0,    0.13],
  [0.63, 0.49, 0.39, 0.56, 0.06, 0.34, 0.05, 0.06, 0.00, 0.01, 0.13, 0 ]
];

const TOP_PAIRS_BC = [
  { pair: 'mean area × radius error',         score: 0.700 },
  { pair: 'mean radius × radius error',        score: 0.667 },
  { pair: 'mean perimeter × radius error',     score: 0.665 },
  { pair: 'mean area × perimeter error',       score: 0.656 },
  { pair: 'radius error × worst perimeter',    score: 0.647 },
  { pair: 'radius error × worst area',         score: 0.642 },
  { pair: 'radius error × worst radius',       score: 0.627 },
  { pair: 'mean radius × perimeter error',     score: 0.624 },
  { pair: 'mean perimeter × perimeter error',  score: 0.620 },
  { pair: 'mean perimeter × smoothness error', score: 0.598 },
];

const BENCHMARK = {
  'Breast Cancer\nWisconsin': {
    note: 'N=569, d=30, C=2  ·  k=45 / 435 pairs used',
    best: 'CS-LLN',
    rows: [
      { model: 'Gaussian NB',    acc: 92.97, std: 2.0, loss: 0.7768, f1: 0.9244 },
      { model: 'LDA',            acc: 95.61, std: 2.0, loss: 0.1264, f1: 0.9517 },
      { model: 'Logistic Reg.',  acc: 97.37, std: 1.7, loss: 0.0764, f1: 0.9714 },
      { model: 'Linear SVM',     acc: 97.37, std: 2.2, loss: 0.0839, f1: 0.9715 },
      { model: 'CS-LLN (k=45)', acc: 97.72, std: 1.6, loss: 0.0779, f1: 0.9755 },
    ]
  },
  'Wine Recognition': {
    note: 'N=178, d=13, C=3  ·  k=5 / 78 pairs used',
    best: 'CS-LLN',
    rows: [
      { model: 'Gaussian NB',   acc: 97.75, std: 2.8, loss: 0.0969, f1: 0.9788 },
      { model: 'Linear SVM',    acc: 96.63, std: 1.1, loss: 0.1111, f1: 0.9648 },
      { model: 'LDA',           acc: 98.30, std: 1.4, loss: 0.0332, f1: 0.9834 },
      { model: 'Logistic Reg.', acc: 98.33, std: 1.4, loss: 0.0611, f1: 0.9829 },
      { model: 'CS-LLN (k=5)', acc: 98.89, std: 1.4, loss: 0.0581, f1: 0.9891 },
    ]
  },
  'Iris': {
    note: 'N=150, d=4, C=3  ·  k=5 (limited benefit, d=4)',
    best: 'LDA',
    rows: [
      { model: 'Gaussian NB',   acc: 94.67, std: 4.0, loss: 0.1370, f1: 0.9465 },
      { model: 'Logistic Reg.', acc: 95.33, std: 4.5, loss: 0.1556, f1: 0.9532 },
      { model: 'CS-LLN (k=5)', acc: 96.67, std: 4.2, loss: 0.1037, f1: 0.9666 },
      { model: 'Linear SVM',    acc: 96.67, std: 5.2, loss: 0.1131, f1: 0.9664 },
      { model: 'LDA',           acc: 97.33, std: 3.9, loss: 0.0618, f1: 0.9733 },
    ]
  }
};

const K_BREAST = [
  {k:1,acc:97.37},{k:5,acc:97.50},{k:10,acc:97.58},{k:15,acc:97.62},
  {k:20,acc:97.66},{k:30,acc:97.70},{k:40,acc:97.71},{k:45,acc:97.72},
  {k:50,acc:97.65},{k:60,acc:97.50},{k:70,acc:97.20},{k:78,acc:96.49}
];

// ── CS-LLN Math ───────────────────────────────

function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function pearson(x, y) {
  const mx = mean(x), my = mean(y);
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < x.length; i++) {
    const a = x[i] - mx, b = y[i] - my;
    num += a * b; dx2 += a * a; dy2 += b * b;
  }
  if (dx2 === 0 || dy2 === 0) return 0;
  return num / Math.sqrt(dx2 * dy2);
}

function classCorrelations(X, y, d) {
  const classes = [...new Set(y)].sort((a, b) => a - b);
  return classes.map(c => {
    const Xc = X.filter((_, i) => y[i] === c);
    return Array.from({length: d}, (_, i) =>
      Array.from({length: d}, (_, j) =>
        pearson(Xc.map(r => r[i]), Xc.map(r => r[j]))
      )
    );
  });
}

function computeDeltaR(corrs) {
  const C = corrs.length, d = corrs[0].length;
  const dr = Array.from({length: d}, () => Array(d).fill(0));
  let n = 0;
  for (let m = 0; m < C; m++) for (let p = m + 1; p < C; p++, n++) {
    for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) {
      dr[i][j] += Math.abs(corrs[m][i][j] - corrs[p][i][j]);
    }
  }
  if (n) for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) {
    dr[i][j] /= n;
    dr[j][i] = dr[i][j];
  }
  return dr;
}

function topK(dr, k, d) {
  const pairs = [];
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++)
    pairs.push({ i, j, score: dr[i][j] });
  pairs.sort((a, b) => b.score - a.score);
  return pairs.slice(0, k);
}

function enrich(X, pairs) {
  return X.map(row => [
    ...row,
    ...pairs.map(({ i, j }) => row[i] * row[j])
  ]);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function trainLR(X, y, lambda = 0.5, lr = 0.08, iters = 1500) {
  const d = X[0].length, n = X.length;
  const w = new Array(d).fill(0);
  let b = 0;
  for (let it = 0; it < iters; it++) {
    const dw = new Array(d).fill(0);
    let db = 0;
    for (let i = 0; i < n; i++) {
      const z = w.reduce((s, wi, j) => s + wi * X[i][j], b);
      const err = sigmoid(z) - y[i];
      for (let j = 0; j < d; j++) dw[j] += err * X[i][j];
      db += err;
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (dw[j] / n + lambda * w[j]);
    b -= lr * db / n;
  }
  return { w, b };
}

// ── Normalisation ─────────────────────────────

// Fits BOTH the z-score parameters used by the model (means/stds — matches the
// paper's StandardScaler, Algorithm 1 Step 1) AND the raw min/max ranges used
// only to lay out the sliders. Keeping both on one object avoids a second pass.
function fitNormalizer(X) {
  const d = X[0].length, n = X.length;
  const mins = Array(d).fill(Infinity);
  const maxs = Array(d).fill(-Infinity);
  const sums = Array(d).fill(0);
  X.forEach(row => row.forEach((v, j) => {
    if (v < mins[j]) mins[j] = v;
    if (v > maxs[j]) maxs[j] = v;
    sums[j] += v;
  }));
  const means = sums.map(s => s / n);
  const sq = Array(d).fill(0);
  X.forEach(row => row.forEach((v, j) => { const dv = v - means[j]; sq[j] += dv * dv; }));
  // Population std; guard against a constant feature (std = 0) → leave scale at 1.
  const stds = sq.map(s => { const sd = Math.sqrt(s / n); return sd > 1e-12 ? sd : 1; });
  return { mins, maxs, means, stds };
}

// Standardise to zero mean / unit variance (z-score), matching the paper.
// Pearson correlations — and therefore the whole ΔR matrix — are invariant to
// this choice, but the product interaction terms and the LR boundary are not,
// so this keeps the live classifier faithful to the paper's feature space.
function applyNorm(row, norm) {
  return row.map((v, j) => (v - norm.means[j]) / norm.stds[j]);
}

function normalizeX(X, norm) {
  return X.map(row => applyNorm(row, norm));
}

// ── Multi-class One-vs-Rest LR ────────────────

function trainOvR(Xe, y, lambda = 0.5, lr = 0.08, iters = 1500) {
  const classes = [...new Set(y)].sort((a, b) => a - b);
  return classes.map(c => trainLR(Xe, y.map(yi => yi === c ? 1 : 0), lambda, lr, iters));
}

function predictProbaOvR(x, models) {
  const logits = models.map(({ w, b }) => w.reduce((s, wi, j) => s + wi * x[j], b));
  const maxL = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ── Inspector State ───────────────────────────

let currentDatasetKey = 'iris';
let currentDataset = null;
let normalizer = null;
let DEMO_K = 3;
let CORRS, DELTA_R, SELECTED_PAIRS, LR_MODELS;

// ── Dataset loading ───────────────────────────

function loadDataset(key) {
  if (!window.DATASET_REGISTRY) {
    console.error('data.js not loaded');
    return;
  }
  currentDatasetKey = key;
  currentDataset = window.DATASET_REGISTRY[key];
  const X = currentDataset.X;
  const y = currentDataset.y;
  const d = currentDataset.features.length;
  const maxPairs = d * (d - 1) / 2;

  normalizer = fitNormalizer(X);
  const Xnorm = normalizeX(X, normalizer);

  CORRS = classCorrelations(Xnorm, y, d);
  DELTA_R = computeDeltaR(CORRS);

  DEMO_K = Math.min(currentDataset.defaultK, maxPairs);
  SELECTED_PAIRS = topK(DELTA_R, DEMO_K, d);

  const Xe = enrich(Xnorm, SELECTED_PAIRS);
  LR_MODELS = trainOvR(Xe, y);

  const kSlider = document.getElementById('k-slider');
  if (kSlider) { kSlider.max = maxPairs; kSlider.value = DEMO_K; }

  updateDatasetUI();
  renderDatasetSelector();
  renderDynamicSliders();
}

function updateDatasetUI() {
  const ds = currentDataset;
  const d = ds.features.length;
  const maxPairs = d * (d - 1) / 2;
  const showN = Math.min(d, 8);

  const $ = id => document.getElementById(id);
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };

  set('ds-badge', ds.label);
  const desc = $('ds-desc');
  if (desc) desc.textContent = ds.description;

  set('stat-features', d);
  set('stat-features-sub', ds.features.slice(0, 3).map(f => f.split(' ')[0].toUpperCase().substring(0, 4)).join('·') + (d > 3 ? '…' : ''));
  set('stat-pairs', DEMO_K);
  set('stat-dim', d + DEMO_K);
  set('k-val', DEMO_K);
  set('k-val-2', DEMO_K);
  set('k-max-pairs', maxPairs);
  set('panel-features-title', ds.label + ' FEATURES');
  set('mini-hm-size-badge', showN + '×' + showN);
}

function renderDatasetSelector() {
  const el = document.getElementById('dataset-selector');
  if (!el || !window.DATASET_REGISTRY) return;
  el.innerHTML = ['iris', 'wine', 'breast_cancer'].map(key => {
    const ds = window.DATASET_REGISTRY[key];
    const on = key === currentDatasetKey;
    return `<button onclick="loadDataset('${key}')" style="
      font-family:var(--font-mono);font-size:12px;font-weight:700;letter-spacing:0.06em;
      padding:10px 20px;border-radius:8px;cursor:pointer;line-height:1.5;
      border:1px solid ${on ? 'var(--accent)' : 'var(--border)'};
      background:${on ? 'rgba(var(--accent-rgb),0.12)' : 'var(--bg-card-2)'};
      color:${on ? 'var(--accent)' : 'var(--text-muted)'};transition:all 150ms;
    ">${ds.label}<br><span style="font-size:10px;font-weight:400;opacity:0.7;">${ds.subLabel}</span></button>`;
  }).join('');
}

// ── Dynamic Sliders ───────────────────────────

function sliderStep(lo, hi) {
  const r = hi - lo;
  if (r < 1) return 0.01;
  if (r < 10) return 0.1;
  if (r < 100) return 1;
  return 10;
}

function fmtVal(v) {
  return v >= 100 ? Math.round(v).toString() : v.toFixed(2);
}

function renderDynamicSliders() {
  const el = document.getElementById('feature-inputs');
  if (!el || !currentDataset || !normalizer) return;
  const { features, units } = currentDataset;
  const { mins, maxs } = normalizer;
  const X = currentDataset.X;
  const d = features.length;

  const sums = Array(d).fill(0);
  X.forEach(row => row.forEach((v, j) => { sums[j] += v; }));
  const means = sums.map(s => s / X.length);

  let html = '';
  features.forEach((name, i) => {
    const lo = mins[i], hi = maxs[i];
    const def = means[i];
    const step = sliderStep(lo, hi);
    const pct = ((def - lo) / (hi - lo) * 100).toFixed(1);
    const unitStr = units[i] ? ' (' + units[i] + ')' : '';
    html += `<div class="slider-row">
      <div class="slider-row__header">
        <span class="slider-row__label">${name.toUpperCase()}${unitStr}</span>
        <span class="slider-row__value" id="val-feat-${i}">${fmtVal(def)}</span>
      </div>
      <div class="slider-wrap">
        <input type="range" id="sl-feat-${i}" min="${lo}" max="${hi}" step="${step}" value="${def}" />
        <div class="slider-track" id="track-feat-${i}" style="width:${pct}%"></div>
        <div class="slider-thumb" id="thumb-feat-${i}" style="left:${pct}%"></div>
      </div>
    </div>`;
  });
  el.innerHTML = html;

  features.forEach((_, i) => {
    const sl = document.getElementById('sl-feat-' + i);
    if (!sl) return;
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      const lo = parseFloat(sl.min), hi = parseFloat(sl.max);
      const vl = document.getElementById('val-feat-' + i);
      const track = document.getElementById('track-feat-' + i);
      const thumb = document.getElementById('thumb-feat-' + i);
      if (vl) vl.textContent = fmtVal(v);
      const pct = (v - lo) / (hi - lo) * 100;
      if (track) track.style.width = pct + '%';
      if (thumb) thumb.style.left = pct + '%';
      runDemo();
    });
  });
}

// ── Run demo ──────────────────────────────────

function runDemo() {
  if (!currentDataset || !normalizer || !DELTA_R || !LR_MODELS) return;
  const d = currentDataset.features.length;

  const rawFeatures = Array.from({ length: d }, (_, i) => {
    const el = document.getElementById('sl-feat-' + i);
    return el ? parseFloat(el.value) : (normalizer.mins[i] + normalizer.maxs[i]) / 2;
  });

  const normFeatures = applyNorm(rawFeatures, normalizer);
  const xe = enrich([normFeatures], SELECTED_PAIRS)[0];
  const probs = predictProbaOvR(xe, LR_MODELS);

  renderMiniHeatmap(DELTA_R);
  renderTopPairs(SELECTED_PAIRS);
  renderEnrichedFeatures(rawFeatures, normFeatures, SELECTED_PAIRS);
  renderResult(probs, rawFeatures, normFeatures);
}

// ── Mini heatmap (inspector) ──────────────────

function renderMiniHeatmap(dr) {
  const container = document.getElementById('mini-heatmap');
  if (!container || !currentDataset) return;
  const features = currentDataset.features;
  const d = features.length;
  const maxShow = Math.min(d, 8);

  // Rank features by total ΔR contribution
  const fScores = Array.from({ length: d }, (_, i) => ({
    idx: i,
    total: dr[i].reduce((s, v, j) => i === j ? s : s + v, 0)
  }));
  fScores.sort((a, b) => b.total - a.total);
  const topF = fScores.slice(0, maxShow).map(f => f.idx).sort((a, b) => a - b);

  const labels = topF.map(fi => features[fi].split(' ')[0].substring(0, 5).toUpperCase());
  const showDr = topF.map(fi => topF.map(fj => dr[fi][fj]));
  const showFeat = topF.map(fi => features[fi]);

  const cw = maxShow <= 5 ? 52 : 42;
  const lw = maxShow <= 5 ? 70 : 58;
  const fs = maxShow <= 5 ? 11 : 9;

  let html = `<div style="display:grid;grid-template-columns:${lw}px repeat(${maxShow},${cw}px);gap:2px;align-items:center;">`;
  html += '<div></div>';
  labels.forEach(l => {
    html += `<div class="heatmap-label" style="text-align:center;padding:2px 0;font-size:${fs}px;">${l}</div>`;
  });
  for (let i = 0; i < maxShow; i++) {
    html += `<div class="heatmap-label" style="text-align:right;padding-right:4px;font-size:${fs}px;">${labels[i]}</div>`;
    for (let j = 0; j < maxShow; j++) {
      const v = showDr[i][j];
      if (i === j) {
        html += `<div class="heatmap-cell" style="background:rgba(255,255,255,0.05);color:var(--text-muted);width:${cw}px;height:38px;font-size:${fs}px;" title="${showFeat[i]}">—</div>`;
      } else {
        const isSelected = SELECTED_PAIRS.some(p =>
          (p.i === topF[i] && p.j === topF[j]) || (p.i === topF[j] && p.j === topF[i])
        );
        const alpha = Math.min(0.9, v * 1.4);
        const bg = isSelected ? `rgba(var(--accent-rgb),${alpha})` : `rgba(var(--blue-rgb),${alpha})`;
        const tc = (v > 0.35 || (isSelected && v > 0.2)) ? '#000' : 'var(--text-muted)';
        html += `<div class="heatmap-cell" style="background:${bg};color:${tc};width:${cw}px;height:38px;font-size:${fs}px;"
          title="${showFeat[i]} × ${showFeat[j]}: ΔR=${v.toFixed(3)}">${v.toFixed(2)}</div>`;
      }
    }
  }
  html += '</div>';
  container.innerHTML = html;
}

// ── Top pairs list (inspector) ────────────────

function renderTopPairs(pairs) {
  const el = document.getElementById('top-pairs-live');
  if (!el || !currentDataset || !DELTA_R) return;
  const features = currentDataset.features;
  const d = features.length;
  const allPairs = [];
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++)
    allPairs.push({ i, j, score: DELTA_R[i][j] });
  allPairs.sort((a, b) => b.score - a.score);
  const maxScore = allPairs[0]?.score || 1;

  let html = '';
  allPairs.slice(0, 6).forEach(({ i, j, score }) => {
    const isSelected = pairs.some(p => p.i === i && p.j === j);
    const pct = Math.round(score / maxScore * 100);
    const fi = features[i].split(' ')[0];
    const fj = features[j].split(' ')[0];
    html += `<div class="pair-row">
      <div class="pair-names" style="color:${isSelected ? 'var(--accent)' : 'var(--text-secondary)'}">
        <span>${fi}</span><span class="pair-names__sep">×</span><span>${fj}</span>
      </div>
      <div class="dot-bar">${dotBar(pct, 12, isSelected ? 'on' : 'blue')}</div>
      <span class="pair-score" style="color:${isSelected ? 'var(--accent)' : 'var(--text-muted)'}">${score.toFixed(3)}</span>
      ${isSelected ? '<span class="badge badge--accent" style="font-size:8px;padding:2px 6px;">SEL</span>' : ''}
    </div>`;
  });
  el.innerHTML = html;
}

function dotBar(pct, count, type) {
  let html = '';
  const filled = Math.round(pct / 100 * count);
  for (let i = 0; i < count; i++)
    html += `<div class="dot ${i < filled ? 'dot--' + type : ''}"></div>`;
  return html;
}

// ── Enriched features display ─────────────────

function renderEnrichedFeatures(rawFeatures, normFeatures, pairs) {
  const el = document.getElementById('enrich-display');
  if (!el || !currentDataset) return;
  const { features } = currentDataset;

  let html = '<div style="margin-bottom:6px;"><div class="panel-title" style="margin-bottom:8px;">ORIGINAL FEATURES</div><div class="enrich-wrap">';
  rawFeatures.forEach((v, i) => {
    html += `<div class="enrich-chip">${features[i].split(' ')[0]}: <span style="color:var(--text-primary)">${fmtVal(v)}</span></div>`;
  });
  html += '</div></div>';
  html += `<div><div class="panel-title" style="margin-bottom:8px;color:var(--accent);">+ INTERACTION TERMS (k=${DEMO_K})</div><div class="enrich-wrap">`;
  pairs.forEach(({ i, j }) => {
    const val = normFeatures[i] * normFeatures[j];
    const fi = features[i].split(' ')[0].substring(0, 4);
    const fj = features[j].split(' ')[0].substring(0, 4);
    html += `<div class="enrich-chip enrich-chip--new">${fi}×${fj}: <span>${val.toFixed(3)}</span></div>`;
  });
  html += '</div></div>';
  el.innerHTML = html;
}

// ── Classification result ─────────────────────

function renderResult(probs, rawFeatures, normFeatures) {
  const el = document.getElementById('class-result');
  if (!el || !currentDataset) return;
  const { classes, classColors } = currentDataset;
  const predicted = probs.indexOf(Math.max(...probs));

  let html = `<div class="result-class" style="color:${classColors[predicted]}">${classes[predicted]}</div>`;
  html += `<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);margin-bottom:12px;letter-spacing:0.06em;">
    CONFIDENCE: ${(Math.max(...probs) * 100).toFixed(1)}%
  </div>`;
  probs.forEach((p, i) => {
    const pct = (p * 100).toFixed(1);
    html += `<div class="result-bar-row">
      <div class="result-bar-label" style="color:${classColors[i]}">${classes[i]}</div>
      <div class="result-bar-track"><div class="result-bar-fill" style="width:${pct}%;background:${classColors[i]}"></div></div>
      <div class="result-pct" style="color:${classColors[i]}">${pct}%</div>
    </div>`;
  });
  if (probs.length > 2) {
    html += `<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);margin-top:6px;line-height:1.5;">
      Multi-class scores are a softmax over one-vs-rest logits — illustrative, not a calibrated posterior.
    </div>`;
  }

  if (SELECTED_PAIRS && SELECTED_PAIRS.length > 0) {
    const sp = SELECTED_PAIRS[0];
    const intVal = normFeatures[sp.i] * normFeatures[sp.j];
    html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);letter-spacing:0.08em;margin-bottom:4px;">KEY INTERACTION SIGNAL</div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--accent);">
        ${currentDataset.features[sp.i].split(' ')[0]} × ${currentDataset.features[sp.j].split(' ')[0]} = ${intVal.toFixed(3)}
        <span style="color:var(--text-muted);font-size:9px;"> (ΔR=${DELTA_R[sp.i][sp.j].toFixed(3)})</span>
      </div>
    </div>`;
  }
  el.innerHTML = html;
}

// ── BC heatmap + benchmark (pre-baked) ────────

function renderHeatmap() {
  const container = document.getElementById('bc-heatmap');
  if (!container) return;
  const d = BC_FEATURES.length;
  const shortLabels = BC_FEATURES.map(f =>
    f.replace(' err','·e').replace('mean ','m·').replace('worst ','w·').replace(' pts','·p')
  );

  let html = `<div style="display:grid;grid-template-columns:84px repeat(${d},54px);gap:3px;align-items:center;">`;
  html += '<div></div>';
  shortLabels.forEach(l => {
    html += `<div class="heatmap-label" style="text-align:center;padding:2px 0;font-size:9px;">${l}</div>`;
  });
  for (let i = 0; i < d; i++) {
    html += `<div class="heatmap-label" style="text-align:right;padding-right:6px;font-size:9px;">${shortLabels[i]}</div>`;
    for (let j = 0; j < d; j++) {
      const v = BC_DELTA_R[i][j];
      if (i === j) {
        html += `<div class="heatmap-cell" style="background:rgba(255,255,255,0.04);color:var(--text-muted);font-size:9px;width:54px;height:40px;">—</div>`;
      } else {
        const alpha = Math.min(0.92, v * 1.5);
        const textColor = v > 0.4 ? '#000' : 'var(--text-muted)';
        const bg = v > 0.5
          ? `rgba(var(--accent-rgb),${alpha})`
          : v > 0.25
            ? `rgba(var(--blue-rgb),${alpha * 0.8})`
            : `rgba(255,255,255,${alpha * 0.3})`;
        html += `<div class="heatmap-cell" style="background:${bg};color:${textColor};font-size:9px;width:54px;height:40px;"
          onmouseenter="showTooltip(event,'${BC_FEATURES[i]} × ${BC_FEATURES[j]}',${v})"
          onmouseleave="hideTooltip()">${v.toFixed(2)}</div>`;
      }
    }
  }
  html += '</div>';
  container.innerHTML = html;
  renderTopPairsBC();
  renderKChart();
}

function renderTopPairsBC() {
  const el = document.getElementById('top-pairs-bc');
  if (!el) return;
  let html = '';
  TOP_PAIRS_BC.forEach((p, idx) => {
    const pct = Math.round(p.score / 0.75 * 100);
    const isTop3 = idx < 3;
    html += `<div class="pair-row" style="margin-bottom:10px;">
      <div style="font-family:var(--font-mono);font-size:9px;min-width:220px;color:${isTop3 ? 'var(--accent)' : 'var(--text-secondary)'}">
        ${String(idx + 1).padStart(2, '0')} — ${p.pair}
      </div>
      <div class="dot-bar">${dotBar(pct, 14, isTop3 ? 'on' : 'blue')}</div>
      <span style="font-family:var(--font-mono);font-size:11px;font-weight:700;min-width:40px;text-align:right;color:${isTop3 ? 'var(--accent)' : 'var(--text-muted)'}">
        ${p.score.toFixed(3)}
      </span>
    </div>`;
  });
  el.innerHTML = html;
}

function renderKChart() {
  const canvas = document.getElementById('k-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 24, right: 24, bottom: 36, left: 52 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#16181D'; ctx.fillRect(0, 0, W, H);

  const minAcc = 92, maxAcc = 98.5, minK = 0, maxK = 80;
  const px = k => pad.left + (k - minK) / (maxK - minK) * cw;
  const py = acc => pad.top + ch - (acc - minAcc) / (maxAcc - minAcc) * ch;

  ctx.strokeStyle = '#2C303A'; ctx.lineWidth = 1;
  [93, 94, 95, 96, 97, 98].forEach(acc => {
    ctx.beginPath(); ctx.moveTo(pad.left, py(acc)); ctx.lineTo(pad.left + cw, py(acc)); ctx.stroke();
    ctx.fillStyle = '#8A90A0'; ctx.font = '10px Space Mono';
    ctx.textAlign = 'right'; ctx.fillText(acc + '%', pad.left - 6, py(acc) + 3);
  });
  [0, 10, 20, 30, 40, 50, 60, 70, 80].forEach(k => {
    ctx.beginPath(); ctx.moveTo(px(k), pad.top); ctx.lineTo(px(k), pad.top + ch); ctx.stroke();
    ctx.fillStyle = '#8A90A0'; ctx.font = '10px Space Mono';
    ctx.textAlign = 'center'; ctx.fillText('k=' + k, px(k), pad.top + ch + 18);
  });

  ctx.strokeStyle = '#F1573E'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.left, py(92.97)); ctx.lineTo(pad.left + cw, py(92.97)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#F1573E'; ctx.font = '10px Space Mono'; ctx.textAlign = 'left';
  ctx.fillText('GNB 92.97%', pad.left + 4, py(92.97) - 4);

  ctx.strokeStyle = '#FFC24B'; ctx.lineWidth = 2;
  ctx.beginPath();
  K_BREAST.forEach(({ k, acc }, idx) => { if (idx === 0) ctx.moveTo(px(k), py(acc)); else ctx.lineTo(px(k), py(acc)); });
  ctx.stroke();

  ctx.beginPath();
  K_BREAST.forEach(({ k, acc }, idx) => { if (idx === 0) ctx.moveTo(px(k), py(acc)); else ctx.lineTo(px(k), py(acc)); });
  ctx.lineTo(px(K_BREAST[K_BREAST.length - 1].k), py(minAcc));
  ctx.lineTo(px(K_BREAST[0].k), py(minAcc));
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0, 'rgba(255,194,75,0.16)');
  grad.addColorStop(1, 'rgba(255,194,75,0)');
  ctx.fillStyle = grad; ctx.fill();

  K_BREAST.forEach(({ k, acc }) => {
    const isOpt = k === 45;
    ctx.beginPath(); ctx.arc(px(k), py(acc), isOpt ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isOpt ? '#FFC24B' : '#A77E2E'; ctx.fill();
    if (isOpt) {
      ctx.fillStyle = '#FFC24B'; ctx.font = 'bold 10px Space Mono'; ctx.textAlign = 'center';
      ctx.fillText('★ 97.72%', px(k), py(acc) - 10);
    }
  });

  ctx.fillStyle = '#8A90A0'; ctx.font = '10px Space Mono';
  ctx.textAlign = 'center'; ctx.fillText('NUMBER OF SELECTED PAIRS  k', pad.left + cw / 2, H - 4);
  ctx.save(); ctx.translate(10, pad.top + ch / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('5-FOLD CV ACCURACY  (%)', 0, 0); ctx.restore();
}

function renderAccuracyBars() {
  const el = document.getElementById('acc-bars');
  if (!el) return;
  const datasets = [
    { label: 'BREAST CANCER', min: 90, max: 100, rows: [
      { model: 'GNB',    acc: 92.97, color: 'var(--text-muted)' },
      { model: 'LDA',    acc: 95.61, color: 'var(--blue)' },
      { model: 'LR',     acc: 97.37, color: 'var(--blue)' },
      { model: 'SVM',    acc: 97.37, color: 'var(--blue)' },
      { model: 'CS-LLN', acc: 97.72, color: 'var(--accent)' },
    ]},
    { label: 'WINE RECOGNITION', min: 94, max: 102, rows: [
      { model: 'SVM',    acc: 96.63, color: 'var(--text-muted)' },
      { model: 'GNB',    acc: 97.75, color: 'var(--blue)' },
      { model: 'LDA',    acc: 98.30, color: 'var(--blue)' },
      { model: 'LR',     acc: 98.33, color: 'var(--blue)' },
      { model: 'CS-LLN', acc: 98.89, color: 'var(--accent)' },
    ]},
    { label: 'IRIS', min: 92, max: 100, rows: [
      { model: 'GNB',    acc: 94.67, color: 'var(--text-muted)' },
      { model: 'LR',     acc: 95.33, color: 'var(--blue)' },
      { model: 'SVM',    acc: 96.67, color: 'var(--blue)' },
      { model: 'CS-LLN', acc: 96.67, color: 'var(--blue)' },
      { model: 'LDA',    acc: 97.33, color: 'var(--accent)' },
    ]},
  ];
  el.innerHTML = datasets.map(ds => {
    const rows = ds.rows.map(r => {
      const pct = Math.min(100, Math.max(0, (r.acc - ds.min) / (ds.max - ds.min) * 100));
      return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <span style="font-family:var(--font-mono);font-size:9px;color:${r.color};min-width:48px;font-weight:700">${r.model}</span>
        <div style="flex:1;height:5px;background:var(--border);border-radius:2px;overflow:hidden;">
          <div style="width:${pct}%;height:100%;background:${r.color};"></div>
        </div>
        <span style="font-family:var(--font-mono);font-size:9px;color:${r.color};min-width:44px;text-align:right">${r.acc}%</span>
      </div>`;
    }).join('');
    return `<div>
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);letter-spacing:0.1em;margin-bottom:12px;">${ds.label}</div>
      ${rows}
    </div>`;
  }).join('');
}

function renderBenchmark() {
  renderAccuracyBars();
  Object.entries(BENCHMARK).forEach(([, data], dIdx) => {
    const el = document.getElementById('bench-' + dIdx);
    if (!el) return;
    const noteEl = el.previousElementSibling;
    if (noteEl && noteEl.classList.contains('bench-note'))
      noteEl.textContent = data.note;
    let html = `<table class="cmp-table"><thead><tr>
      <th>MODEL</th><th>ACCURACY ± STD</th><th>LOG-LOSS</th><th>MACRO-F1</th>
    </tr></thead><tbody>`;
    data.rows.forEach(r => {
      const isBest = r.model.includes(data.best) || r.model === data.best;
      html += `<tr class="${isBest ? 'best' : ''}">
        <td>${r.model}</td>
        <td>${r.acc.toFixed(2)}% ± ${r.std.toFixed(1)}</td>
        <td>${r.loss.toFixed(4)}</td>
        <td>${r.f1.toFixed(4)}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    el.innerHTML = html;
  });
}

// ── Tooltip ───────────────────────────────────

function showTooltip(e, label, val) {
  const t = document.getElementById('hm-tooltip');
  if (!t) return;
  t.style.display = 'block';
  t.style.left = (e.clientX + 12) + 'px';
  t.style.top = (e.clientY - 32) + 'px';
  t.innerHTML = `<span style="color:var(--accent)">${label}</span><br>ΔR<sub>ij</sub> = <span style="color:var(--accent)">${val.toFixed(3)}</span>`;
}
function hideTooltip() {
  const t = document.getElementById('hm-tooltip');
  if (t) t.style.display = 'none';
}

// ── Init ─────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => setPage(btn.dataset.page));
  });

  const kSlider = document.getElementById('k-slider');
  if (kSlider) {
    kSlider.addEventListener('input', () => {
      if (!currentDataset || !DELTA_R) return;
      const d = currentDataset.features.length;
      const maxPairs = d * (d - 1) / 2;
      DEMO_K = Math.max(1, Math.min(parseInt(kSlider.value), maxPairs));
      kSlider.value = DEMO_K;

      const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
      set('k-val', DEMO_K);
      set('k-val-2', DEMO_K);
      set('stat-pairs', DEMO_K);
      set('stat-dim', d + DEMO_K);

      SELECTED_PAIRS = topK(DELTA_R, DEMO_K, d);
      const Xnorm = normalizeX(currentDataset.X, normalizer);
      const Xe = enrich(Xnorm, SELECTED_PAIRS);
      LR_MODELS = trainOvR(Xe, currentDataset.y);
      runDemo();
    });
  }

  loadDataset('iris');
  setPage('inspector');
});
