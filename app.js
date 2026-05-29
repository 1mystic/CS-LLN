/* ─────────────────────────────────────────────
   CS-LLN Inspector — app.js
   Full CS-LLN pipeline in vanilla JS
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

// ── Pre-baked Breast Cancer ΔRij data (Fig. 2 of paper) ──
const BC_FEATURES = [
  'radius err', 'smooth err', 'sym err', 'perim err',
  'mean perim', 'mean conc', 'mean area', 'mean radius',
  'worst perim', 'worst area', 'conc pts err', 'worst radius'
];

// Upper triangle ΔRij values from Figure 2
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

// Top 10 pairs (Figure 5)
const TOP_PAIRS_BC = [
  { pair: 'mean area × radius error',         score: 0.695 },
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

// ── Benchmark data (Tables I-III) ──
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

// ── k-ablation data (Fig. 4, approximate) ──
const K_BREAST = [
  {k:1,acc:97.37},{k:5,acc:97.50},{k:10,acc:97.58},{k:15,acc:97.62},
  {k:20,acc:97.66},{k:30,acc:97.70},{k:40,acc:97.71},{k:45,acc:97.72},
  {k:50,acc:97.65},{k:60,acc:97.50},{k:70,acc:97.20},{k:78,acc:96.49}
];

// ── Toy Student Dataset ──────────────────────
// Features: [attendance, study_hrs, sleep_quality, stress_level]
// Normalized 0-1. Class 0 = high, Class 1 = at-risk
const STUDENT_FEATURES = ['Attendance', 'Study Hrs', 'Sleep Quality', 'Stress Level'];

const TRAIN_X = [
  [0.90,0.85,0.70,0.20],[0.80,0.90,0.60,0.30],[0.85,0.80,0.65,0.25],
  [0.95,0.95,0.75,0.15],[0.70,0.75,0.60,0.35],[0.80,0.85,0.70,0.20],
  [0.90,0.90,0.60,0.25],[0.75,0.80,0.65,0.30],[0.85,0.90,0.70,0.20],
  [0.80,0.80,0.60,0.30],[0.88,0.82,0.68,0.22],[0.78,0.88,0.62,0.28],
  [0.30,0.25,0.30,0.85],[0.40,0.35,0.25,0.90],[0.25,0.20,0.35,0.80],
  [0.35,0.30,0.20,0.90],[0.40,0.45,0.30,0.75],[0.30,0.25,0.25,0.85],
  [0.20,0.15,0.20,0.95],[0.45,0.40,0.35,0.70],[0.30,0.30,0.25,0.80],
  [0.25,0.20,0.30,0.85],[0.35,0.25,0.28,0.82],[0.42,0.38,0.32,0.78],
];
const TRAIN_Y = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1];

// ── CS-LLN JavaScript Implementation ─────────

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
  const classes = [...new Set(y)].sort();
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
  for (let m = 0; m < C; m++) for (let p = m+1; p < C; p++, n++) {
    for (let i = 0; i < d; i++) for (let j = i+1; j < d; j++) {
      dr[i][j] += Math.abs(corrs[m][i][j] - corrs[p][i][j]);
    }
  }
  if (n) for (let i = 0; i < d; i++) for (let j = i+1; j < d; j++) {
    dr[i][j] /= n;
    dr[j][i] = dr[i][j];
  }
  return dr;
}

function topK(dr, k, d) {
  const pairs = [];
  for (let i = 0; i < d; i++) for (let j = i+1; j < d; j++)
    pairs.push({i, j, score: dr[i][j]});
  pairs.sort((a, b) => b.score - a.score);
  return pairs.slice(0, k);
}

function enrich(X, pairs) {
  return X.map(row => [
    ...row,
    ...pairs.map(({i, j}) => row[i] * row[j])
  ]);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function trainLR(X, y, lambda = 0.5, lr = 0.08, iters = 2000) {
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
    for (let j = 0; j < d; j++) w[j] -= lr * (dw[j]/n + lambda * w[j]);
    b -= lr * db / n;
  }
  return {w, b};
}

function predictProba(x, model) {
  const {w, b} = model;
  const z = w.reduce((s, wi, j) => s + wi * x[j], b);
  const p1 = sigmoid(z);
  return [1 - p1, p1];
}

// ── Train CS-LLN on student data ──
let CORRS, DELTA_R, SELECTED_PAIRS, LR_MODEL;
let DEMO_K = 3;

function initCSLLN() {
  const d = STUDENT_FEATURES.length;
  CORRS = classCorrelations(TRAIN_X, TRAIN_Y, d);
  DELTA_R = computeDeltaR(CORRS);
  SELECTED_PAIRS = topK(DELTA_R, DEMO_K, d);
  const Xe = enrich(TRAIN_X, SELECTED_PAIRS);
  LR_MODEL = trainLR(Xe, TRAIN_Y);
}

// ── Run demo from sliders ──
function runDemo() {
  const features = ['att', 'study', 'sleep', 'stress'].map(id => {
    const el = document.getElementById('sl-' + id);
    return el ? parseFloat(el.value) / 5.0 : 0.5;
  });

  // Live ΔRij on training set
  renderMiniHeatmap(DELTA_R);

  // Top pairs
  renderTopPairs(SELECTED_PAIRS);

  // Enriched features
  renderEnrichedFeatures(features, SELECTED_PAIRS);

  // Classification
  const xe = enrich([features], SELECTED_PAIRS)[0];
  const probs = predictProba(xe, LR_MODEL);
  renderResult(probs, features);
}

// ── Render mini heatmap (4×4 student) ──
function renderMiniHeatmap(dr) {
  const container = document.getElementById('mini-heatmap');
  if (!container) return;
  const d = STUDENT_FEATURES.length;
  const labels = STUDENT_FEATURES.map(f => f.split(' ')[0]);

  let html = `<div style="display:grid;grid-template-columns:60px repeat(${d},44px);gap:2px;align-items:center;">`;
  // header row
  html += '<div></div>';
  labels.forEach(l => {
    html += `<div class="heatmap-label" style="text-align:center;padding:2px 0;">${l}</div>`;
  });
  // data rows
  for (let i = 0; i < d; i++) {
    html += `<div class="heatmap-label" style="text-align:right;padding-right:6px;">${labels[i]}</div>`;
    for (let j = 0; j < d; j++) {
      const v = dr[i][j];
      const alpha = i === j ? 0.05 : Math.min(0.9, v * 1.4);
      const isSelected = i !== j && SELECTED_PAIRS.some(p => (p.i===i&&p.j===j)||(p.i===j&&p.j===i));
      const bg = i === j
        ? 'rgba(255,255,255,0.05)'
        : isSelected
          ? `rgba(224,255,83,${alpha})`
          : `rgba(59,130,246,${alpha})`;
      const textColor = v > 0.35 && i !== j ? '#000' : isSelected && v > 0.2 ? '#000' : 'var(--text-muted)';
      html += `<div class="heatmap-cell" style="background:${bg};color:${textColor};width:44px;height:32px;font-size:9px;"
        title="${STUDENT_FEATURES[i]} × ${STUDENT_FEATURES[j]}: ΔR=${v.toFixed(3)}">${i===j?'—':v.toFixed(2)}</div>`;
    }
  }
  html += '</div>';
  container.innerHTML = html;
}

// ── Render top pairs list ──
function renderTopPairs(pairs) {
  const el = document.getElementById('top-pairs-live');
  if (!el) return;
  const d = STUDENT_FEATURES.length;
  const allPairs = [];
  for (let i = 0; i < d; i++) for (let j = i+1; j < d; j++)
    allPairs.push({i, j, score: DELTA_R[i][j]});
  allPairs.sort((a, b) => b.score - a.score);

  let html = '';
  allPairs.slice(0, 6).forEach(({i, j, score}) => {
    const isSelected = pairs.some(p => p.i===i && p.j===j);
    const pct = Math.round(score / 0.8 * 100);
    const dotColor = isSelected ? '--accent' : '--blue';
    const dots = dotBar(pct, 12, isSelected ? 'on' : 'blue');
    html += `<div class="pair-row">
      <div class="pair-names" style="color:${isSelected?'var(--accent)':'var(--text-secondary)'}">
        <span>${STUDENT_FEATURES[i]}</span>
        <span class="pair-names__sep">×</span>
        <span>${STUDENT_FEATURES[j]}</span>
      </div>
      <div class="dot-bar">${dots}</div>
      <span class="pair-score" style="color:${isSelected?'var(--accent)':'var(--text-muted)'}">${score.toFixed(3)}</span>
      ${isSelected ? '<span class="badge badge--accent" style="font-size:8px;padding:2px 6px;">SELECTED</span>' : ''}
    </div>`;
  });
  el.innerHTML = html;
}

function dotBar(pct, count, type) {
  let html = '';
  const filled = Math.round(pct / 100 * count);
  for (let i = 0; i < count; i++) {
    const cls = i < filled ? `dot--${type}` : '';
    html += `<div class="dot ${cls}"></div>`;
  }
  return html;
}

// ── Render enriched features ──
function renderEnrichedFeatures(features, pairs) {
  const el = document.getElementById('enrich-display');
  if (!el) return;
  let html = '<div style="margin-bottom:6px;"><div class="panel-title" style="margin-bottom:8px;">ORIGINAL FEATURES</div><div class="enrich-wrap">';
  features.forEach((v, i) => {
    html += `<div class="enrich-chip">${STUDENT_FEATURES[i]}: <span style="color:var(--text-primary)">${(v*5).toFixed(1)}</span></div>`;
  });
  html += '</div></div>';
  html += '<div><div class="panel-title" style="margin-bottom:8px;color:var(--accent);">+ INTERACTION TERMS (k=' + DEMO_K + ')</div><div class="enrich-wrap">';
  pairs.forEach(({i, j}) => {
    const val = features[i] * features[j];
    html += `<div class="enrich-chip enrich-chip--new">${STUDENT_FEATURES[i][0]}×${STUDENT_FEATURES[j][0]}: <span>${val.toFixed(3)}</span></div>`;
  });
  html += '</div></div>';
  el.innerHTML = html;
}

// ── Render classification result ──
function renderResult(probs, features) {
  const el = document.getElementById('class-result');
  if (!el) return;
  const labels = ['HIGH PERFORMER', 'AT-RISK'];
  const colors = ['var(--accent)', 'var(--danger)'];
  const predicted = probs[1] > 0.5 ? 1 : 0;

  let html = `<div class="result-class" style="color:${colors[predicted]}">${labels[predicted]}</div>`;
  html += `<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);margin-bottom:12px;letter-spacing:0.06em;">
    CONFIDENCE: ${(Math.max(...probs)*100).toFixed(1)}%
  </div>`;
  probs.forEach((p, i) => {
    const pct = (p * 100).toFixed(1);
    const barColor = colors[i];
    html += `<div class="result-bar-row">
      <div class="result-bar-label" style="color:${barColor}">${labels[i]}</div>
      <div class="result-bar-track">
        <div class="result-bar-fill" style="width:${pct}%;background:${barColor}"></div>
      </div>
      <div class="result-pct" style="color:${barColor}">${pct}%</div>
    </div>`;
  });

  // Key signal
  const signalPair = SELECTED_PAIRS[0];
  const intVal = features[signalPair.i] * features[signalPair.j];
  html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
    <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);letter-spacing:0.08em;margin-bottom:4px;">KEY INTERACTION SIGNAL</div>
    <div style="font-family:var(--font-mono);font-size:11px;color:var(--accent);">
      ${STUDENT_FEATURES[signalPair.i]} × ${STUDENT_FEATURES[signalPair.j]} = ${intVal.toFixed(3)}
      <span style="color:var(--text-muted);font-size:9px;"> (ΔR=${DELTA_R[signalPair.i][signalPair.j].toFixed(3)})</span>
    </div>
  </div>`;

  el.innerHTML = html;
}

// ── Render Breast Cancer heatmap (full) ──
function renderHeatmap() {
  const container = document.getElementById('bc-heatmap');
  if (!container) return;
  const d = BC_FEATURES.length;
  const shortLabels = BC_FEATURES.map(f => f.replace(' err','·e').replace('mean ','m·').replace('worst ','w·').replace(' pts','·p'));

  let html = `<div style="display:grid;grid-template-columns:72px repeat(${d},48px);gap:2px;align-items:center;">`;
  html += '<div></div>';
  shortLabels.forEach(l => {
    html += `<div class="heatmap-label" style="text-align:center;padding:2px 0;font-size:7px;">${l}</div>`;
  });

  for (let i = 0; i < d; i++) {
    html += `<div class="heatmap-label" style="text-align:right;padding-right:6px;font-size:7px;">${shortLabels[i]}</div>`;
    for (let j = 0; j < d; j++) {
      const v = BC_DELTA_R[i][j];
      if (i === j) {
        html += `<div class="heatmap-cell" style="background:rgba(255,255,255,0.04);color:var(--text-muted);font-size:8px;width:48px;height:36px;">—</div>`;
      } else {
        const alpha = Math.min(0.92, v * 1.5);
        const textColor = v > 0.4 ? '#000' : 'var(--text-muted)';
        const bg = v > 0.5
          ? `rgba(224,255,83,${alpha})`
          : v > 0.25
            ? `rgba(59,130,246,${alpha * 0.8})`
            : `rgba(255,255,255,${alpha * 0.3})`;
        html += `<div class="heatmap-cell" style="background:${bg};color:${textColor};font-size:8px;width:48px;height:36px;"
          onmouseenter="showTooltip(event,'${BC_FEATURES[i]} × ${BC_FEATURES[j]}',${v})"
          onmouseleave="hideTooltip()">${v.toFixed(2)}</div>`;
      }
    }
  }
  html += '</div>';
  container.innerHTML = html;

  // Top pairs ranked bar
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
      <div style="font-family:var(--font-mono);font-size:9px;min-width:220px;color:${isTop3?'var(--accent)':'var(--text-secondary)'}">
        ${String(idx+1).padStart(2,'0')} — ${p.pair}
      </div>
      <div class="dot-bar">${dotBar(pct, 14, isTop3?'on':'blue')}</div>
      <span style="font-family:var(--font-mono);font-size:11px;font-weight:700;min-width:40px;text-align:right;color:${isTop3?'var(--accent)':'var(--text-muted)'}">
        ${p.score.toFixed(3)}
      </span>
    </div>`;
  });
  el.innerHTML = html;
}

// ── k-ablation chart (canvas) ──
function renderKChart() {
  const canvas = document.getElementById('k-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {top: 24, right: 24, bottom: 36, left: 52};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#161616';
  ctx.fillRect(0, 0, W, H);

  const ks = K_BREAST.map(d => d.k);
  const accs = K_BREAST.map(d => d.acc);
  const minAcc = 92, maxAcc = 98.5;
  const minK = 0, maxK = 80;

  function px(k) { return pad.left + (k - minK) / (maxK - minK) * cw; }
  function py(acc) { return pad.top + ch - (acc - minAcc) / (maxAcc - minAcc) * ch; }

  // Grid lines
  ctx.strokeStyle = '#2A2A2A'; ctx.lineWidth = 1;
  [93, 94, 95, 96, 97, 98].forEach(acc => {
    ctx.beginPath(); ctx.moveTo(pad.left, py(acc)); ctx.lineTo(pad.left + cw, py(acc)); ctx.stroke();
    ctx.fillStyle = '#888888'; ctx.font = '9px Space Mono';
    ctx.textAlign = 'right'; ctx.fillText(acc + '%', pad.left - 6, py(acc) + 3);
  });
  [0, 10, 20, 30, 40, 50, 60, 70, 80].forEach(k => {
    ctx.beginPath(); ctx.moveTo(px(k), pad.top); ctx.lineTo(px(k), pad.top + ch); ctx.stroke();
    ctx.fillStyle = '#888888'; ctx.font = '9px Space Mono';
    ctx.textAlign = 'center'; ctx.fillText('k=' + k, px(k), pad.top + ch + 18);
  });

  // GNB baseline
  ctx.strokeStyle = '#F14336'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.left, py(92.97)); ctx.lineTo(pad.left + cw, py(92.97)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#F14336'; ctx.font = '9px Space Mono'; ctx.textAlign = 'left';
  ctx.fillText('GNB 92.97%', pad.left + 4, py(92.97) - 4);

  // CS-LLN curve
  ctx.strokeStyle = '#E0FF53'; ctx.lineWidth = 2;
  ctx.beginPath();
  K_BREAST.forEach(({k, acc}, idx) => {
    if (idx === 0) ctx.moveTo(px(k), py(acc));
    else ctx.lineTo(px(k), py(acc));
  });
  ctx.stroke();

  // Area under curve fill
  ctx.beginPath();
  K_BREAST.forEach(({k, acc}, idx) => {
    if (idx === 0) ctx.moveTo(px(k), py(acc));
    else ctx.lineTo(px(k), py(acc));
  });
  ctx.lineTo(px(K_BREAST[K_BREAST.length-1].k), py(minAcc));
  ctx.lineTo(px(K_BREAST[0].k), py(minAcc));
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0, 'rgba(224,255,83,0.15)');
  grad.addColorStop(1, 'rgba(224,255,83,0)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Dots + best point
  K_BREAST.forEach(({k, acc}) => {
    const isOpt = k === 45;
    ctx.beginPath();
    ctx.arc(px(k), py(acc), isOpt ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isOpt ? '#E0FF53' : '#88A020';
    ctx.fill();
    if (isOpt) {
      ctx.fillStyle = '#E0FF53'; ctx.font = 'bold 9px Space Mono'; ctx.textAlign = 'center';
      ctx.fillText('★ 97.72%', px(k), py(acc) - 10);
    }
  });

  // Axes labels
  ctx.fillStyle = '#888888'; ctx.font = '9px Space Mono';
  ctx.textAlign = 'center'; ctx.fillText('NUMBER OF SELECTED PAIRS  k', pad.left + cw/2, H - 4);
  ctx.save(); ctx.translate(10, pad.top + ch/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('5-FOLD CV ACCURACY  (%)', 0, 0); ctx.restore();
}

// ── Accuracy bars ──
function renderAccuracyBars() {
  const el = document.getElementById('acc-bars');
  if (!el) return;
  const datasets = [
    { label: 'BREAST CANCER', min: 90, max: 100, rows: [
      {model:'GNB',    acc:92.97, color:'var(--text-muted)'},
      {model:'LDA',    acc:95.61, color:'var(--blue)'},
      {model:'LR',     acc:97.37, color:'var(--blue)'},
      {model:'SVM',    acc:97.37, color:'var(--blue)'},
      {model:'CS-LLN', acc:97.72, color:'var(--accent)'},
    ]},
    { label: 'WINE RECOGNITION', min: 94, max: 102, rows: [
      {model:'SVM',    acc:96.63, color:'var(--text-muted)'},
      {model:'GNB',    acc:97.75, color:'var(--blue)'},
      {model:'LDA',    acc:98.30, color:'var(--blue)'},
      {model:'LR',     acc:98.33, color:'var(--blue)'},
      {model:'CS-LLN', acc:98.89, color:'var(--accent)'},
    ]},
    { label: 'IRIS', min: 92, max: 100, rows: [
      {model:'GNB',    acc:94.67, color:'var(--text-muted)'},
      {model:'LR',     acc:95.33, color:'var(--blue)'},
      {model:'SVM',    acc:96.67, color:'var(--blue)'},
      {model:'CS-LLN', acc:96.67, color:'var(--blue)'},
      {model:'LDA',    acc:97.33, color:'var(--accent)'},
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

// ── Benchmark tables ──
function renderBenchmark() {
  renderAccuracyBars();
  Object.entries(BENCHMARK).forEach(([name, data], dIdx) => {
    const el = document.getElementById('bench-' + dIdx);
    if (!el) return;
    el.innerHTML = '';

    const noteEl = el.previousElementSibling;
    if (noteEl && noteEl.classList.contains('bench-note'))
      noteEl.textContent = data.note;

    let html = `<table class="cmp-table"><thead><tr>
      <th>MODEL</th><th>ACCURACY ± STD</th><th>LOG-LOSS</th><th>MACRO-F1</th>
    </tr></thead><tbody>`;
    data.rows.forEach(r => {
      const isBest = r.model.includes(data.best) || r.model === data.best;
      html += `<tr class="${isBest?'best':''}">
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

// ── Tooltip ──
function showTooltip(e, label, val) {
  const t = document.getElementById('hm-tooltip');
  if (!t) return;
  t.style.display = 'block';
  t.style.left = (e.clientX + 12) + 'px';
  t.style.top  = (e.clientY - 32) + 'px';
  t.innerHTML = `<span style="color:var(--accent)">${label}</span><br>ΔR<sub>ij</sub> = <span style="color:var(--accent)">${val.toFixed(3)}</span>`;
}
function hideTooltip() {
  const t = document.getElementById('hm-tooltip');
  if (t) t.style.display = 'none';
}

// ── Init ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Navigation
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => setPage(btn.dataset.page));
  });

  // Slider wiring
  ['att', 'study', 'sleep', 'stress'].forEach(id => {
    const sl = document.getElementById('sl-' + id);
    const vl = document.getElementById('val-' + id);
    const track = document.getElementById('track-' + id);
    const thumb = document.getElementById('thumb-' + id);
    if (!sl) return;
    const update = () => {
      const v = parseFloat(sl.value);
      if (vl) vl.textContent = v.toFixed(1);
      const pct = (v - 1) / 4 * 100;
      if (track) track.style.width = pct + '%';
      if (thumb) thumb.style.left = pct + '%';
      runDemo();
    };
    sl.addEventListener('input', update);
    update();
  });

  // k slider
  const kSlider = document.getElementById('k-slider');
  const kVal = document.getElementById('k-val');
  const kVal2 = document.getElementById('k-val-2');
  const statPairs = document.getElementById('stat-pairs');
  const statDim = document.getElementById('stat-dim');
  if (kSlider) {
    kSlider.addEventListener('input', () => {
      DEMO_K = parseInt(kSlider.value);
      if (kVal) kVal.textContent = DEMO_K;
      if (kVal2) kVal2.textContent = DEMO_K;
      if (statPairs) statPairs.textContent = DEMO_K;
      if (statDim) statDim.textContent = STUDENT_FEATURES.length + DEMO_K;
      SELECTED_PAIRS = topK(DELTA_R, DEMO_K, STUDENT_FEATURES.length);
      const Xe = enrich(TRAIN_X, SELECTED_PAIRS);
      LR_MODEL = trainLR(Xe, TRAIN_Y);
      runDemo();
    });
  }

  // Train model
  initCSLLN();
  setPage('inspector');
});
