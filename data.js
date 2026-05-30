/* ─────────────────────────────────────────────
   data.js  —  Three UCI datasets for CS-LLN
   ─────────────────────────────────────────────
   Iris:         actual Fisher 1936 data (150×4)
   Wine:         seeded-deterministic, matches UCI stats + known
                 within-class correlation structure (180×13)
   Breast Cancer: seeded-deterministic, matches UCI statistics
                 including geometric feature co-elevation in
                 malignant tissue (120×10 top features)
   ───────────────────────────────────────────── */

// ── Seeded PRNG (Mulberry32) ─────────────────
function mkrng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller Gaussian sample
function gauss(rng, mu, sigma) {
  const u = Math.max(1e-12, rng()), v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Clip to [lo, hi]
const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));


// ════════════════════════════════════════════
// 1. IRIS  (actual Fisher 1936 data, 150×4)
//    features: sepal_len, sepal_wid, petal_len, petal_wid
//    classes:  0=Setosa, 1=Versicolor, 2=Virginica
// ════════════════════════════════════════════
const IRIS_X = [
  // Setosa (0)
  [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],
  [5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],
  [5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],
  [5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
  [5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],
  [5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],
  [4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],
  [5.0,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],
  [5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],
  [4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],
  // Versicolor (1)
  [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],
  [5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],
  [5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],
  [6.7,3.1,4.4,1.4],[5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
  [5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],
  [6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],[6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],
  [5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],[5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],
  [6.0,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],
  [5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],[5.6,2.7,4.2,1.3],
  [5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],
  // Virginica (2)
  [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],
  [7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],
  [6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],
  [6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
  [6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],
  [7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],
  [7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],
  [7.7,3.0,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],
  [6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],
  [6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],[6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8],
];
const IRIS_Y = [
  ...Array(50).fill(0), ...Array(50).fill(1), ...Array(50).fill(2)
];


// ════════════════════════════════════════════
// 2. WINE  (180×13, seeded-deterministic)
//    Matches UCI statistics + known within-class correlations
//    Key interaction: flavanoids × total_phenols shifts across classes
//    Class 1→ high quality, Class 2→ medium, Class 3→ low-flavanoid
// ════════════════════════════════════════════
(function buildWine() {
  const rng = mkrng(0xDEADBEEF);

  // [mu, sigma] for each feature per class
  // Features: alcohol, malic_acid, ash, alcalinity, magnesium,
  //           total_phenols, flavanoids, nonflavanoid_phenols,
  //           proanthocyanins, color_intensity, hue, od280, proline
  const STATS = [
    // class 1 (high quality, 60 samples)
    [[13.74,0.51],[2.01,0.68],[2.46,0.23],[17.0,2.24],[106,14],
     [2.84,0.34],[2.98,0.43],[0.29,0.06],[1.90,0.44],[5.53,1.27],
     [1.06,0.12],[3.16,0.40],[1115,241]],
    // class 2 (medium, 60 samples)
    [[12.28,0.54],[1.93,1.06],[2.24,0.31],[20.2,3.2],[94,14],
     [2.26,0.44],[2.08,0.58],[0.36,0.12],[1.63,0.57],[3.09,0.98],
     [1.05,0.19],[2.79,0.49],[519,147]],
    // class 3 (low-flavanoid, 60 samples)
    [[13.15,0.52],[3.33,1.01],[2.44,0.20],[21.4,2.4],[99,12],
     [1.68,0.27],[0.78,0.24],[0.45,0.09],[1.15,0.36],[7.40,2.32],
     [0.68,0.10],[1.68,0.37],[629,117]],
  ];

  // Correlation factors injected per class so ΔRij is meaningful
  // Class 1: total_phenols (idx5) and flavanoids (idx6) share a common factor (r≈0.87)
  // Class 3: malic_acid (idx1) and alcalinity (idx3) share a common factor (r≈0.73)
  //          color_intensity (idx9) and malic_acid (idx1) share a factor (r≈0.70)

  window.WINE_X = [];
  window.WINE_Y = [];

  STATS.forEach((classStats, ci) => {
    for (let s = 0; s < 60; s++) {
      // Independent base draws
      const row = classStats.map(([mu, sg]) => gauss(rng, mu, sg));

      if (ci === 0) {
        // Class 1: flavanoids tracks total_phenols closely
        const phenolFactor = (row[5] - classStats[5][0]) / classStats[5][1];
        row[6] = classStats[6][0] + classStats[6][1] * (0.87 * phenolFactor + 0.49 * gauss(rng, 0, 1));
        row[6] = clip(row[6], 1.0, 5.1);
        // alcohol correlates mildly with proline
        const alFactor = (row[0] - classStats[0][0]) / classStats[0][1];
        row[12] = classStats[12][0] + classStats[12][1] * (0.55 * alFactor + 0.83 * gauss(rng, 0, 1));
        row[12] = clip(row[12], 400, 1680);
      }
      if (ci === 1) {
        // Class 2: weaker correlations — mostly independent
        const f = gauss(rng, 0, 1);
        row[5] = classStats[5][0] + classStats[5][1] * (0.40 * f + 0.92 * gauss(rng, 0, 1));
        row[6] = classStats[6][0] + classStats[6][1] * (0.35 * f + 0.94 * gauss(rng, 0, 1));
      }
      if (ci === 2) {
        // Class 3: malic acid and alcalinity share a factor
        const malFactor = (row[1] - classStats[1][0]) / classStats[1][1];
        row[3] = classStats[3][0] + classStats[3][1] * (0.73 * malFactor + 0.68 * gauss(rng, 0, 1));
        row[3] = clip(row[3], 10, 30);
        // color intensity co-varies with malic acid
        row[9] = classStats[9][0] + classStats[9][1] * (0.70 * malFactor + 0.71 * gauss(rng, 0, 1));
        row[9] = clip(row[9], 1.0, 13.0);
        // flavanoids very low and weakly correlated with phenols
        row[6] = clip(gauss(rng, classStats[6][0], classStats[6][1]), 0.34, 1.5);
        row[5] = clip(gauss(rng, classStats[5][0], classStats[5][1]), 0.9, 2.5);
      }

      window.WINE_X.push(row.map(v => +v.toFixed(3)));
      window.WINE_Y.push(ci);
    }
  });
})();


// ════════════════════════════════════════════
// 3. BREAST CANCER  (120×10 top features, seeded)
//    Using top 10 features from paper Fig. 2 / Fig. 5:
//    worst_radius, worst_area, worst_perimeter, mean_area,
//    mean_radius, mean_perimeter, radius_error, perimeter_error,
//    worst_concavity, mean_concavity
//    Key ΔRij: malignant has geometric co-elevation;
//             benign has much weaker co-elevation
//    Classes: 0=Malignant, 1=Benign
// ════════════════════════════════════════════
(function buildBC() {
  const rng = mkrng(0xCAFEBABE);

  // Feature means/stds from UCI per class
  // Features: [worst_rad, worst_area, worst_perim, mean_area, mean_rad,
  //            mean_perim, rad_error, perim_error, worst_conc, mean_conc]
  const MALIGNANT_STATS = [
    [21.13,4.28],[2226,981],[136.7,26.9],[978,367],[17.46,3.20],
    [115.4,21.0],[1.217,0.583],[8.69,3.92],[0.452,0.174],[0.160,0.080]
  ];
  const BENIGN_STATS = [
    [14.27,2.18],[703,216],[92.1,13.5],[462,134],[12.15,1.78],
    [78.1,11.4],[0.412,0.189],[2.90,1.52],[0.105,0.088],[0.046,0.043]
  ];

  window.BC_X = [];
  window.BC_Y = [];

  // Malignant: worst_rad, worst_perim, worst_area all driven by shared geometry factor
  // → very high within-class correlation r≈0.95 (paper confirms this)
  for (let s = 0; s < 60; s++) {
    const geoFactor = gauss(rng, 0, 1);  // shared size signal
    const row = MALIGNANT_STATS.map(([mu, sg]) => gauss(rng, mu, sg));
    // Override geometric features to be co-elevated
    row[0] = MALIGNANT_STATS[0][0] + MALIGNANT_STATS[0][1] * (0.88 * geoFactor + 0.47 * gauss(rng, 0, 1));
    row[1] = MALIGNANT_STATS[1][0] + MALIGNANT_STATS[1][1] * (0.92 * geoFactor + 0.39 * gauss(rng, 0, 1));
    row[2] = MALIGNANT_STATS[2][0] + MALIGNANT_STATS[2][1] * (0.90 * geoFactor + 0.44 * gauss(rng, 0, 1));
    row[3] = MALIGNANT_STATS[3][0] + MALIGNANT_STATS[3][1] * (0.87 * geoFactor + 0.49 * gauss(rng, 0, 1));
    row[4] = MALIGNANT_STATS[4][0] + MALIGNANT_STATS[4][1] * (0.85 * geoFactor + 0.53 * gauss(rng, 0, 1));
    row[5] = MALIGNANT_STATS[5][0] + MALIGNANT_STATS[5][1] * (0.84 * geoFactor + 0.54 * gauss(rng, 0, 1));
    row[6] = MALIGNANT_STATS[6][0] + MALIGNANT_STATS[6][1] * (0.75 * geoFactor + 0.66 * gauss(rng, 0, 1));
    // Clip to physical bounds
    row[0] = clip(row[0], 10, 36);
    row[1] = clip(row[1], 200, 4254);
    row[2] = clip(row[2], 72, 251);
    row[3] = clip(row[3], 144, 2501);
    row[4] = clip(row[4], 7.5, 29);
    row[5] = clip(row[5], 50, 200);
    row[8] = clip(row[8], 0, 1.25);
    row[9] = clip(row[9], 0, 0.43);
    window.BC_X.push(row.map(v => +v.toFixed(4)));
    window.BC_Y.push(0);
  }

  // Benign: geometric features vary more independently (weaker co-elevation)
  for (let s = 0; s < 60; s++) {
    const geoFactor = gauss(rng, 0, 1);
    const row = BENIGN_STATS.map(([mu, sg]) => gauss(rng, mu, sg));
    // Weaker shared factor → lower within-class correlation r≈0.50
    row[0] = BENIGN_STATS[0][0] + BENIGN_STATS[0][1] * (0.52 * geoFactor + 0.85 * gauss(rng, 0, 1));
    row[1] = BENIGN_STATS[1][0] + BENIGN_STATS[1][1] * (0.58 * geoFactor + 0.81 * gauss(rng, 0, 1));
    row[2] = BENIGN_STATS[2][0] + BENIGN_STATS[2][1] * (0.55 * geoFactor + 0.83 * gauss(rng, 0, 1));
    row[3] = BENIGN_STATS[3][0] + BENIGN_STATS[3][1] * (0.50 * geoFactor + 0.87 * gauss(rng, 0, 1));
    row[4] = BENIGN_STATS[4][0] + BENIGN_STATS[4][1] * (0.48 * geoFactor + 0.88 * gauss(rng, 0, 1));
    row[5] = BENIGN_STATS[5][0] + BENIGN_STATS[5][1] * (0.47 * geoFactor + 0.88 * gauss(rng, 0, 1));
    row[6] = BENIGN_STATS[6][0] + BENIGN_STATS[6][1] * (0.40 * geoFactor + 0.92 * gauss(rng, 0, 1));
    row[0] = clip(row[0], 6.9, 21);
    row[1] = clip(row[1], 143, 1610);
    row[2] = clip(row[2], 43, 136);
    row[3] = clip(row[3], 143, 1160);
    row[4] = clip(row[4], 6.9, 17.8);
    row[5] = clip(row[5], 43, 115);
    row[8] = clip(row[8], 0, 0.71);
    row[9] = clip(row[9], 0, 0.20);
    window.BC_X.push(row.map(v => +v.toFixed(4)));
    window.BC_Y.push(1);
  }
})();


// ════════════════════════════════════════════
// Dataset registry
// ════════════════════════════════════════════
window.DATASET_REGISTRY = {
  iris: {
    key: 'iris',
    label: 'IRIS',
    subLabel: 'Fisher 1936 · exact data',
    description: 'N=150, d=4, C=3  ·  Classic iris flower morphology. CS-LLN achieves 96.67% (LDA leads at 97.33% — low d limits interaction benefit).',
    features: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    units: ['cm', 'cm', 'cm', 'cm'],
    classes: ['Setosa', 'Versicolor', 'Virginica'],
    classColors: ['var(--accent)', 'var(--blue)', 'var(--purple)'],
    X: IRIS_X,
    y: IRIS_Y,
    defaultK: 3,
    bestResult: '96.67% (CS-LLN), 97.33% (LDA)',
  },
  wine: {
    key: 'wine',
    label: 'WINE',
    subLabel: 'seeded synthetic proxy',
    description: 'd=13, C=3  ·  Italian wine cultivar chemistry. Live demo uses a seeded synthetic proxy (180 rows) that reproduces the paper’s within-class correlation structure; the paper’s benchmark uses the real UCI set (N=178), where CS-LLN reaches 98.89% (k=1 → 99.44%).',
    features: ['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity', 'Magnesium',
               'Total Phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
               'Color Intensity', 'Hue', 'OD280/315', 'Proline'],
    units: ['%', 'g/L', 'g/L', 'meq/L', 'mg/L', 'g/L', 'g/L', 'g/L', 'g/L', 'AU', '', '', 'mg/L'],
    classes: ['Cultivar 1', 'Cultivar 2', 'Cultivar 3'],
    classColors: ['var(--accent)', 'var(--blue)', 'var(--purple)'],
    get X() { return window.WINE_X; },
    get y() { return window.WINE_Y; },
    defaultK: 5,
    bestResult: '98.89% (CS-LLN) · k=1 → 99.44%',
  },
  breast_cancer: {
    key: 'breast_cancer',
    label: 'BREAST CANCER',
    subLabel: 'seeded synthetic proxy',
    description: 'd=10, C=2  ·  Cell-nucleus measurements. Live demo uses a seeded synthetic proxy (120 rows, 10 features) with the same geometric co-elevation the paper reports. The paper’s benchmark uses the real UCI set (N=569, d=30), where CS-LLN reaches 97.72% (63% error ↓ vs GNB).',
    features: ['Worst Radius', 'Worst Area', 'Worst Perimeter', 'Mean Area', 'Mean Radius',
               'Mean Perimeter', 'Radius Error', 'Perim. Error', 'Worst Concavity', 'Mean Concavity'],
    units: ['μm','μm²','μm','μm²','μm','μm','μm','μm','',''],
    classes: ['Malignant', 'Benign'],
    classColors: ['var(--danger)', 'var(--accent)'],
    get X() { return window.BC_X; },
    get y() { return window.BC_Y; },
    defaultK: 4,
    bestResult: '97.72% (CS-LLN) · 63% error ↓ vs GNB',
  },
};
