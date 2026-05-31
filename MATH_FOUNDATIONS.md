# Mathematical Foundations of ML — Complete Reference

*Every concept you need, built from scratch, with fully worked examples.*
*Pair with `LEARN.md` (the CS-LLN guide) and `MATH_FOUNDATIONS_BOUNDARIES.md` (decision-boundary deep-dive).*

---

## Contents

**Part A — Pure Math**
1. [Numbers, algebra, and functions](#part-a1--numbers-algebra-and-functions)
2. [Vectors](#part-a2--vectors)
3. [Matrices](#part-a3--matrices)
4. [Calculus — derivatives and gradients](#part-a4--calculus)
5. [Exponents and logarithms](#part-a5--exponents-and-logarithms)

**Part B — Probability and Statistics**
6. [Probability fundamentals](#part-b6--probability-fundamentals)
7. [Conditional probability and Bayes theorem](#part-b7--conditional-probability-and-bayes-theorem)
8. [Probability distributions](#part-b8--probability-distributions)
9. [Descriptive statistics](#part-b9--descriptive-statistics)
10. [Correlation and covariance](#part-b10--correlation-and-covariance)

**Part C — The Core ML Math**
11. [Decision boundaries — linear and quadratic](#part-c11--decision-boundaries)
12. [Logistic regression — full derivation](#part-c12--logistic-regression)
13. [Naive Bayes — full derivation](#part-c13--naive-bayes)
14. [Loss functions](#part-c14--loss-functions)
15. [Gradient descent](#part-c15--gradient-descent)
16. [Regularisation — L1 and L2](#part-c16--regularisation)
17. [Feature engineering and product features](#part-c17--feature-engineering)

**Part D — Evaluation and Model Selection**
18. [Model evaluation metrics](#part-d18--model-evaluation-metrics)
19. [Cross-validation](#part-d19--cross-validation)

**Part E — CS-LLN Specific Math**
20. [The ΔR criterion — full worked calculation](#part-e20--the-δr-criterion)
21. [Why product terms cure the NB blind spot](#part-e21--why-product-terms-cure-naive-bayes)

---

## PART A — Pure Math

---

## Part A.1 — Numbers, algebra, and functions

### The number types you'll see

| Name | Example | Where it appears |
|---|---|---|
| Integer | −3, 0, 7 | class labels, counts |
| Real | 3.14, −0.007 | feature values, weights |
| Positive real | 0.0001, 99.9 | probabilities (bounded 0–1), log-loss |

### Functions

A **function** `f(x)` takes an input and returns an output. The ones that
appear constantly in ML:

| Function | Formula | Key property |
|---|---|---|
| Linear | `f(x) = mx + c` | straight line |
| Exponential | `f(x) = eˣ` | always positive; `e ≈ 2.718` |
| Natural log | `f(x) = ln(x)` | inverse of `eˣ`; only defined for `x > 0` |
| Sigmoid | `f(x) = 1/(1+e⁻ˣ)` | squashes any real to (0,1) |

**Worked: simplify `e^(ln 5)`.**

`ln` and `e^` cancel each other (they're inverses): `e^(ln 5) = 5`. Always.

**Worked: solve `ln(x) = −2` for `x`.**

Apply `e^` to both sides: `x = e^(−2) = 1/e² ≈ 0.135`.

### Summation notation

`Σᵢ xᵢ` means "add up all xᵢ". Example: `Σᵢ₌₁³ i² = 1² + 2² + 3² = 14`.

---

## Part A.2 — Vectors

A **vector** is an ordered list of numbers. Think of it as a point in space or
a list of feature values.

```
x = [x₁, x₂, …, x_d]     (a row or column of d numbers)
```

### Operations

**Addition** — add element-wise:
```
[1, 2] + [3, −1] = [4, 1]
```

**Scalar multiplication** — multiply every element:
```
3 · [1, 2] = [3, 6]
```

**Dot product** — multiply element-wise then sum:
```
[2, 3] · [4, −1] = 2·4 + 3·(−1) = 8 − 3 = 5
```

**Norm (length)** of vector `x`:
```
‖x‖ = √(x₁² + x₂² + … + x_d²)
```

**Worked.** `x = [3, 4]`. `‖x‖ = √(9 + 16) = √25 = 5`.

### The dot product as projection

`w · x = ‖w‖ · ‖x‖ · cos(θ)` where `θ` is the angle between them.
- When `w · x = 0`, the vectors are **perpendicular**.
- When `w · x > 0`, they point in roughly the same direction.
- When `w · x < 0`, they point in roughly opposite directions.

This is the geometric reason a linear classifier sorts points: `w · x + b`
measures how far along the direction `w` the point `x` sits.

### Unit vectors

A **unit vector** has norm 1. Divide any vector by its norm to get one:
`û = x / ‖x‖`. Example: `[3,4] / 5 = [0.6, 0.8]`.

---

## Part A.3 — Matrices

A **matrix** is a 2-D grid of numbers. Notation: `A ∈ ℝ^(m×n)` = m rows, n cols.

```
A = [ 1  2  3 ]    ← 2 rows, 3 columns  (2×3 matrix)
    [ 4  5  6 ]
```

`A[i,j]` or `Aᵢⱼ` = element in row i, column j. `A[1,2] = 2`.

### Matrix–vector product

If `A` is `m×n` and `x` is a column vector of length `n`, then `Ax` is a
column vector of length `m`. Row `i` of the result = dot product of row `i`
of `A` with `x`:

```
A = [ 2  1 ]     x = [ 3 ]
    [ 0  3 ]         [ 4 ]
    [ 1 −1 ]

Row 1: 2·3 + 1·4 = 10
Row 2: 0·3 + 3·4 = 12
Row 3: 1·3 + (−1)·4 = −1

Ax = [ 10 ]
     [ 12 ]
     [ −1 ]
```

In ML this computes **all dot products at once**. The score for every training
example is `Xw` where `X` is the data matrix (each row = one example) and
`w` is the weight vector.

### Transpose

Flip rows and columns. `A^T[i,j] = A[j,i]`.

```
A = [ 1  2  3 ]    A^T = [ 1  4 ]
    [ 4  5  6 ]          [ 2  5 ]
                         [ 3  6 ]
```

### Why matrices matter for CS-LLN

The correlation matrix `R^(c) ∈ ℝ^(d×d)` is a matrix: `R[i,j]` = Pearson
correlation of features `i` and `j` within class `c`. `R` is **symmetric**
(`R[i,j] = R[j,i]`) and has 1s on the diagonal. Computing it for each class
is the whole of Step 2.

---

## Part A.4 — Calculus

You only need three things from calculus for this project.

### 4.1 The derivative — what it means

The derivative `f'(x)` (or `df/dx`) is the **slope of `f` at point `x`**.

Key rules:

| Function `f(x)` | Derivative `f'(x)` |
|---|---|
| `c` (constant) | `0` |
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `ln(x)` | `1/x` |
| `f(x) + g(x)` | `f'(x) + g'(x)` |
| `c · f(x)` | `c · f'(x)` |
| `f(g(x))` | `f'(g(x)) · g'(x)` ← **chain rule** |

**Worked: differentiate `f(x) = 3x² + 2x + 7`.**
```
f'(x) = 3·2x + 2·1 + 0 = 6x + 2
```
At `x = 1`: `f'(1) = 8` (the curve has slope 8 at that point).

**Worked: differentiate `f(x) = ln(1 + e^(−x))` (appears in logistic loss).**

Let `u = 1 + e^(−x)`, then `f = ln(u)`.
- `df/du = 1/u`
- `du/dx = −e^(−x)` (chain rule: derivative of `e^(−x)` is `−e^(−x)`)
- So `f'(x) = (1/u) · (−e^(−x)) = −e^(−x) / (1 + e^(−x)) = −(1 − σ(x))`

This result shows up directly when deriving the gradient of the log-loss.

### 4.2 Partial derivatives

When a function depends on multiple variables, the **partial derivative**
`∂f/∂xᵢ` is the derivative with respect to `xᵢ` while **treating all other
variables as constants**.

**Worked: `f(w₁, w₂) = w₁² + 3w₁w₂ + w₂³`.**
```
∂f/∂w₁ = 2w₁ + 3w₂       (treat w₂ as a constant)
∂f/∂w₂ = 3w₁ + 3w₂²      (treat w₁ as a constant)
```

### 4.3 The gradient

The **gradient** `∇f` is the vector of all partial derivatives — it points in
the direction of steepest ascent of `f`:

```
∇f(w) = [ ∂f/∂w₁,  ∂f/∂w₂,  …,  ∂f/∂w_d ]
```

For the function above at `(w₁, w₂) = (1, 2)`:
```
∂f/∂w₁ = 2(1) + 3(2) = 8
∂f/∂w₂ = 3(1) + 3(4) = 15
∇f(1,2) = [8, 15]
```

**Gradient descent** moves in the direction `−∇f` (downhill).

---

## Part A.5 — Exponents and Logarithms

These rules appear in every probability calculation.

### Rules to memorise

| Rule | Formula |
|---|---|
| Product | `ln(ab) = ln a + ln b` |
| Quotient | `ln(a/b) = ln a − ln b` |
| Power | `ln(aᵇ) = b · ln a` |
| Inverse | `e^(ln a) = a`,  `ln(eᵃ) = a` |
| Special | `ln(1) = 0`,  `e⁰ = 1` |

**Why we take log of probabilities:** Probabilities are products of many small
numbers — multiplied together they underflow to zero on a computer.
`ln(P₁ · P₂ · … · Pₙ) = ln P₁ + ln P₂ + … + ln Pₙ` (a sum, safe to compute).
Also: **maximising `ln P` is identical to maximising `P`**, since `ln` is
monotonically increasing.

**Worked: simplify `ln(P(x|y)) = ln(0.3) + ln(0.7)`.**
```
= ln(0.3 · 0.7) = ln(0.21) ≈ −1.561
```
This is the log-likelihood computation inside Naive Bayes.

---

## PART B — Probability and Statistics

---

## Part B.6 — Probability Fundamentals

### What probability means

`P(A) = 0.3` means: if you repeated the experiment many times, A would happen
about 30% of the time.

**Axioms:**
- `0 ≤ P(A) ≤ 1`
- `P(everything) = 1`
- `P(A or B) = P(A) + P(B)` if A and B can't both happen (mutually exclusive)

### The complement

`P(not A) = 1 − P(A)`.

**Example:** P(tumour is malignant) = 0.37 → P(benign) = 0.63.

### The addition rule (general)

`P(A or B) = P(A) + P(B) − P(A and B)`

If A and B are **mutually exclusive** (can't both happen), `P(A and B) = 0`,
so `P(A or B) = P(A) + P(B)`.

### Independence

`A` and `B` are **independent** if:
```
P(A and B) = P(A) · P(B)
```

**Worked.** P(die shows 4) = 1/6. P(coin shows heads) = 1/2.
Since the die and coin don't affect each other:
`P(4 AND heads) = 1/6 · 1/2 = 1/12`.

### Conditional independence (the Naive Bayes assumption)

`A` and `B` are **conditionally independent given C** if:
```
P(A and B | C) = P(A | C) · P(B | C)
```

This does **not** require `A` and `B` to be independent overall — just that
knowing `C` breaks any remaining link between them. Naive Bayes assumes every
pair of features satisfies this, given the class.

---

## Part B.7 — Conditional Probability and Bayes Theorem

### Conditional probability

```
P(A | B)  =  P(A and B) / P(B)
```

Read: "probability of A, *given* B has happened."

**Worked.** A bag has 3 red, 2 blue balls. You pick one without looking.
- `P(red) = 3/5`
- You're told it's not blue. Remaining: 3 red, 0 blue.
- `P(red | not blue) = 3/3 = 1`

Using the formula: `P(A and B)` = P(red and not-blue) = P(red) = 3/5.
`P(not blue) = 3/5`. `P(red | not blue) = (3/5)/(3/5) = 1`. ✓

### The multiplication rule

Rearranging the definition: `P(A and B) = P(A | B) · P(B) = P(B | A) · P(A)`.

### Bayes Theorem

Equating the two ways to write `P(A and B)`:

```
P(A | B) · P(B) = P(B | A) · P(A)
```
```
        P(B | A) · P(A)
P(A|B) = ─────────────────
               P(B)
```

In ML language (A = class, B = features):

```
         P(features | class) · P(class)
P(class | features) = ─────────────────────────
                            P(features)
```

- **Prior** `P(class)` — how common the class is before seeing features.
- **Likelihood** `P(features | class)` — how typical these features are for
  this class.
- **Evidence** `P(features)` — same for all classes, so ignored when comparing.

**Worked classification example.**

Disease prevalence: 1%. Test is 99% accurate (same for positive/negative).
You test positive. What's P(disease | positive)?

```
P(disease)   = 0.01       P(no disease)  = 0.99
P(+ | disease) = 0.99     P(+ | no disease) = 0.01

P(disease | +) = P(+ | disease) · P(disease)
                 ─────────────────────────────────────────
                 P(+ | disease)·P(disease) + P(+ | no disease)·P(no disease)

               = 0.99 · 0.01
                 ──────────────────────────────
                 0.99·0.01 + 0.01·0.99

               = 0.0099 / (0.0099 + 0.0099) = 0.0099 / 0.0198 = 0.5
```

Even with a 99% accurate test, the answer is only 50%! Why? Because disease
is rare — most positives come from the 99% of healthy people who still get a
false positive. **This is why priors matter enormously.**

---

## Part B.8 — Probability Distributions

### Discrete distributions

A **discrete** distribution assigns probabilities to countable outcomes
(e.g. which class a sample belongs to).

| Outcome | Probability |
|---|---|
| Setosa | 0.33 |
| Versicolor | 0.33 |
| Virginica | 0.34 |

All probabilities sum to 1. ✓

### The Gaussian (Normal) distribution

Used by **Gaussian Naive Bayes** to model each feature within each class.

```
P(x | μ, σ) = (1 / σ√(2π)) · exp(−(x − μ)² / (2σ²))
```

- `μ` = mean (centre of the bell curve)
- `σ` = standard deviation (width of the bell curve)
- `σ²` = variance

**Worked.** Feature "petal length" in Setosa: `μ = 1.46`, `σ = 0.17`.
What is `P(petal length = 1.5 | Setosa)`?

```
z = (1.5 − 1.46) / 0.17 = 0.235
P ≈ (1 / (0.17 · 2.507)) · exp(−0.235²/2)
  = (1 / 0.426) · exp(−0.0276)
  = 2.347 · 0.9727
  ≈ 2.28
```

Note: this is a **probability density**, so it can be > 1. We only ever
compare it across classes — we never interpret it as a standalone probability.

### The 68-95-99.7 rule

For a Gaussian, approximately:
- 68% of values fall within `μ ± 1σ`
- 95% fall within `μ ± 2σ`
- 99.7% fall within `μ ± 3σ`

A value more than 3σ from the mean is very rare. If two classes have different
means, a point far from one mean is much more likely to belong to the other.

---

## Part B.9 — Descriptive Statistics

### Mean (average)

```
x̄ = (x₁ + x₂ + … + xₙ) / n = (1/n) Σᵢ xᵢ
```

**Worked.** `x = [2, 5, 7, 8]`. `x̄ = 22/4 = 5.5`.

### Variance

The average squared distance from the mean:

```
σ² = (1/n) Σᵢ (xᵢ − x̄)²
```

**Worked.** `x̄ = 5.5`, `x = [2, 5, 7, 8]`:

```
Deviations: [2−5.5, 5−5.5, 7−5.5, 8−5.5] = [−3.5, −0.5, 1.5, 2.5]
Squared:    [12.25, 0.25, 2.25, 6.25]
σ² = (12.25 + 0.25 + 2.25 + 6.25) / 4 = 21/4 = 5.25
```

### Standard deviation

`σ = √(σ²) = √5.25 ≈ 2.29`. Same units as the original data.

### Sample vs population

If `x` is a **sample** from a larger population, use `n−1` (not `n`) in the
denominator — this is **Bessel's correction**, making the estimate less biased.
For n = 4, sample variance = `21/3 = 7`.

**In ML**, most implementations use `n−1` for statistics estimated from a
training set. Pearson correlation is the same either way (numerator and
denominator both change by the same factor).

### Z-score (standardisation)

```
zᵢ = (xᵢ − x̄) / σ
```

After z-scoring, `x̄ = 0` and `σ = 1` exactly.

**Why we do it:** features measured in very different units (temperature in °C
vs area in cm²) would dominate the dot product computation by magnitude alone.
Z-scoring puts everyone on the same footing. Also required for the L2 penalty
to be fair across features.

**Worked.** Original: `[2, 5, 7, 8]`, `x̄ = 5.5`, `σ = 2.29`.
```
Z-scores: [(2−5.5)/2.29, (5−5.5)/2.29, (7−5.5)/2.29, (8−5.5)/2.29]
        = [−1.53,        −0.22,          0.65,          1.09]
```
Check: mean of z-scores ≈ 0 ✓, std ≈ 1 ✓.

---

## Part B.10 — Correlation and Covariance

### Covariance

Measures how two features move together:

```
Cov(A, B) = (1/n) Σᵢ (Aᵢ − Ā)(Bᵢ − B̄)
```

- Positive: A and B tend to rise together.
- Negative: A rises when B falls.
- Zero: no *linear* co-movement.

**Worked.** `A = [1, 2, 3, 4]`, `B = [4, 3, 2, 1]` (they move opposite).

`Ā = 2.5`, `B̄ = 2.5`.

| Aᵢ−Ā | Bᵢ−B̄ | product |
|---|---|---|
| −1.5 | +1.5 | −2.25 |
| −0.5 | +0.5 | −0.25 |
| +0.5 | −0.5 | −0.25 |
| +1.5 | −1.5 | −2.25 |
| Σ | | **−5** |

`Cov(A, B) = −5/4 = −1.25`. Negative — confirms opposite movement.

### Pearson correlation coefficient

Covariance divided by the product of the standard deviations — **removes units**,
forces the result into `[−1, 1]`:

```
r = Cov(A, B) / (σ_A · σ_B) = Σᵢ(Aᵢ−Ā)(Bᵢ−B̄) / √(Σᵢ(Aᵢ−Ā)² · Σᵢ(Bᵢ−B̄)²)
```

**Continuing the worked example.** `σ_A = σ_B = √(5/4) = √1.25 ≈ 1.118`.

```
r = −1.25 / (1.118 · 1.118) = −1.25 / 1.25 = −1.0
```

`r = −1`: perfect negative linear relationship. ✓

### Interpreting r

| r | Relationship |
|---|---|
| +1.0 | Perfect positive line |
| +0.7 to +0.9 | Strong positive |
| +0.3 to +0.7 | Moderate positive |
| 0 to +0.3 | Weak/no linear |
| 0 | No *linear* relationship (could still be nonlinear!) |
| −0.3 to −1.0 | (same, but negative) |

### Critical insight: r = 0 does not mean independent

`A = [−2, −1, 0, 1, 2]`, `B = [4, 1, 0, 1, 4] = A²`.

`Ā = 0`. Σ(A−Ā)(B−B̄) = (−2)(2) + (−1)(−1) + (0)(−2) + (1)(−1) + (2)(2) = −4+1+0−1+4 = **0**.

So `r = 0`, yet `B` is *entirely determined* by `A`. Pearson only sees
**straight-line** (linear) dependence. Curved, quadratic, or any nonlinear
dependence is invisible to it. **This is the key limitation of ΔR in CS-LLN.**

### The correlation matrix

For `d` features, `R ∈ ℝ^(d×d)` where `R[i,j] = Pearson(xᵢ, xⱼ)`.
- Always symmetric: `R[i,j] = R[j,i]`
- Diagonal = 1: `R[i,i] = Pearson(xᵢ, xᵢ) = 1`

**Worked (3 features, 4 samples).**

```
X:   x₁  x₂  x₃
     1    2    5
     2    4    4
     3    6    3
     4    8    2

x₁ = [1,2,3,4]   x̄₁ = 2.5
x₂ = [2,4,6,8]   x̄₂ = 5.0     (x₂ = 2x₁ → r(x₁,x₂) = +1)
x₃ = [5,4,3,2]   x̄₃ = 3.5     (x₃ decreases as x₁ increases → negative r)
```

By inspection (or calculation): R =
```
    x₁    x₂    x₃
x₁ [ 1.0  +1.0  −1.0 ]
x₂ [+1.0   1.0  −1.0 ]
x₃ [−1.0  −1.0   1.0 ]
```

This is the kind of matrix CS-LLN computes *separately for each class*.

---

## PART C — The Core ML Math

---

## Part C.11 — Decision Boundaries

*(The full treatment. Summary of what's in MATH_FOUNDATIONS_BOUNDARIES.md.)*

A **decision boundary** is the surface in feature space where a classifier
assigns equal probability to two classes — the exact line/curve of indecision.

### Linear boundary (degree 1)

The model score is `z = w₁x₁ + w₂x₂ + b`. Boundary: `z = 0`.
In 2-D: `x₂ = −(w₁/w₂)x₁ − b/w₂` — a **straight line**.

### How the weights control the line

- `w₁/w₂` = the **slope** of the boundary line.
- `b/w₂` = the **intercept** (vertical shift).
- The weight vector `[w₁, w₂]` is **perpendicular** to the boundary.

**Worked.** `w = [1, 1]`, `b = −3`. Boundary: `x₁ + x₂ = 3`.
- Point `[1, 1]`: `z = 1 + 1 − 3 = −1 < 0` → class B (below the line).
- Point `[2, 2]`: `z = 2 + 2 − 3 = 1 > 0` → class A (above the line).
- Point `[1, 2]`: `z = 1 + 2 − 3 = 0` → exactly on the boundary.

### Quadratic boundary (degree 2)

Add squared and cross terms. The boundary `z = 0` can now be an ellipse,
parabola, or hyperbola. Full derivation in Section C.17.

### The fundamental principle

> **The boundary's shape = the polynomial degree of the score function.**

Degree 1 → flat. Degree 2 → curved. Adding product features `xᵢxⱼ` to a
linear model raises the effective degree to 2 **in the original space**, while
keeping all training mechanics linear (convex, easy to optimise).

---

## Part C.12 — Logistic Regression

### From score to probability

The raw score `z = w·x + b` can be any real number. We need a probability in
`[0, 1]`. The **sigmoid** function does this:

```
σ(z) = 1 / (1 + e^(−z))
```

Properties:
- `σ(0) = 0.5` — the decision boundary.
- `σ(z) → 1` as `z → ∞`.
- `σ(z) → 0` as `z → −∞`.
- Derivative: `σ'(z) = σ(z)(1 − σ(z))` — used in gradient derivation.

**Worked (prove the derivative).**
```
σ(z) = (1 + e^(−z))^(−1)

σ'(z) = −(1 + e^(−z))^(−2) · (−e^(−z))     [chain rule]
       = e^(−z) / (1 + e^(−z))²
       = [1/(1+e^(−z))] · [e^(−z)/(1+e^(−z))]
       = σ(z) · (1 − σ(z))   ✓
```

### Multi-class: softmax

Each class `c` gets its own weight vector `wc` and bias `bc`:

```
z_c = wc · x + bc

P(y = c | x) = exp(z_c) / Σ_k exp(z_k)
```

All class probabilities sum to 1. The predicted class is the one with the
highest `z_c`.

**Worked.** 3 classes, `z₀ = 1, z₁ = 2, z₂ = 0.5`.
```
exp(1) = 2.718,  exp(2) = 7.389,  exp(0.5) = 1.649
Sum = 11.756

P(0) = 2.718/11.756 = 0.231
P(1) = 7.389/11.756 = 0.628     ← predicted class
P(2) = 1.649/11.756 = 0.140
Sum = 0.999  ≈ 1  ✓
```

### The decision boundary proof

Boundary between classes A and B: where `P(A|x) = P(B|x)`, i.e. odds = 1,
i.e. log-odds = 0:

```
ln[P(A|x)/P(B|x)] = 0
```

For logistic regression `P(A|x) = σ(z)`, `P(B|x) = 1−σ(z)`:
```
ln[σ(z)/(1−σ(z))] = z = w·x + b = 0
```

The boundary is `w·x + b = 0`. This is **degree 1** → straight. **QED.**

---

## Part C.13 — Naive Bayes

### The algorithm

**Predict** by Bayes' theorem (ignoring constant denominator):
```
ŷ = argmax_c  [ ln P(y=c) + Σᵢ ln P(xᵢ | y=c) ]
```

**Gaussian NB**: model `P(xᵢ | y=c) = Gaussian(xᵢ; μᵢ^(c), σᵢ^(c))`.

**Train**: estimate from the data:
```
μᵢ^(c) = mean of feature i in class c
σᵢ^(c) = std  of feature i in class c
P(y=c)  = nᶜ / n         (fraction of training examples in class c)
```

**Worked (full 2-class, 2-feature example).**

Training data:

| x₁ | x₂ | y |
|---|---|---|
| 1  | 2  | 0 |
| 1  | 3  | 0 |
| 2  | 2  | 0 |
| 5  | 1  | 1 |
| 5  | 2  | 1 |
| 6  | 1  | 1 |

**Step 1: priors.** `P(y=0) = 3/6 = 0.5`, `P(y=1) = 3/6 = 0.5`.

**Step 2: class-conditional means and stds.**
```
Class 0:  μ₁ = (1+1+2)/3 = 4/3 ≈ 1.33,   μ₂ = (2+3+2)/3 = 7/3 ≈ 2.33
          σ₁ = √([(1−4/3)²+(1−4/3)²+(2−4/3)²]/3)
             = √[(1/9 + 1/9 + 4/9)/3] = √[6/27] ≈ 0.47
          σ₂ ≈ 0.47

Class 1:  μ₁ = (5+5+6)/3 = 16/3 ≈ 5.33,   μ₂ = (1+2+1)/3 = 4/3 ≈ 1.33
          σ₁ ≈ 0.47,   σ₂ ≈ 0.47
```

**Step 3: classify new point `x = [3, 2]`.**

Log-score for class 0:
```
ln P(y=0) = ln(0.5) = −0.693
ln P(x₁=3 | y=0) = ln[Gaussian(3; 1.33, 0.47)]
                  = −ln(0.47√(2π)) − (3−1.33)²/(2·0.47²)
                  = −1.077 − (1.67)²/0.442
                  = −1.077 − 6.31 = −7.39
ln P(x₂=2 | y=0) = ln[Gaussian(2; 2.33, 0.47)]
                  = −1.077 − (−0.33)²/0.442
                  = −1.077 − 0.247 = −1.32

Score(0) = −0.693 + (−7.39) + (−1.32) = −9.40
```

Log-score for class 1:
```
ln P(x₁=3 | y=1): z = (3−5.33)/0.47 = −4.96 → very low probability
Score(1) = −0.693 + very_negative + (−1.32)  ← even lower
```

Predict **class 0**. Point `[3, 2]` is closer to class 0's feature means, so
this makes intuitive sense.

### Why NB breaks (mathematically)

The assumption `P(x | y) = Π P(xᵢ | y)` is equivalent to assuming a
**diagonal** covariance matrix within each class — off-diagonal elements
(covariances between pairs of features) are all set to zero.

Actual score with the Gaussian:
```
ln P(x | y) = −(1/2)(x − μ)^T Σ^(−1) (x − μ) − (1/2)ln|Σ| − (d/2)ln(2π)
```

NB forces `Σ = diag(σ₁², σ₂², …)`. When the true covariance has large
off-diagonal terms that **differ between classes**, the diagonal approximation
gives the wrong decision boundary — see Section E.21.

---

## Part C.14 — Loss Functions

A **loss function** measures how wrong the model is. Training = minimise it.

### Binary cross-entropy (log-loss)

For binary classification with true label `y ∈ {0,1}` and predicted probability
`ŷ = P(y=1|x)`:

```
L = −[y · ln(ŷ) + (1−y) · ln(1−ŷ)]
```

- If `y = 1` and `ŷ = 0.9`: `L = −ln(0.9) = 0.105` (small loss, good prediction)
- If `y = 1` and `ŷ = 0.1`: `L = −ln(0.1) = 2.303` (large loss, bad prediction)
- If `y = 1` and `ŷ → 0`: `L = −ln(0) → ∞` (infinite loss, catastrophic)

**Why log?** Because we want the model to be **confident when right**. If you
predict 50% when you should predict 95%, the log-loss still penalises you even
though the classification is correct.

**Worked.** Three predictions, labels and probabilities:

| y | ŷ | L = −[y·ln(ŷ) + (1−y)·ln(1−ŷ)] |
|---|---|---|
| 1 | 0.8 | −ln(0.8) = 0.223 |
| 0 | 0.3 | −ln(1−0.3) = −ln(0.7) = 0.357 |
| 1 | 0.2 | −ln(0.2) = 1.609 ← costly |

Mean log-loss = (0.223 + 0.357 + 1.609) / 3 = **0.730**.

### Multi-class cross-entropy

With `C` classes and true label as a one-hot vector `[0,0,1,0]`:

```
L = −Σ_c  y_c · ln(P(y=c|x))
```

Since all `y_c = 0` except the true class, this reduces to
`L = −ln(P(true class | x))`. Minimising the loss = maximising the probability
assigned to the correct class.

### Mean Squared Error (MSE)

Used for regression (not this project, but good to know):
```
MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²
```

---

## Part C.15 — Gradient Descent

### The intuition

Imagine standing on a hilly landscape (the loss surface). You want to find the
lowest valley (minimum loss). Gradient descent: look at which direction is
steepest uphill, then step in the **opposite direction**.

```
w_new = w_old − η · ∇L(w_old)
```

- `η` (eta) = **learning rate** — how big a step to take. Too large: overshoot.
  Too small: slow convergence.
- `∇L` = gradient of the loss with respect to weights.

### Deriving the gradient for logistic regression

Loss for one example: `L = −[y·ln(ŷ) + (1−y)·ln(1−ŷ)]` where `ŷ = σ(w·x + b)`.

We need `∂L/∂w` and `∂L/∂b`. Using chain rule:

```
∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w)
```

Step 1: `∂L/∂ŷ = −y/ŷ + (1−y)/(1−ŷ)`

Step 2: `∂ŷ/∂z = σ(z)(1−σ(z)) = ŷ(1−ŷ)` (from the sigmoid derivative).

Step 3: `∂z/∂w = x` (since `z = w·x + b`).

Combining:
```
∂L/∂w = [−y/ŷ + (1−y)/(1−ŷ)] · ŷ(1−ŷ) · x
       = [−y(1−ŷ) + (1−y)ŷ] · x
       = (ŷ − y) · x
```

**Result: gradient of log-loss w.r.t. weights = `(prediction − truth) × feature`.**

This elegant formula is the reason logistic regression is easy to implement.
Similarly, `∂L/∂b = ŷ − y`.

### Full worked update

Setup: 3 training examples, `d = 2`.

```
X = [[2, 1],     y = [1, 0, 1]
     [1, 3],
     [3, 2]]

Start: w = [0, 0],  b = 0,  η = 0.1
```

**Iteration 1:**

For each example, compute `z`, `ŷ`, and error `(ŷ − y)`:

| i | x | z = w·x+b | ŷ = σ(z) | y | error = ŷ−y |
|---|---|---|---|---|---|
| 1 | [2,1] | 0 | 0.5 | 1 | −0.5 |
| 2 | [1,3] | 0 | 0.5 | 0 | +0.5 |
| 3 | [3,2] | 0 | 0.5 | 1 | −0.5 |

Gradient (average over all examples):
```
∂L/∂w₁ = (1/3)[(−0.5)·2 + (0.5)·1 + (−0.5)·3] = (1/3)[−1+0.5−1.5] = −2/3 ≈ −0.667
∂L/∂w₂ = (1/3)[(−0.5)·1 + (0.5)·3 + (−0.5)·2] = (1/3)[−0.5+1.5−1] = 0/3 = 0
∂L/∂b  = (1/3)[(−0.5) + (0.5) + (−0.5)] = −0.5/3 ≈ −0.167
```

Update:
```
w₁ ← 0 − 0.1·(−0.667) = +0.0667
w₂ ← 0 − 0.1·0 = 0
b  ← 0 − 0.1·(−0.167) = +0.0167
```

After iteration 1: `w = [0.067, 0]`, `b = 0.017`. The model has started to
learn that high `x₁` correlates with class 1.

### Batch vs stochastic vs mini-batch

| Variant | Update using | Pros | Cons |
|---|---|---|---|
| Batch GD | All N examples | Stable gradient | Slow for large N |
| Stochastic GD (SGD) | 1 example at a time | Fast | Noisy updates |
| Mini-batch | 32–256 examples | Balance | Most common in practice |

The `trainLR` in `app.js` uses **full-batch gradient descent** — fine for
small datasets like Iris (150 examples).

---

## Part C.16 — Regularisation

### The problem: overfitting

A model with too many parameters can **memorise** training data (high training
accuracy) but fail on new data (low test accuracy). Called **overfitting**.

Signs: training accuracy >> test accuracy; model makes very confident but wrong
predictions; weights become very large.

### L2 regularisation (Ridge)

Add a penalty term to the loss proportional to the **sum of squared weights**:

```
L_regularised = L_original + (λ/2) · Σⱼ wⱼ²  =  L_original + (λ/2) · ‖w‖²
```

`λ` (lambda) controls the strength of the penalty.

Effect on the gradient:
```
∂L_reg/∂wⱼ = ∂L_orig/∂wⱼ + λ · wⱼ
```

Update rule:
```
wⱼ ← wⱼ − η·(∂L_orig/∂wⱼ + λ·wⱼ) = wⱼ(1 − ηλ) − η·∂L_orig/∂wⱼ
```

The factor `(1 − ηλ)` is slightly less than 1, so weights are **shrunk** every
step. This prevents them from growing huge. L2 regularisation = **weight decay**.

In sklearn, the regularisation is parameterised as `C = 1/λ`. Small `C` → strong
regularisation (weights forced small). Large `C` → weak regularisation.

**Worked.** `w = 2.0`, `η = 0.1`, `λ = 0.5`, gradient = 0.3.
```
w ← 2.0 − 0.1·(0.3 + 0.5·2.0)
   = 2.0 − 0.1·(0.3 + 1.0)
   = 2.0 − 0.13 = 1.87
```
Without regularisation: `w ← 2.0 − 0.03 = 1.97`. L2 pulls the weight down harder.

### L1 regularisation (Lasso)

Penalty is `λ · Σ |wⱼ|` (sum of absolute values).

Effect: **sparse weights** — many weights go exactly to zero, effectively
performing feature selection. L2 rarely gives exactly zero weights; L1 does.

CS-LLN uses **L2** (in the logistic regression step). The sparsity in CS-LLN
comes from the **ΔR selection** (step 4), not from L1.

### Multicollinearity

When two features are nearly identical (or a product term duplicates information
already in the originals), the design matrix becomes ill-conditioned and weights
can blow up. L2 regularisation mitigates this by keeping weights bounded.

This is why adding **all** `C(d,2)` product terms to a linear model causes
problems — many pairs are highly redundant — and why CS-LLN's sparse selection
is essential.

---

## Part C.17 — Feature Engineering and Product Features

### Why engineer features?

**Raw features** might not expose the signal a linear model needs. Engineering
creates new, derived features from the raw ones.

### Polynomial features

For `d = 2`, degree-2 polynomial expansion adds: `x₁², x₂², x₁x₂`.
Full enriched vector: `x̃ = [x₁, x₂, x₁², x₂², x₁x₂]` — 5 features from 2.

For `d` features, full degree-2 gives `d + d + C(d,2) = d + d(d+1)/2` new
features. For `d = 30`: `30 + 435 = 465` features. **This is what CS-LLN avoids.**

### The product-feature trick in detail

Start: `x̃ = [x₁, x₂, x₁x₂]`. Linear logistic regression learns:
```
z = w₁x₁ + w₂x₂ + w₃(x₁x₂) + b
```

Boundary `z = 0` in original space (substituting back):
```
w₁x₁ + w₂x₂ + w₃x₁x₂ + b = 0
```

This is a **degree-2 equation** in `(x₁, x₂)` — the shape depends on `w₃`:

| w₃ | Boundary shape |
|---|---|
| 0 | Straight line |
| > 0 | Hyperbola opening one way |
| < 0 | Hyperbola opening the other way |
| Very large | Asymptotically two crossing lines |

**Worked: classify a point using an enriched model.**

`w = [−1, −1, 3]` (one weight per feature in `x̃ = [x₁, x₂, x₁x₂]`), `b = 0`.

Point `x = [2, 0.5]`:
```
x̃ = [2, 0.5, 2·0.5] = [2, 0.5, 1.0]
z  = −1·2 + (−1)·0.5 + 3·1.0 + 0 = −2 − 0.5 + 3 = +0.5
ŷ  = σ(0.5) = 1/(1+e^{−0.5}) = 1/(1+0.607) = 0.622 → class A
```

Point `x = [0.2, 3]`:
```
x̃ = [0.2, 3, 0.6]
z  = −0.2 − 3 + 1.8 = −1.4
ŷ  = σ(−1.4) = 0.197 → class B
```

Same weights, same model — but the product feature caused these two points to
land on opposite sides even though simple features might have classified them
similarly.

### CS-LLN's selection rule

Instead of adding all `C(d,2)` products, CS-LLN asks: **which interaction
changes most across classes?** It computes `ΔR[i,j]` for every pair and adds
only the top-k product terms. Those are the pairs where the class-specific
covariance structure most strongly distinguishes the classes.

---

## PART D — Evaluation and Model Selection

---

## Part D.18 — Model Evaluation Metrics

### The confusion matrix

For binary classification:

```
                 Predicted Positive   Predicted Negative
Actual Positive  True Positive (TP)   False Negative (FN)
Actual Negative  False Positive (FP)  True Negative (TN)
```

**Worked.** 100 test examples: 40 are malignant (positive), 60 are benign.
Model predicts: 35 TP, 5 FN, 8 FP, 52 TN.

```
           Pred +   Pred −
Actual +     35       5      ← 40 total malignant
Actual −      8      52      ← 60 total benign
```

### Accuracy

```
Accuracy = (TP + TN) / Total = (35 + 52) / 100 = 87/100 = 87%
```

Problem: **misleading on imbalanced data**. If 95% of samples are benign,
predicting "always benign" gives 95% accuracy — yet it catches zero cancers.

### Precision and Recall

```
Precision = TP / (TP + FP) = 35 / (35+8) = 35/43 ≈ 0.814
Recall    = TP / (TP + FN) = 35 / (35+5) = 35/40 = 0.875
```

- **Precision**: of all "malignant" predictions, how many were right? (False
  alarm rate)
- **Recall**: of all actual malignant cases, how many did we catch? (Miss rate)

For cancer: **recall is more important** — a missed cancer (FN) is worse than a
false alarm (FP).

### F1-score

Harmonic mean of precision and recall (punishes imbalance between them):

```
F1 = 2 · Precision · Recall / (Precision + Recall)
   = 2 · 0.814 · 0.875 / (0.814 + 0.875)
   = 1.4245 / 1.689
   ≈ 0.843
```

**Macro-averaged F1** (used in the paper): compute F1 separately for each class,
then average. This treats all classes equally regardless of size.

### Log-loss (cross-entropy loss)

Measures **calibration** — not just whether the prediction is correct, but
whether the **confidence** is appropriate:

```
Log-Loss = −(1/n) Σᵢ [yᵢ·ln(ŷᵢ) + (1−yᵢ)·ln(1−ŷᵢ)]
```

- Lower is better. 0 = perfect. Random classifier ≈ 0.693.
- A model that says "60% malignant" when it's actually malignant gets a smaller
  loss than one saying "90%" — but only if it's correctly ranked most of the time.

**Why GNB has terrible log-loss on Breast Cancer.** GNB ignores correlations,
so its class-conditional probabilities are poorly calibrated — it pushes
posteriors toward 0 or 1 even for ambiguous examples. CS-LLN's log-loss (0.078)
vs GNB's (0.777) reflects this: CS-LLN is **~10× better calibrated** on this
dataset (though this comparison is against GNB; against LR the gap is tiny).

---

## Part D.19 — Cross-Validation

### Why not just split once?

A single train/test split gives one number. If you happened to split unluckily
(hard examples in the train set, easy in test), the number is wrong. You'd never
know.

### K-fold cross-validation

1. Shuffle the data.
2. Split into `k` equal parts (folds).
3. For each fold: train on the other `k−1` folds, test on this fold.
4. Average the `k` test scores.

```
Fold 1: [TEST ] [train] [train] [train] [train]
Fold 2: [train] [TEST ] [train] [train] [train]
Fold 3: [train] [train] [TEST ] [train] [train]
…
```

Result: every example is tested exactly once, and you get `k` estimates of
performance → mean and standard deviation.

### Stratified k-fold

Ensures each fold has the same class proportion as the full dataset. Essential
when classes are imbalanced. **Always use stratified CV.** The paper uses
5-fold stratified CV throughout.

### Inner vs outer CV (nested CV)

Used when both selecting a hyperparameter (`k` in CS-LLN) **and** evaluating
the model:

```
Outer fold 1: test on fold 1
   Inner CV on folds 2-5:
      try k=1,2,3,… → pick best k by inner CV
   Train final model (with best k) on folds 2-5
   Evaluate on fold 1

Repeat for outer folds 2,3,4,5.
```

This ensures the test fold never influences hyperparameter choice. Without it,
you have **data leakage** and your accuracy estimate is optimistically biased.

### Worked: reading the results tables

From Table I (Breast Cancer):
```
CS-LLN:  97.72% ± 1.6
```

This means:
- The **mean accuracy** across 5 folds is 97.72%.
- The **standard deviation** across the 5 fold-level accuracies is 1.6pp.
- So fold-level accuracies were spread roughly in `[96.1%, 99.3%]`.
- The difference from LR (97.37% ± 1.7) is **0.35pp** — well within 1 std of
  either model. No significance test was run, so this margin should be read as
  "competitive", not "decisively better".

---

## PART E — CS-LLN Specific Math

---

## Part E.20 — The ΔR Criterion — Full Worked Calculation

We work through the entire ΔR computation on a tiny example.

### Setup

3 training examples per class, 2 features (`d = 2`), 2 classes (`C = 2`).

```
Class 0 (Benign):       Class 1 (Malignant):
  x₁   x₂                 x₁   x₂
  1.0  1.1                 3.0  2.9
  1.2  0.9                 2.8  3.1
  1.1  1.0                 3.2  3.0
```

**Step 1: Standardise (z-score over all 6 examples).**

```
All x₁: [1.0, 1.2, 1.1, 3.0, 2.8, 3.2]
mean_x₁ = (1.0+1.2+1.1+3.0+2.8+3.2)/6 = 12.3/6 = 2.05
std_x₁  = √(Σ(xᵢ−2.05)²/6)

Deviations² from 2.05:
  (1.0−2.05)² = 1.1025
  (1.2−2.05)² = 0.7225
  (1.1−2.05)² = 0.9025
  (3.0−2.05)² = 0.9025
  (2.8−2.05)² = 0.5625
  (3.2−2.05)² = 1.3225
  Sum = 5.515,  Var = 5.515/6 ≈ 0.919,  std ≈ 0.959

By symmetry x₂ has the same statistics (the classes are perfectly mirrored).
```

Standardised values (rounded):

```
Class 0:  [−1.10, −0.90], [−0.89, −1.15], [−0.99, −1.10]
Class 1:  [+0.99, +0.88], [+0.78, +1.10], [+1.20, +0.99]
```

**Step 2: Class-conditional correlation matrices.**

Within class 0, compute Pearson(`x₁_std`, `x₂_std`):

```
Class 0 standardised:
  x₁: [−1.10, −0.89, −0.99]    mean ≈ −0.993
  x₂: [−0.90, −1.15, −1.10]    mean ≈ −1.050

Deviations from within-class mean:
  x₁: [−0.107, +0.103, −0.003]
  x₂: [+0.150, −0.100, −0.050]

Cross-products: (−0.107)(+0.150) + (0.103)(−0.100) + (−0.003)(−0.050)
              = −0.0161 − 0.0103 + 0.00015 = −0.0263

Σ(x₁−x̄₁)² = 0.0114 + 0.0106 + 0.000009 = 0.0220
Σ(x₂−x̄₂)² = 0.0225 + 0.0100 + 0.0025 = 0.0350

r₀ = −0.0263 / √(0.0220 · 0.0350) = −0.0263 / √0.000770 = −0.0263/0.02775 ≈ −0.95
```

With only 3 samples the estimate is noisy, but the story is correct: in class 0
(benign), once both features are high they don't particularly rise together —
if anything slightly negatively correlated given the small variation.

Within class 1, by the symmetric structure, `r₁ ≈ +0.95`.

```
R⁽⁰⁾ = [ 1     −0.95 ]    R⁽¹⁾ = [ 1    +0.95 ]
         [−0.95  1    ]             [+0.95  1    ]
```

**Step 3: Compute ΔR.**

For `C = 2`, there is one class pair `(m=0, n=1)`:

```
ΔR₁₂ = (2/(2·1)) · |R⁽⁰⁾₁₂ − R⁽¹⁾₁₂|
       = 1 · |−0.95 − (+0.95)|
       = |−1.90| = 1.90
```

`ΔR₁₂ = 1.90` — very close to the maximum of 2. This pair **screams** context
shift: the correlation between features 1 and 2 goes from −0.95 in benign to
+0.95 in malignant. Naive Bayes treats this pair as uncorrelated in both classes.

**Step 4: Select top-k pairs.** With `d = 2` there is only one pair, so `Ωₖ = {(1,2)}`.

**Step 5: Enrich and classify.**

New feature: `x̃₃ = x₁_std · x₂_std`.

In class 0, both standardised values are negative → their product is **positive**.
In class 1, both are positive → product is also **positive**.

But the product is LARGE in class 1 (values ≈ +0.99 × +0.99 ≈ +0.98) and also
positive but moderate in class 0 (−1.0 × −1.0 ≈ +1.0). So the product alone may
not separate them here — but the **combination** of x₁, x₂, and x₁x₂ in
logistic regression creates a richer boundary.

Actually the key discriminative signal here is the *sign* of covariance (the
pattern above), and the logistic regression will learn weights that exploit it.

---

## Part E.21 — Why Product Terms Cure the NB Blind Spot

### The NB score with a correlated pair

Suppose features `x₁, x₂` have correlation `ρ^(c)` in class `c`. The **true**
log-likelihood under a bivariate Gaussian is:

```
ln P(x₁, x₂ | y=c) = −(1/2) [x₁_z², x₂_z] Σ_c^{−1} [x₁_z, x₂_z]^T + const
```

where `Σ_c` is the 2×2 class-conditional covariance and `x_z` are z-scored features.

For the correlated 2-feature case:

```
Σ_c = [ 1     ρ^(c) ]       Σ_c^{−1} = 1/(1−ρ^(c)²) · [ 1       −ρ^(c) ]
      [ ρ^(c)  1    ]                                    [ −ρ^(c)   1     ]
```

Expanding the quadratic:

```
True ln P(x|c) ∝ −(x₁_z² + x₂_z² − 2ρ^(c)·x₁_z·x₂_z) / (2(1−ρ^(c)²)) + const_c
```

The cross-term `−2ρ^(c)·x₁_z·x₂_z / (1−ρ^(c)²)` is exactly the coefficient on the
**product feature** `x₁·x₂`. NB drops this term entirely (sets the off-diagonal
of `Σ` to 0).

### The decision boundary error

Binary classification boundary: `ln P(x|0) − ln P(x|1) = 0`. The NB boundary
(ignoring the cross-term) is a line:

```
NB boundary: w₁x₁ + w₂x₂ + b = 0
```

The **true** boundary (with correct Gaussian) adds the cross-term:

```
True boundary: w₁x₁ + w₂x₂ + α·x₁x₂ + b' = 0
```

where `α = −[ρ^(0)/(1−ρ^(0)²)] + [ρ^(1)/(1−ρ^(1)²)]`.

**If `ρ^(0) = ρ^(1)`, then `α = 0`** — the NB boundary is exact. NB is harmlessly
wrong.

**If `ρ^(0) ≠ ρ^(1)`, then `α ≠ 0`** — NB misses the cross-term, its boundary
bends the wrong way. The larger `|ρ^(0) − ρ^(1)|` = `ΔR` for this pair, the
larger `|α|`, and the more wrong NB is.

**CS-LLN restores `α`** by adding `x₁x₂` as a feature. Logistic regression then
learns the correct `α` (and any contribution from other pairs) through training.

### Numeric demonstration

Class 0: `ρ^(0) = 0`, `α₀ = 0/(1−0) = 0`.
Class 1: `ρ^(1) = 0.9`, `α₁ = 0.9/(1−0.81) = 0.9/0.19 ≈ 4.74`.

```
α = −α₀ + α₁ = −0 + 4.74 = 4.74
```

The product term `x₁x₂` should have weight 4.74 in the correct decision
function. NB sets this to 0. The classification error introduced is proportional
to how far this missing term bends the boundary — and `ΔR₁₂ = |0 − 0.9| = 0.9`
directly flags this pair as the one to fix.

---

## Formula Cheat-Sheet

| Concept | Formula |
|---|---|
| Dot product | `w·x = Σᵢ wᵢxᵢ` |
| Linear score | `z = w·x + b` |
| Sigmoid | `σ(z) = 1/(1+e^{−z})`;  `σ'(z) = σ(z)(1−σ(z))` |
| Softmax | `P(c) = e^{zc}/Σ e^{zk}` |
| Decision boundary | `z = 0` (binary);  `za = zb` (multi-class) |
| Mean | `x̄ = (1/n)Σxᵢ` |
| Variance | `σ² = (1/n)Σ(xᵢ−x̄)²` |
| Z-score | `(x−x̄)/σ` |
| Pearson r | `Σ(xᵢ−x̄)(yᵢ−ȳ) / √(Σ(xᵢ−x̄)²·Σ(yᵢ−ȳ)²)` |
| Gaussian density | `(1/σ√2π)·exp(−(x−μ)²/2σ²)` |
| Log-loss (binary) | `−[y·ln(ŷ)+(1−y)·ln(1−ŷ)]` |
| LR gradient | `∂L/∂w = (ŷ−y)x`,  `∂L/∂b = ŷ−y` |
| L2 gradient addition | `+λwⱼ` |
| Gradient step | `w ← w − η·∂L/∂w` |
| ΔR score | `(2/C(C−1))·Σ_{m<n}·|R^(m)ij − R^(n)ij|` |
| Accuracy | `(TP+TN)/N` |
| Precision | `TP/(TP+FP)` |
| Recall | `TP/(TP+FN)` |
| F1 | `2·P·R/(P+R)` |

**The five sentences that connect everything:**
1. A classifier is a score function; its boundary is where the score is zero.
2. The boundary's shape = the polynomial degree of the score.
3. Adding product features `xᵢxⱼ` raises the degree; a linear model draws curved boundaries.
4. Which products to add? The ones where class-conditional Pearson correlation differs most — measured by ΔR.
5. ΔR = 0 means NB is consistently wrong (harmless); ΔR >> 0 means NB is class-specifically wrong (distorts the boundary) — those are the pairs CS-LLN fixes.
