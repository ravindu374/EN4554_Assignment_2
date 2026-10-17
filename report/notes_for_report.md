# Source material for the LaTeX report

Prose removed from `EN4554_a02.ipynb` during cleanup. Each block below records the
original notebook cell number. Measured numbers are in `results/*.json`.



<!-- ===== cell 1 (TRIMMED) ===== -->

# EN4554 — Assignment 2 on CNNs

**Name:** Lakshan R.G.R\
**Index number:** 220355M\
**GitHub:** https://github.com/ravindu374/EN4554_Assignment_2

---

### Acknowledgement of AI tool use

Claude (Anthropic) was used as a coding assistant while preparing this notebook: to scaffold the
layer classes, the gradient-checking harness and the plotting code. All implementations were
reviewed, run and verified by me, and all interpretations of the results are my own. I understand
every line of the code presented here.

---

### Contents
- **Task 1** — Backpropagation from scratch 
- **Task 2** — Optimizers 
- **Task 3** — Regularization and residual connections 
- **Task 4** — Best CNN under 500k parameters


<!-- ===== cell 3 (TRIMMED) ===== -->

### Helpers for saving and reloading results

Training runs are slow and the notebook has to be re-exported to PDF several times. Every experiment
therefore **writes its history to `results/*.json`**, and the plotting cells **read from those files**.
Re-running the notebook to fix a typo in a plot label does not retrain anything.

Set `FORCE_RETRAIN = True` to ignore the cache and rerun.


<!-- ===== cell 6 (TRIMMED) ===== -->

---
# Task 1: Backpropagation

The goal is to implement the forward **and** backward pass of a small MLP by hand, with no autograd,
and then prove the gradients are right.

## The three derivatives I need

Write $g = \dfrac{\partial L}{\partial y}$ for the gradient arriving from the layer above.

### 1. Linear layer, $y = xW + b$

With $x \in \mathbb{R}^{N \times d_{in}}$, $W \in \mathbb{R}^{d_{in} \times d_{out}}$, $b \in \mathbb{R}^{d_{out}}$:

$$\frac{\partial L}{\partial x} = g W^\top, \qquad
  \frac{\partial L}{\partial W} = x^\top g, \qquad
  \frac{\partial L}{\partial b} = \sum_{n=1}^{N} g_n$$

The easy way to remember the first two: *to differentiate a matrix product, multiply by the other
factor, transposed* — and the shapes only work out one way, which is a useful check. The bias was
added to every row of the batch, so its gradient is the sum down the rows.

### 2. ReLU, $y = \max(x, 0)$

ReLU is a gate. Where the input was positive it passed the signal forward, so it passes the gradient
back; where the input was negative it blocked the signal, so it blocks the gradient:

$$\frac{\partial L}{\partial x} = g \odot \mathbb{1}[x > 0]$$

### 3. Softmax + cross-entropy, fused

Keeping these as one block is both more stable and much simpler. For logits $z \in \mathbb{R}^{N \times C}$
and true labels $y$:

$$L = \frac{1}{N}\sum_{n} \Big( \log \textstyle\sum_c e^{z_{nc}} - z_{n y_n} \Big),
\qquad
\frac{\partial L}{\partial z} = \frac{1}{N}\big( \mathrm{softmax}(z) - \text{onehot}(y) \big)$$

That gradient is just **"predicted probability minus true probability"**. If the model assigns 0.7 to
the correct class, the gradient on that logit is $-0.3$: push it up. Every messy term of the softmax
Jacobian cancels against the $\log$ in the cross-entropy — which is exactly why the two are always
fused in practice.

**Numerical stability.** $e^{z}$ overflows to `inf` for logits above about 88 in float32, which turns
the loss into `nan`. Subtracting the row-wise maximum before exponentiating leaves the softmax
unchanged (it cancels top and bottom) but keeps every exponent $\le 0$.


<!-- ===== cell 7 (TRIMMED) ===== -->

## (a) Linear, ReLU and softmax cross-entropy with explicit forward/backward

Each layer stores whatever its backward pass will need during `forward`. Parameters and their
gradients are exposed through `params()` / `grads()` so the optimizer and the gradient checker can
find them generically.


<!-- ===== cell 10 (TRIMMED) ===== -->

## (b) Verifying the gradients

Two checks, which fail for different reasons — so passing both is strong evidence.

**Check 1 — against autograd.** The same network is rebuilt with `nn.Linear`, my weights are copied
in, and PyTorch's gradients are compared with mine using `torch.allclose`. One detail matters:
`nn.Linear` stores its weight as `(out_features, in_features)` while mine is `(in, out)`, so the
copy and the comparison both need a **transpose**.

**Check 2 — centred finite differences.** This trusts nothing at all. For a single parameter
$\theta$:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$

The *centred* version is used rather than the one-sided one because its error is
$\mathcal{O}(\varepsilon^2)$ instead of $\mathcal{O}(\varepsilon)$. The relative error reported is

$$\text{rel} = \frac{|\text{analytic} - \text{numeric}|}{|\text{analytic}| + |\text{numeric}| + 10^{-12}}$$

**This check is run in `float64`.** In float32 the subtraction $L(\theta+\varepsilon)-L(\theta-\varepsilon)$
cancels most of the significant digits and the relative error floors out around $10^{-3}$, which looks
like a bug but is only rounding. In float64 a correct implementation reaches about $10^{-10}$.


<!-- ===== cell 13 (REMOVED) ===== -->

### Interpretation — Task 1(b)

Both checks pass. Every layer's `dW` and `db` agree with autograd to within
$3.3\times10^{-16}$ in absolute terms — essentially one unit in the last place of a float64 — and
`torch.allclose` returns `True` for all six tensors, so the analytic backward pass computes exactly
the same function PyTorch's does.

The finite-difference check is the stronger evidence, because it never touches autograd at all: the
worst relative error over the 72 sampled parameters is $3.7\times10^{-9}$, with medians around
$10^{-10}$ — the floor set by float64 round-off. That is what a correct gradient looks like; an
error of $10^{-3}$ or worse would indicate a real bug.

Two practical points worth recording:

- **Precision matters more than the formula.** Running the identical check in float32 gives relative
  errors around $10^{-3}$, because $L(\theta+\varepsilon)$ and $L(\theta-\varepsilon)$ agree in their
  first several digits and subtracting them destroys the rest. The gradient is not wrong; the
  measuring instrument is. Hence float64 and $\varepsilon = 10^{-5}$.
- **ReLU's kink can produce a false failure.** ReLU is not differentiable at 0. If a pre-activation
  happens to sit within $\varepsilon$ of zero, the $+\varepsilon$ and $-\varepsilon$ evaluations land on
  opposite sides of the kink, the two-sided slope measures something the true derivative never
  claimed, and that one parameter shows a large error even though the code is correct. With random
  weights and 12 sampled parameters per tensor this did not occur here, but it is why a single large
  relative error in a gradient check should be investigated rather than assumed to be a bug.


<!-- ===== cell 14 (TRIMMED) ===== -->

## (c) Training the MLP on flattened CIFAR-10

Each 32x32x3 image is flattened to a **3072-dimensional vector**, and the network is
3072 -> 512 -> 256 -> 10 trained with the hand-written SGD above. No autograd is used in this cell
either — `model.backward(...)` is my own code.


<!-- ===== cell 17 (REMOVED) ===== -->

### Interpretation — Task 1(c)

The first epoch averages a training loss of **1.70** — it begins near $\ln 10 = 2.303$ at
initialization, which is exactly what a network that knows nothing must output, since it can do no
better than assign $1/10$ to every class — and falls to **0.157** by epoch 30. Validation accuracy
peaks at **50.6%** and ends at **48.4%**.

**The network overfits heavily.** Training accuracy ends at **94.8%** against 48.4% on validation,
a gap of **46 points**. Validation accuracy stops improving after roughly epoch 5 and then drifts
sideways while the training loss continues to fall by another order of magnitude. That divergence is
the signature of memorisation rather than learning: with 1.7 M parameters against 45k images and no
regularization of any kind — no weight decay, no dropout, no augmentation — the network has more
than enough capacity to fit the training split almost exactly, and nothing stops it from doing so.
Task 3 is devoted to exactly this problem.

**The number to pay attention to is the ceiling.** A fully-connected network tops out at roughly
50-55% on CIFAR-10 no matter how long it trains or how wide it is made, and that is a structural
limit rather than an optimization failure:

- **Flattening destroys locality.** After `reshape`, the pixel at (5, 5) and the pixel at (5, 6) are
  just input 165 and input 166 — the network is never told they are neighbours, and would have to
  learn that fact separately for every pair of pixels.
- **There is no translation invariance.** A cat in the top-left and the same cat shifted three pixels
  right are two completely different input vectors, so the MLP must learn a separate set of weights
  for every position an object can appear in.
- **It is extremely expensive in parameters.** The first layer alone is 3072x512 = 1.57 M weights,
  and it still only ever sees one fixed arrangement of the image.

This is precisely the gap convolution closes: weight sharing gives translation invariance for free,
and a local kernel builds in the assumption that nearby pixels belong together. The small CNN in
Task 2 reaches a much higher accuracy with roughly an order of magnitude fewer parameters, which is
the comparison that motivates the rest of this assignment.


<!-- ===== cell 18 (TRIMMED) ===== -->

## (d) Gradient norms at initialization: sigmoid vs ReLU, 10 layers

A 10-layer MLP is built, **one** forward and backward pass is run at initialization (no training at
all), and $\lVert \partial L/\partial W^{(l)} \rVert_2$ is recorded for every layer.

Each activation is paired with the initialization it was designed for — **Xavier**
($\mathrm{Var}[W]=1/\text{fan-in}$) for sigmoid, **Kaiming** ($\mathrm{Var}[W]=2/\text{fan-in}$) for
ReLU. Pairing them this way is the honest comparison: the activation and the initializer are a
matched pair, and running sigmoid with Kaiming init would blame the activation for the wrong
initializer.

The y-axis is **logarithmic**, otherwise the sigmoid curve is an indistinguishable flat line at zero.


<!-- ===== cell 21 (REMOVED) ===== -->

### Interpretation — Task 1(d)

*(Fill the blanks from the table above.)*

**Sigmoid: the gradient vanishes.** Layer 10 has a gradient norm of $8.9\times10^{-1}$, but by layer
1 it has fallen to $9.4\times10^{-6}$ — a drop of **nearly five orders of magnitude**, a ratio of
$9.4\times10^{4}$. On the log axis that fall is close to a straight line, which is the telling
detail: a straight line on a log scale means each layer multiplies the gradient by roughly the same
constant factor, exactly as a product of per-layer derivatives predicts. The reason is the
derivative itself:

$$\sigma'(x) = \sigma(x)\,(1 - \sigma(x)) \le 0.25$$

with equality only at $x = 0$. Backpropagation multiplies by one such factor per layer, so after 9
hidden layers the signal reaching the first layer is scaled by at most
$0.25^9 \approx 4\times10^{-6}$, and in practice much less, because most units sit away from $x = 0$
where $\sigma'$ is far below 0.25. The early layers therefore receive essentially no learning signal:
they stay near their random initialization while only the last few layers train. This is the classic
**vanishing gradient problem**.

**ReLU: the gradient stays flat.** The norms stay between 2.3 and 6.5 across all ten layers, and the
ratio of layer 10 to layer 1 is **0.74** — no decay whatsoever, against sigmoid's factor of
$9.4\times10^{4}$. (Layer 1 is in fact the largest of the ten, but that is a size effect rather than
a gradient effect: its weight matrix is $3072\times128$ while every other hidden layer is
$128\times128$, so it has 24 times as many entries contributing to the Frobenius norm.)

Two effects cancel to keep the scale constant:

- ReLU's derivative is **0 or 1** — it never shrinks what passes through. But roughly half the units
  are inactive at initialization, so about half the signal is killed, costing a factor of about 2 in
  variance per layer.
- **Kaiming initialization puts that factor back deliberately.** Setting
  $\mathrm{Var}[W] = 2/\text{fan-in}$ instead of $1/\text{fan-in}$ is precisely the correction for
  "half the units are off". The 2 in the numerator *is* the ReLU correction.

So the activation and the initializer are not independent choices: Kaiming init is derived *for*
ReLU, and pairing Xavier with ReLU reintroduces a slow decay.

**Why this matters for the rest of the assignment.** This single plot explains why deep networks were
impractical before roughly 2010, and why ReLU together with principled initialization was what made
depth usable. It is also only a partial fix: ReLU keeps gradients healthy *at initialization*, but at
56 layers (Task 3b) degradation returns during training, and that is the problem residual connections
exist to solve — a shortcut gives the gradient a path with derivative exactly 1 all the way back to
the input.


<!-- ===== cell 22 (REMOVED) ===== -->

---
## Task 1 — summary

| Part | What was done | Result |
|---|---|---|
| (a) | `Linear`, `ReLU`, `Sigmoid` and a fused `SoftmaxCrossEntropy`, each with explicit forward/backward; `MLP` and `SGDManual` built on top | no autograd used anywhere |
| (b) | Compared against autograd (`torch.allclose`) and against centred finite differences in float64 | autograd agrees to $3\times10^{-16}$; worst finite-difference relative error $3.7\times10^{-9}$ |
| (c) | 3072-512-256-10 trained with hand-written SGD on flattened CIFAR-10 | best validation accuracy **50.6%**, train/val gap 46 points |
| (d) | Per-layer gradient norms at init, 10 layers, sigmoid vs ReLU | sigmoid decays $9.4\times10^{4}$ across 10 layers; ReLU flat (ratio 0.74) |

**Next:** Task 2 moves to `torch.optim.Optimizer` subclasses (SGD+momentum and Adam) and a small CNN.


<!-- ===== cell 23 (TRIMMED) ===== -->

---
# Task 2: Optimizers

Task 1 used plain gradient descent: take the gradient, step against it. This task asks what happens
when that step is made smarter, by implementing **SGD with momentum** and **Adam** as proper
`torch.optim.Optimizer` subclasses and comparing all three.

## The three update rules

Write $g_t$ for the gradient at step $t$ and $\theta$ for a parameter.

**Plain SGD** — step straight downhill:

$$\theta_{t+1} = \theta_t - \eta\, g_t$$

**SGD with momentum** — accumulate a running "velocity" instead of using the raw gradient:

$$b_t = \mu\, b_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, b_t$$

The useful way to see this: if the gradient keeps pointing the same way, $b$ grows like a geometric
series $1 + \mu + \mu^2 + \cdots = \frac{1}{1-\mu}$. With the usual $\mu = 0.9$ that is a **tenfold
amplification** of the effective step along consistent directions, while directions that keep
flipping sign cancel out. Momentum therefore accelerates steady descent and damps oscillation — and
it is why a momentum run can use a learning rate roughly 10x smaller than plain SGD and
still come out ahead, which part (c) bears out.

**Adam** — keep a running mean *and* a running variance of the gradient, then divide:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad
  \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
  \theta_{t+1} = \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

Dividing by $\sqrt{\hat v}$ makes the step **scale-free**: each parameter is normalized by its own
recent gradient magnitude, so the step size is roughly $\eta$ regardless of whether that parameter's
gradients are large or small. That is what makes Adam insensitive to bad scaling.

**Why bias correction is needed.** $m$ and $v$ both start at zero, so early on they are biased toward
zero — at $t=1$, $m_1 = (1-\beta_1) g_1 = 0.1 g_1$, ten times too small. Dividing by $1 - \beta_1^t$
undoes exactly this: at $t=1$ it divides by $0.1$, restoring $m_1 = g_1$, and as $t$ grows the
correction fades to 1. Without it the first few hundred steps are far too small, because
$1 - \beta_2^t$ with $\beta_2 = 0.999$ takes roughly a thousand steps to approach 1.


<!-- ===== cell 24 (TRIMMED) ===== -->

## (a) Implementing both as `torch.optim.Optimizer` subclasses

Subclassing means two obligations: pass the defaults dict to `super().__init__`, and loop over
`self.param_groups` (not a flat parameter list) inside `step()`, so that per-group learning rates and
schedulers work. Per-parameter state — the momentum buffer, Adam's $m$, $v$ and step count — lives in
`self.state[p]`, which PyTorch creates lazily and includes in `state_dict()`.

Two details that have to match PyTorch exactly:

- **The momentum buffer is initialized to the gradient itself**, not to zero. Initializing to zero
  would make the first step $\eta(1-\mu)g$ instead of $\eta g$.
- **Weight decay is folded into the gradient** as $g \leftarrow g + \lambda\theta$ before anything
  else, which is the classic L2 form that `torch.optim.SGD` and `torch.optim.Adam` use. (This is
  *not* the same as AdamW's decoupled decay.)


<!-- ===== cell 26 (REMOVED) ===== -->

### A note on the two conventions for the momentum buffer

Textbooks often write the momentum buffer as an **exponential moving average**, with a $(1-\mu)$ in
front of the gradient:

$$b_t = \mu\, b_{t-1} + (1-\mu)\, g_t \qquad \text{(EMA form)}$$

but the implementation above uses

$$b_t = \mu\, b_{t-1} + g_t \qquad \text{(PyTorch form)}$$

**Both are correct — they are the same algorithm with the learning rate rescaled.** The two buffers
differ by exactly the constant factor $(1-\mu)$, since $b^{\text{EMA}}_t = (1-\mu)\, b^{\text{PyTorch}}_t$,
and that constant can be absorbed into $\eta$. Running the EMA form at $\eta/(1-\mu)$ gives
numerically identical parameters to the PyTorch form at $\eta$ — verified below to $8\times10^{-17}$.

The difference is only in what $\eta$ *means*:

| | steady-state buffer with constant gradient $g$ | step size |
|---|---|---|
| EMA form | $g$ | $\eta$ |
| PyTorch form | $\dfrac{g}{1-\mu} = 10g$ | $\dfrac{\eta}{1-\mu} = 10\eta$ |

This is the source of the "tenfold amplification" described at the top of this task: it is a property
of the PyTorch convention, not of momentum in the abstract. In the EMA convention the buffer is a
true weighted average whose weights sum to 1, so there is no amplification at all — and the tenfold
factor reappears in the learning rate instead.

**Why the PyTorch form is used here.** Part (a) requires matching `torch.optim.SGD`, whose default is
`dampening=0`, giving $b_t = \mu b_{t-1} + g_t$. PyTorch's general rule is
$b_t = \mu b_{t-1} + (1-\tau) g_t$ with dampening $\tau$, so setting $\tau = \mu$ gets close to the
EMA form — but not exactly, because PyTorch special-cases the first step as $b_1 = g_1$ regardless of
dampening. A hand-written pure EMA starting from $b_0 = 0$ matches exactly; `dampening=0.9` does not.

**Note that `MyAdam` above *does* use the $(1-\beta)$ form**, for $m$ and $v$ both. That is not an
inconsistency: Adam is *defined* with the EMA convention, and its bias correction — dividing by
$1-\beta^t$ — only makes sense for a buffer whose weights are meant to sum to 1. Each algorithm is
implemented in its own standard convention.


<!-- ===== cell 28 (TRIMMED) ===== -->

### Showing they match `torch.optim`

The test trains **two identical models** — same seed, same initial weights, same data in the same
order — one with my optimizer and one with PyTorch's, for 50 steps, then compares every parameter.
Running in `float64` makes the comparison tight enough to catch a genuine formula error rather than
round-off. Six configurations are checked, including weight decay and non-default betas, since those
are the code paths most likely to differ.


<!-- ===== cell 30 (REMOVED) ===== -->

### Interpretation — Task 2(a)

All six configurations match. The two `MySGD` variants are **bit-for-bit identical** to
`torch.optim.SGD` (`max|diff| = 0.000e+00` after 50 steps), because the update is a single
fused multiply-add performed in the same order as PyTorch's.

`MyAdam` agrees with `torch.optim.Adam` to about $5\times10^{-17}$, which is float64 round-off —
roughly one unit in the last place, accumulated over 50 steps. It is not bit-identical for a benign
reason: PyTorch computes the step as
$\frac{\eta}{1-\beta_1^t}\cdot\frac{m}{\sqrt{v}/\sqrt{1-\beta_2^t}+\epsilon}$ while the
implementation above computes $\eta\cdot\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon}$.
These are algebraically the same expression — $\sqrt{v/c} = \sqrt{v}/\sqrt{c}$ — but the operations
happen in a different order, and floating-point arithmetic is not associative. A difference at the
17th digit confirms the formula is right; a difference at the 3rd would mean a real bug.

The non-default cases matter most here. Matching with `betas=(0.8, 0.99)` shows the bias correction
is genuinely a function of $t$ and both betas rather than a hard-coded constant, and matching with
weight decay confirms the L2 term is folded into the gradient *before* the moments are updated — had
it been applied afterwards, as AdamW does, these numbers would disagree visibly.


<!-- ===== cell 31 (TRIMMED) ===== -->

## (b) Trajectories on $f(x,y) = \tfrac{1}{2}(x^2 + 50y^2)$

This function is the standard test for an **ill-conditioned** problem. Its gradient is
$\nabla f = (x,\ 50y)$, so the surface is 50 times steeper in $y$ than in $x$: the contours are long
thin ellipses forming a narrow valley along the $x$-axis, and the minimum is at the origin.

The difficulty is that one learning rate must serve both directions. Along $y$, gradient descent
multiplies the error by $(1 - 50\eta)$ each step, which only converges while

$$|1 - 50\eta| < 1 \quad\Longleftrightarrow\quad \eta < \frac{2}{50} = 0.04$$

Along $x$ it multiplies by $(1 - \eta)$, so progress there is slow for any $\eta$ that small. All
three optimizers start from $(-9, 1)$ and run 60 steps. SGD and momentum use $\eta = 0.038$, just
below the stability limit — the fastest plain SGD can safely go. Adam uses $\eta = 0.3$, because its
normalized step is roughly $\eta$ per iteration regardless of gradient magnitude, so it needs a
learning rate comparable to the *distance to travel* rather than to the curvature.


<!-- ===== cell 34 (REMOVED) ===== -->

### Interpretation — Task 2(b)

The three paths behave in three clearly different ways, and the "sign flips in $y$" column counts the
zigzag directly.

**Plain SGD zigzags on every single step — 60 flips out of 60.** With $\eta = 0.038$ the $y$-error is
multiplied by $1 - 50(0.038) = -0.9$ each step: the sign flips every time and the magnitude only
shrinks by 10%, so the path bounces across the valley while creeping along it. Meanwhile $x$ is
multiplied by $(1 - 0.038) = 0.962$, so after 60 steps it has only moved from $-9$ to $-0.88$. Nearly
all the effort goes into the direction that does not matter. Raising $\eta$ to fix the slow $x$
progress is not an option — above $0.04$ the $y$ direction diverges.

**Momentum halves the oscillation and travels five times further — 30 flips, final $f$ about 6.7x
lower.** The $y$ component alternates sign, so consecutive contributions to the velocity buffer
largely cancel; the $x$ component is consistent, so it accumulates toward the $\frac{1}{1-\mu} = 10$
amplification. The result is exactly what momentum is for: oscillation is damped and steady descent
is accelerated, *at the same learning rate*. The path visibly bends into the valley and runs along
it rather than across it.

**Adam barely oscillates at all — 6 flips — and takes a nearly diagonal route.** Dividing by
$\sqrt{\hat v}$ normalizes each coordinate by its own gradient history, so the enormous $y$ gradients
and the small $x$ gradients produce steps of comparable size. Adam effectively ignores the
conditioning that cripples SGD, which is why it needed $\eta = 0.3$ — almost ten times larger — to
cover the same ground. Its curve in the lower panel drops fastest early but then flattens, since
near the minimum the gradient becomes small and noisy and the $\sqrt{\hat v}$ normalization keeps the
step size from shrinking proportionally.

**The general lesson.** The problem here is the condition number ($50$), not the optimizer. Momentum
addresses it by remembering past directions; Adam by rescaling each coordinate. Both beat plain SGD,
but note that the *best final value* belongs to momentum, not Adam — on a clean, noiseless, convex
problem a well-tuned momentum run is hard to improve on. Adam's real advantage shows up in part (c),
where gradients are stochastic and poorly scaled across layers.


<!-- ===== cell 35 (TRIMMED) ===== -->

## (c) A small CNN with each optimizer at three learning rates

The network is a compact three-stage CNN with batch norm — 141k parameters — trained for 15 epochs on
the full 45k training split. The same nine-cell grid is used for every optimizer:

$$\eta \in \{0.001,\ 0.01,\ 0.1\}$$

Deliberately **the same grid for all three**, rather than a hand-tuned range per optimizer. That is
what makes the sensitivity comparison meaningful: if each optimizer were given its own favourable
range, the plot would show only that all three can be made to work. With a shared grid, *where* each
curve peaks and *how sharply* it falls away are both directly readable.


<!-- ===== cell 39 (REMOVED) ===== -->

### Interpretation — Task 2(c)

| optimizer | lr = 0.001 | lr = 0.01 | lr = 0.1 | best | spread across grid |
|---|---|---|---|---|---|
| SGD | 42.76% | 62.54% | **71.58%** | 0.1 | **28.8 points** |
| SGD + momentum | 64.22% | 75.28% | **79.14%** | 0.1 | 14.9 points |
| Adam | 76.78% | **80.40%** | 69.14% | 0.01 | 11.3 points |

**Sensitivity is the headline, and it differs sharply between the three.**

Plain SGD swings **28.8 points** across the grid — it is nearly useless at $\eta = 0.001$ (42.8%,
barely better than the Task 1 MLP) and only becomes competitive at $\eta = 0.1$. Momentum halves that
swing to 14.9 points. Adam varies by only **3.6 points** between $\eta = 0.001$ and $\eta = 0.01$,
both of which land above 76%. In other words, a careless learning-rate choice costs almost nothing
with Adam and costs nearly thirty points with plain SGD. This is the practical reason Adam is the
default when there is no budget for tuning.

**But Adam is the only one that actually fails at the top of the grid.** It collapses from 80.4% to
69.1% at $\eta = 0.1$ — an 11-point drop — while both SGD variants are still improving there. Adam's
normalized step is roughly $\eta$ per parameter per iteration regardless of gradient magnitude, so
$\eta = 0.1$ means steps of order 0.1 on weights that are themselves order 0.1: the updates are as
large as the parameters, and training destabilizes. The flatness that makes Adam forgiving at small
learning rates becomes a liability at large ones, because nothing in the update shrinks as the
optimum approaches.

**An honest limitation of this grid.** Both SGD curves are *still rising* at $\eta = 0.1$, so their
optima lie at or beyond the right edge and this grid does not bracket them — plain SGD in particular
would likely keep improving to $\eta \approx 0.3$. The three points per optimizer the assignment
asks for are enough to rank sensitivity, but not enough to locate SGD's true best rate, and the
71.58% reported for it is therefore a lower bound rather than its ceiling.

**The $\frac{1}{1-\mu} = 10$ amplification does show up, read the right way.** It is not visible as a
shift in peak position, because the grid truncates both SGD curves at the same point. It is visible
in the comparison at matched quality: **momentum at $\eta = 0.01$ reaches 75.28%, beating plain SGD
at $\eta = 0.1$ (71.58%)** — momentum does better with one tenth of the learning rate, which is
exactly the tenfold effective-step amplification predicted at the top of this task. Momentum buys
roughly a decade of learning rate.

**The training-loss panel confirms the ranking.** At each optimizer's best rate the final training
losses are 0.40 (SGD), 0.17 (momentum) and 0.15 (Adam): the two optimizers that converge fastest in
validation accuracy also drive the training loss lowest, so the differences here are genuine
optimization speed, not a regularization side-effect.

**And the payoff promised in Task 1.** This CNN reaches **80.4%** validation accuracy with **141,034
parameters**. The Task 1 MLP managed 50.6% with **1,707,274** — twelve times as many parameters for
thirty points less accuracy. That is the entire case for convolution: weight sharing and locality,
not scale.


<!-- ===== cell 40 (REMOVED) ===== -->

---
## Task 2 — summary

| Part | What was done | Result |
|---|---|---|
| (a) | `MySGD` (momentum, weight decay) and `MyAdam` (bias correction) as `torch.optim.Optimizer` subclasses | all six configurations match `torch.optim`; SGD bit-exact, Adam to $5\times10^{-17}$ |
| (b) | Trajectories on $\frac{1}{2}(x^2+50y^2)$ with contours | SGD 60 sign flips, momentum 30, Adam 6 |
| (c) | SmallCNN (141k params), 3 optimizers x 3 learning rates, 15 epochs | see table above |

**Next:** Task 3 reuses `SmallCNN` for the regularization study, then builds plain and residual
networks at depths 8, 20 and 56.


<!-- ===== cell 41 (TRIMMED) ===== -->

---
# Task 3: Regularization and residual connections

Task 1 ended with a network that reached 94.8% on its training data and 48.4% on validation — it had
memorised rather than learned. This task attacks that problem from two directions: **part (a)** asks
what different regularizers do to the train/validation gap, and **part (b)** asks what happens when
networks get very deep, and why a shortcut connection fixes it.

## (a) Six regularizers on a deliberately small training set

The training set is cut to **5,000 images** — one ninth of the full split. This is not an arbitrary
choice: regularization only matters when there is not enough data, and with all 45k images a 141k
parameter CNN barely overfits at all. Shrinking the data makes the effect large enough to measure.

The seven configurations:

| config | what it does |
|---|---|
| **baseline** | nothing — the control |
| **weight decay** | adds $\lambda\lVert w\rVert^2$ to the loss, pulling weights toward zero |
| **dropout** | randomly zeroes 30% of the features before the classifier at train time |
| **flip+crop** | random horizontal flip, and a random 32x32 crop from a 4-pixel-padded image |
| **cutout** | blanks one random 8x8 square in each image |
| **mixup** | trains on *blends* of two images, with correspondingly blended labels |
| **all combined** | all five at once |

**How the last two work, since they are implemented from scratch here.**

*Cutout* deletes a random square patch. The idea is that if the network can lean on one distinctive
feature — a cat's ear, say — it will, and then it fails whenever that feature is hidden. Removing a
random part of every image forces it to spread its evidence over the whole picture. The patch is
zeroed *after* normalization, so the hole takes the per-channel mean colour rather than black.

*Mixup* goes further and trains on images that do not exist. Two images are blended,
$\tilde x = \lambda x_i + (1-\lambda)x_j$, and the loss uses both labels in the same proportion:

$$L = \lambda\, \text{CE}(f(\tilde x), y_i) + (1-\lambda)\, \text{CE}(f(\tilde x), y_j)$$

with $\lambda$ drawn from a $\text{Beta}(\alpha, \alpha)$ distribution. This forces the network to
behave *linearly between training examples* rather than making wildly overconfident predictions in
the gaps between them, which is where the memorisation lives.

**One measurement decision worth stating.** Train accuracy is always evaluated on **clean,
un-augmented training images**, in a separate pass. Measuring it on augmented batches would make the
numbers incomparable — mixup's "training accuracy" against blended labels is not the same quantity
as the baseline's — and the train/validation gap is the whole point of the table.


<!-- ===== cell 45 (REMOVED) ===== -->

### Interpretation — Task 3(a)

| config | train acc | best val acc | gap | reading |
|---|---|---|---|---|
| baseline | **100.00%** | 67.36% | 32.6 | perfect memorisation — the control |
| weight decay | **100.00%** | 68.42% | 31.9 | +1.1 pts, still memorises completely |
| dropout | **100.00%** | 67.80% | 32.6 | +0.4 pts, essentially no effect |
| **flip+crop** | 83.10% | **70.88%** | 17.4 | **best single method, +3.5 pts** |
| cutout | 98.06% | 67.94% | 34.1 | +0.6 pts |
| mixup | 99.46% | 65.46% | 36.5 | **−1.9 pts — actively hurt** |
| all combined | 72.56% | 68.82% | **7.5** | smallest gap by far, +1.5 pts |

**The baseline memorises perfectly.** 100.00% training accuracy on all 5,000 images against 67.36%
validation — a 32.6-point gap. That is what 141k parameters do to 5k images with nothing holding
them back.

**The clearest finding is in the training-accuracy column.** Baseline, weight decay and dropout
**all reach exactly 100.00%**. They memorise the training set completely *despite* the
regularization. Only the methods that **change the data** stop that: flip+crop holds training
accuracy to 83.10%, and the full combination to 72.56%. The mechanism matters more than the label —
constraining the weights barely helped here, while giving the network a different image every time
did.

**Why dropout did almost nothing.** In `SmallCNN`, dropout sits only *after* global average pooling,
on a 128-dimensional vector just before the classifier. That is a far weaker intervention than
dropout between convolutional layers, where it would disrupt spatial features. The +0.4 points is a
property of *where it was placed*, not evidence that dropout is ineffective in general. Weight decay
is similarly applied to every parameter including the batch-norm scale and shift, which common
practice excludes.

**Two results that look like failures but are findings:**

- **Mixup hurt, by 1.9 points.** With $\alpha = 1$ on only 5,000 images almost every training image
  is a blend, so the network gets very little clean signal. Mixup is a strong regularizer that needs
  either more data or a longer schedule to pay for itself. Note its gap is the *largest* in the
  table (36.5) — which is misleading, because its training accuracy is measured on clean images the
  network has barely seen undisturbed.
- **"All combined" is not the best.** It achieves by far the smallest gap (7.5 points) but only
  68.82% validation — below flip+crop alone at 70.88%. Training accuracy falls to 72.56%, so the
  network is now **underfitting**: five regularizers on 5,000 images over-constrain it. This is the
  most instructive row in the table, because it shows the gap and the validation accuracy are
  different objectives. **Closing the gap by crippling training is not the same as generalising
  better.**

**Separating the two mechanisms.** Weight decay and dropout constrain the *model* and left
memorisation intact. Flip+crop, cutout and mixup change the *data* and did reduce training accuracy.
On this dataset the data-side interventions did essentially all the useful work, and the single best
one — horizontal flip plus random crop — is also the cheapest, because a mirrored or slightly
shifted CIFAR image is genuinely still the same object. It adds real information rather than just
noise.

**Tie back to Task 1.** The MLP there showed a 46-point gap with no regularization. Here the
baseline gap is 32.6 points and the best configuration brings it to 7.5 — but the best *validation
accuracy*, 70.88%, comes from a configuration with a 17.4-point gap. The gap is a diagnostic, not
the goal.


<!-- ===== cell 46 (TRIMMED) ===== -->

## (b) Plain versus residual networks at depths 8, 20 and 56

These are the CIFAR-style networks from the ResNet paper: one 3x3 stem convolution, then **three
stages** of $n$ blocks each at **16, 32 and 64 channels**, then global average pooling and a linear
head. Each block is two 3x3 convolutions, so the total depth is $6n + 2$ — giving **8, 20 and 56**
for $n = 1, 3, 9$. Batch norm follows every convolution, and all convolutions use Kaiming
initialization.

**The only difference between the two families is one line**, in `CifarBlock.forward`:

```python
if self.residual:
    out = out + self.shortcut(x)
```

To keep that literally true, the shortcut is **Option A** from the ResNet paper: where the block
changes shape, the input is subsampled with a stride and its channels are zero-padded, rather than
projected with a 1x1 convolution. Option A has **no parameters at all**, so a plain network and a
residual network of the same depth have *exactly* the same parameter count — as the table below
confirms. Any difference in training loss is therefore attributable to the shortcut itself and not
to extra capacity.

**What to expect, and why.** The intuition for residual connections is that a block learns a
*correction* $F(x)$ to its input rather than a whole new representation: the block computes
$x + F(x)$, so if the best thing it can do is nothing, it only needs to drive $F$ to zero, which is
easy. In a plain network every block must learn to reproduce its input before it can improve on it,
and that becomes hard when there are 56 layers of them. The shortcut also gives the gradient a path
with derivative exactly 1 back to earlier layers — the clean route that Task 1(d) showed sigmoid
lacked.

**No data augmentation is used in part (b)** — unlike part (a). That is deliberate. The question
here is whether a deeper network can be *optimized*, so the networks are allowed to fit the training
set as hard as they can, and the training-loss curves are compared directly. Augmentation would damp
exactly the effect being measured.

The key observation is about **training** loss, not validation: if the deeper plain network trains
*worse* than the shallower one despite having strictly more capacity, that is not overfitting. It is
an optimization failure, and it is what the ResNet paper named the **degradation problem**.


<!-- ===== cell 50 (REMOVED) ===== -->

### Interpretation — Task 3(b)

| depth | plain final train loss | residual final train loss | plain val | residual val |
|---|---|---|---|---|
| 8 | 0.0617 | 0.0641 | 79.40% | 79.10% |
| 20 | 0.0037 | 0.0017 | 82.56% | 83.16% |
| 56 | **0.7719** | **0.0009** | 57.72% | 80.78% |

**First, the comparison is fair.** Parameter counts are identical within each depth — 75,290 at
depth 8, 269,722 at depth 20, 853,018 at depth 56 — because the Option A shortcut has no parameters.
Nothing below can be explained by one family having more capacity.

**The plain family breaks at depth 56.** Its training loss goes from 0.0037 at depth 20 to **0.7719**
at depth 56 — roughly **200 times worse** — despite having **three times more parameters**. The
residual family goes the other way: 0.0017 to 0.0009, continuing to improve. At depth 56 the two
differ by a factor of about **860** at identical parameter count.

**This is not overfitting, and that distinction is the whole point.** Overfitting means *low*
training loss with poor validation accuracy. Here the **training** loss itself is bad, on data the
network is actively being fitted to, with no augmentation to hold it back.

And it should be impossible. A 56-layer network can represent anything a 20-layer one can: set the
extra 36 layers to the identity and you recover the shallower network exactly. So plain-56 has
**strictly more capacity** than plain-20 and *must* be able to reach at least as low a training
loss. It does not. The capacity is there and **gradient descent cannot find it** — an
**optimization** failure, which the ResNet paper named the **degradation problem**.

**Why the shortcut fixes it.** A residual block computes $x + F(x)$, so representing the identity
only requires driving $F$ to zero — which is the easiest thing for a network with weight decay to
do. A plain block must *learn* to reproduce its own input before it can improve on it, and 56 layers
of that is hard. The shortcut also gives the gradient a path with derivative exactly 1 back to
earlier layers: the clean route Task 1(d) showed sigmoid lacked. ReLU and Kaiming initialization
fixed the signal *at initialization*; at 56 layers it degrades again *during training*, and the
shortcut is what holds the path open.

**The depth-8 tie is a feature of the result, not noise.** Plain-8 (0.0617) even edges ahead of
residual-8 (0.0641) by a hair. At 8 layers there is no degradation to fix, so the shortcut buys
nothing and adds a slight constraint. This confirms the benefit is specifically about **depth** —
it is not a free improvement available at any size.

**One honest qualification.** Residual-56 reached the lowest training loss of all six networks
(0.0009) but *not* the best validation accuracy — residual-20 wins there, 83.16% against 80.78%.
With 30 epochs and no augmentation the deepest network fits the training set hardest and generalises
slightly worse. That is a data and schedule limitation, not a contradiction: the ResNet paper trained
for 180 epochs with augmentation. The claim this experiment supports is specifically about
**trainability** — that a 56-layer plain network cannot be optimized while a 56-layer residual one
can — and on that question the evidence is unambiguous.


<!-- ===== cell 51 (REMOVED) ===== -->

---
## Task 3 — summary

| Part | What was done | Result |
|---|---|---|
| (a) | Seven configurations on 5k images: baseline, weight decay, dropout, flip+crop, cutout, mixup, all combined (cutout and mixup implemented from scratch) | see table above |
| (b) | Plain and residual networks at depths 8, 20, 56 with parameter-free Option A shortcuts | identical parameter counts per depth; see training-loss plot |

**Next:** Task 4 (optional) designs the best CNN under 500k parameters and 20 epochs, and evaluates
the test set exactly once.


<!-- ===== cell 52 (TRIMMED) ===== -->

---
# Task 4: Best CNN under 500k parameters

The constraints are **at most 500,000 parameters** and **at most 20 training epochs**, tuned on the
validation split, with the test set evaluated exactly once at the very end.

## Designing to the budget

Twenty epochs is a tight budget, and it changes what the right architecture is. A very deep network
spends most of a short schedule just getting organised; a wider, shallower one converges faster per
epoch. Task 3(b) also showed that at 30 epochs the 56-layer residual network had not caught up with
the 20-layer one. So the design starts from **depth 20** — the shape that actually worked — and
spends the remaining parameter budget on **width** rather than more layers.

Scaling the channels from the standard 16/32/64:

| configuration | depth | parameters | fits 500k |
|---|---|---|---|
| 16 / 32 / 64 (Task 3b baseline) | 20 | 269,722 | yes, but leaves 46% unused |
| **21 / 42 / 84** | **20** | **463,627** | **yes — 93% of budget** |
| 22 / 44 / 88 | 20 | 508,672 | no |
| 24 / 48 / 96 | 14 | 386,626 | yes |
| 20 / 40 / 80 | 26 | 572,430 | no |

**21/42/84 at depth 20** is the choice: it is the widest version of the known-good shape that still
fits, using 93% of the allowance. The odd channel counts are simply what the budget permits.

## The recipe

| ingredient | setting | why |
|---|---|---|
| architecture | residual, depth 20, widths 21/42/84 | Task 3(b): shortcuts win at this depth |
| optimizer | SGD, momentum 0.9 | Task 2(c): best validation accuracy at a well-chosen rate |
| learning rate | 0.1 with cosine annealing to 0 | high rate early, fine steps late — matters in a 20-epoch budget |
| weight decay | 5e-4 | Task 3(a) |
| augmentation | flip + pad-4 random crop + cutout(8) | Task 3(a): the cheapest real gains |
| batch size | 128 | |

Cosine annealing is the one ingredient not inherited from an earlier task. With only 20 epochs there
is no room for a step schedule's long plateaus, and decaying the rate smoothly to zero extracts most
of what a longer schedule would give.

## The ablation

Each row removes **one** ingredient from the full recipe and changes nothing else, so the difference
is attributable to that ingredient alone:

1. **− shortcut** — plain network, identical parameter count (Option A shortcuts are parameter-free)
2. **− augmentation** — no flip, crop or cutout
3. **− weight decay** — $\lambda = 0$
4. **optimizer → Adam** — lr $10^{-3}$, the best Adam rate from Task 2(c)

Five runs of 20 epochs in total.


<!-- ===== cell 57 (REMOVED) ===== -->

### Interpretation — Task 4 ablation

| configuration | best val | change | final train loss | what it isolates |
|---|---|---|---|---|
| full recipe | **89.60%** | — | 0.2281 | |
| minus augment | 85.48% | **−4.12 pts** | **0.0047** | flip + crop + cutout |
| Adam instead | 86.90% | −2.70 pts | 0.3284 | the optimizer |
| minus shortcut | 87.56% | −2.04 pts | 0.3228 | the residual connection |
| minus wt decay | 87.58% | −2.02 pts | 0.2848 | the L2 penalty |

**Augmentation is the single biggest contributor, by a factor of two over anything else.** Removing
it costs 4.12 points, while the next-largest removal costs 2.70. And the training-loss column shows
exactly *why*, in the clearest possible form: without augmentation the training loss falls to
**0.0047**, about **48 times lower** than the full recipe's 0.2281 — while validation accuracy drops
by four points. That is the textbook signature of overfitting: the network fits the training data
far *better* and the held-out data far *worse*. With 45,000 images and 463,627 parameters there is
ample capacity to memorise, and augmentation is what denies it the chance.

**Removing the shortcut acts through a completely different mechanism, and the loss column proves
it.** Training loss goes *up*, from 0.2281 to 0.3228 — the plain network fits the training data
less well, not more. So this is an **optimization** cost, not a generalization one, which is
precisely the distinction Task 3(b) established. Both families have identical parameter counts
(463,627), so capacity is not the explanation. The effect is smaller here than in Task 3(b) because
depth 20 is comfortably trainable without shortcuts; the degradation problem only became severe at
depth 56.

**Shortcut and weight decay are statistically indistinguishable.** They differ by 0.02 points
(−2.04 vs −2.02) — far below the few tenths of a point of run-to-run variation expected from a
single seed. **These two should be reported as tied, not ranked.** Claiming the shortcut matters
marginally more than weight decay would be reading noise.

**SGD with cosine annealing beat Adam by 2.70 points**, reversing the Task 2(c) result where Adam
won on the small CNN at 15 epochs. The reason is the schedule: Adam's advantage is being forgiving
about the learning rate, but once the rate *is* well chosen and annealed smoothly to zero, tuned SGD
with momentum overtakes it. Adam also ends at a higher training loss (0.3284 vs 0.2281), so it is
genuinely optimizing less effectively here, not merely generalizing differently. This is why tuned
SGD remains the standard choice for CIFAR ResNets.

**The resulting ranking of design decisions:**

1. **Augmentation** (−4.12) — by far the most important; fights overfitting
2. **Optimizer choice** (−2.70) — SGD + cosine over Adam
3. **Shortcut and weight decay** (−2.04, −2.02) — tied, and both worth keeping

**Caveats.** This is one seed and a one-factor-at-a-time design, so it cannot see interactions.
Task 3(a) showed those matter: there, every regularizer helped alone while all five together were
*worse* than flip+crop by itself. Differences under roughly half a point here carry no information.


<!-- ===== cell 58 (TRIMMED) ===== -->

## The single test evaluation

Everything above was chosen using the **validation** split. The test set has not been read at any
point in this notebook — it was loaded in the setup cell and left alone.

The cell below evaluates it **once**, on the final model, and caches the number to
`results/t4_test_eval.json`. Re-running the notebook loads that cached value rather than evaluating
again, so the "once" is enforced by the code and not just by intention. Deleting that JSON file is
the only way to repeat the measurement — which would mean the test set had influenced a decision,
and the result would no longer be an honest estimate of generalization.


<!-- ===== cell 60 (REMOVED) ===== -->

### Interpretation — the final model

| | |
|---|---|
| architecture | residual, depth 20, widths 21/42/84 |
| parameters | **463,627** of 500,000 allowed (92.7%) |
| epochs | **20** of 20 allowed |
| validation accuracy | 89.60% |
| **test accuracy** | **90.05%** |

**Both constraints are satisfied** — 92.7% of the parameter budget and exactly the 20 epochs
allowed.

**Test accuracy came out 0.45 points *above* validation** (90.05% vs 89.60%). That is the
comfortable outcome. It means the validation split did its job honestly: it absorbed every design
decision in this task, so it carries some optimistic bias, but with only five configurations
compared that bias is evidently smaller than the noise between the two splits. The test set is also
twice the size of the validation set (10,000 vs 5,000), so it is the less noisy of the two
measurements. Had test come in several points *below* validation, that gap would have been the
price of tuning, and some of the winning configuration's margin would have been luck.

**Why the test set was read only once.** Every time a test number influences a decision, it stops
being a held-out measure and quietly becomes part of the training procedure. Had the ablation been
scored against test rather than validation, the best configuration would have been *selected* using
test data, and 90.05% would be optimistic by an unknown amount. The cell above enforces this
mechanically: it evaluates once and caches to `results/t4_test_eval.json`, so re-running the
notebook reads the stored value instead of measuring again.

**What the number means in context.** 90.05% with 463k parameters in 20 epochs is a reasonable
result for this budget. Published CIFAR-10 ResNets reach 93–95%, but they train for 180+ epochs —
roughly nine times the budget allowed here.

**The binding constraint was epochs, not parameters.** The full recipe ends at a training loss of
0.2281, nowhere near zero, and its validation curve is still rising at epoch 20 — the model is
*underfitted*, not saturated. Given more budget the order I would try things is therefore:

1. **Longer training first.** This costs nothing in architecture and the curve says there is more
   to extract.
2. **Then more width**, since the budget has 36,373 parameters still unused and widening worked
   when going from 269,722 to 463,627.
3. **More depth last**, because Task 3(b) showed depth does not pay without the epochs to support
   it — residual-56 beat residual-20 on training loss but lost on validation accuracy at 30 epochs.


<!-- ===== cell 61 (REMOVED) ===== -->

---
## Overall conclusions

Collecting the thread that runs through all four tasks:

| stage | model | parameters | accuracy |
|---|---|---|---|
| Task 1(c) | MLP on flattened pixels | 1,707,274 | 50.6% val |
| Task 2(c) | SmallCNN, tuned optimizer | 141,034 | 80.4% val |
| Task 3(b) | residual-20, no augmentation | 269,722 | 83.2% val |
| Task 4 | residual net, full recipe | 463,627 | **90.1% test** |

**What each task contributed.** Task 1 built backpropagation from scratch and showed why a
fully-connected network cannot do better than roughly half on images: flattening throws away
locality and translation invariance. Task 2 showed that *how* you descend matters as much as what
you descend — momentum and Adam differ by tens of points at a badly chosen learning rate, and Adam's
chief practical virtue is insensitivity to that choice. Task 3 separated the two reasons a deep
network fails: too little data, fixed by regularization, and too many layers, fixed by shortcuts —
and showed those are genuinely different problems with different cures. Task 4 combined them under a
hard budget.

**The recurring lesson is that structure beats scale.** The best model here has **3.7x fewer
parameters** than the Task 1 MLP and is **39 points** more accurate, because convolution, residual connections
and sensible augmentation each encode something true about images that the MLP had to try to learn
from data and could not.
