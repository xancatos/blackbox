# Paper vs. Code: where this implementation simplifies the originals

This repo is a **teaching** implementation. Every attack reproduces its paper's *core mechanism*,
but the *scale* (query budgets, dimensionality, number of probes) and several *refinements* are
simplified for clarity and CPU speed. The lineage and query-efficiency **trends** are real; the
**absolute numbers are illustrative**, not a benchmark.

This document lists, per attack, what the paper does vs. what [`attacks.py`](./attacks.py) /
[`cifar_lineage.py`](./cifar_lineage.py) actually do. Code is referenced by function name.

---

## 0. Where do the surrogates come from? (an important assumption gap)

Three attacks use a surrogate: **Transfer**, **Biased Boundary**, and **SQBA**. A natural question
is whether the papers tell you how to *obtain* that surrogate. They mostly don't:

| Paper | Does it propose training the surrogate? | This repo |
|---|---|---|
| **Transfer** (Papernot 2016) | **Yes — it is the contribution.** Query the victim to label a synthetic seed set, train a substitute, then **Jacobian-based dataset augmentation**: synthesize points near the substitute's boundary, query the victim for their labels, retrain. Iterate. Costs queries up front. | **Not implemented.** The surrogate is trained independently (`build_surrogate`) on the same data distribution — no victim queries. |
| **Biased Boundary** (Brunner 2019) | **No.** Assumes a *pretrained* surrogate is already available (a public model of a different architecture) and uses its gradients as a "transfer prior." The contribution is biased *sampling*, not surrogate training. | Matches the assumption: surrogate is given (`build_surrogate`, `SurrogateNet`). |
| **SQBA** (Park 2024) | **No.** Assumes a pretrained surrogate/substitute is available; the contribution is the small-query algorithm + query-based fallback. | Matches the assumption: surrogate is given. |

**Consequence for this repo:** `build_surrogate` trains a *generic* model (`SurrogateNet`, a different
architecture — 5×5 then 3×3 convs) on the CIFAR data directly, with **zero victim queries**. That
aligns with Brunner/Park ("surrogate assumed available") but **differs from Papernot**, whose whole
point is to *build* the substitute by querying the victim. So when this repo's transfer attack
reports "8 queries," that is only the ε-sweep transfer-checks — the (paper-mandated) substitute
*training* query cost is not modeled at all.

The white-box "replica" surrogate (`surr_wb`) is a separate idealization: it loads the **victim's own
checkpoint** (identical weights), representing a worst-case "attacker has full parameter access."

---

## 1. White-box

### FGSM — Goodfellow et al., 2014
- **Paper:** single gradient-sign step on the victim, `x' = x + ε·sign(∇ₓL)`.
- **Code (`attack_fgsm` / `pgd_on_surr`, steps=1):** identical formula, but applied to a **surrogate**,
  then transferred. `ε=0.05` is a fixed teaching value.
- **Impact:** faithful; the only twist is white-box-on-surrogate rather than on the victim.

### PGD — Madry et al., 2017
- **Paper:** iterated FGSM with a **random start inside the ε-ball** (and often random restarts),
  step `α ≈ 2.5·ε/steps`, projecting back to the ε-ball each step.
- **Code (`pgd_on_surr`, steps=20):** **no random initialization** — starts at `x = x0.copy()` and
  steps deterministically. Step size is the heuristic `α = ε/4`.
- **Impact:** real difference. Random init/restarts are central to Madry's robustness claim. (Note:
  the slide deck lists "Random Initialization" as a PGD property, but the code omits it.)

---

## 2. Transfer — Papernot et al., 2016
- **Paper:** build the substitute *by querying the victim* (Jacobian-based augmentation), then craft
  white-box examples on it and transfer. Crafting is then ~0-query.
- **Code (`attack_transfer`):** surrogate is pre-trained independently (see §0). The attack crafts
  PGD on the surrogate (no victim queries) and sweeps `ε ∈ linspace(0.01, 0.25, 16)`, spending **one
  victim hard-label query per ε** until the label flips (~8 in the demo).
- **Impact:** mechanism (transferability) is faithful; the **substitute-training query cost is
  absent**, so reported query counts are far lower than the paper's real cost.

---

## 3. Hard-label / decision-based

### Boundary Attack — Brendel, Rauber & Bethge, 2018
- **Paper:** orthogonal (on-sphere) + radial perturbations sampled from a Gaussian; **two separate
  step sizes** (orthogonal and source/radial) each tuned to their **own** acceptance statistics
  (~25% each). Converges over **10⁴–10⁶** queries.
- **Code (`attack_boundary`):** same orthogonal-step-then-radial-pull geometry, but a **single**
  combined acceptance rate over a 30-step window controls **both** step sizes (`sph=0.10`, `src=0.05`,
  ×0.9 if rate<0.2, ×1.1 if rate>0.5). Runs ~400 steps.
- **Impact:** mechanism faithful; coupled step-size control and tiny budget are simplifications.

### OPT — Cheng et al., 2019
- **Paper:** minimize boundary distance `g(θ)`; estimate its gradient with **many** random directions
  (RGF), with careful fine-grained search for `g(θ)`.
- **Code (`attack_opt`, `g_dist`):** same formulation but only `q=6` probe directions, `iters=14`,
  and a **loosened** binary-search tolerance `tol=4e-2` on `g(θ)`.
- **Impact:** faithful mechanism; far fewer probes/precision than the paper's high-dim settings.

### Sign-OPT — Cheng et al., 2020
- **Paper:** replace the per-direction distance measurement with the **sign** of the directional
  derivative (1 query); typically ~100–200 directions per gradient estimate in high dim.
- **Code (`attack_sign_opt`):** exact sign trick (`s = +1 if is_adv(x0 + g_theta·θ̂) else −1`), but
  only `q=12` directions.
- **Impact:** faithful; probe count scaled down ~10–20×.

### HopSkipJump — Chen, Jordan & Wainwright, 2020
- **Paper:** Monte-Carlo normal estimate with a **baseline subtraction** for variance reduction,
  `ĝ = (1/B) Σ (φ_b − φ̄) u_b`; **grows** the probe count `B_t ∝ √t` over iterations; step size carries
  a **1/√d** (dimension) scaling; has both L2 and L∞ variants.
- **Code (`attack_hsj`, `mc_normal`):**
  - normal is plain `(phi[:,None]*U).mean(0)` — **no `−φ̄` baseline term**;
  - **fixed `B=20`** probes (the √t growth exists only in `grad_hb_mc`, used by SQBA-full);
  - step size `best_d/√(t+1)` — **no 1/√d** factor;
  - **L2 only.**
- **Impact:** several real but mechanism-preserving omissions (variance reduction, adaptive probe
  count, dimensional step scaling).

### Triangle Attack — Wang et al., 2022
- **Paper:** operate on the **2-D DCT** of the image; select a **low-frequency square block** as the
  perturbation subspace; law of sines `δₜ₊₁ = δₜ·sin(α+β)/sin(α)`.
- **Code (`attack_triangle`):** law of sines is exact, but the frequency handling is a **1-D DCT on
  the flattened 3072-vector** (`idct(v_dct, norm='ortho')`), "low frequency" = **first 10% of the
  flattened coefficients**, and only **`d=3` random indices** are perturbed per step (`N=2`,
  `iters=40`).
- **Impact:** real difference — the paper's 2-D low-frequency mask is approximated by a 1-D
  flattened-DCT heuristic.

---

## 4. Surrogate-assisted

### Biased Boundary ("Guessing Smart") — Brunner et al., 2019
- **Paper:** biases the **sampling distribution** of perturbations using **three** priors:
  (1) low-frequency **Perlin-noise** bias, (2) the **surrogate-gradient** prior, and (3) regional
  **masking**.
- **Code (`attack_boundary`, `bias=0.5`):** implements **only the surrogate-gradient bias**, as a
  deterministic convex blend `eta = (1-bias)·eta + bias·sgrad(x_b, y0)`. **No Perlin/low-frequency
  bias and no masking.**
- **Impact:** significant simplification — this is why the slide-deck "low-frequency focus" point does
  not correspond to this implementation.

### SQBA — Park, Miller & McLaughlin, 2024
- **Paper (Algorithm 1):** surrogate "multi-gradient" along the path + a **β-switch** that toggles
  between the free surrogate direction and query-based Monte-Carlo estimation.
- **Code:** ships **two** versions on purpose:
  - `attack_sqba` — explicit **teaching** simplification: one surrogate gradient at the boundary,
    fall back to `mc_normal` on failure.
  - `attack_sqba_full` — **paper-faithful Algorithm 1**: `grad_hw_multigrad` (multi-gradient,
    `n=5` path samples) + the β-switch. Constants (β-switch at `used_alpha<0.25` or no improvement,
    improvement threshold `1e-4`) are teaching choices, not the paper's exact values.
- **Impact:** `attack_sqba_full` is faithful in structure; `attack_sqba` is intentionally a reduced
  version for the lineage narrative.

---

## 5. Cross-cutting simplifications (apply to all hard-label attacks)

1. **Query budgets ~100–1000× smaller** than the papers (hundreds of steps vs. 10⁴–10⁶ queries).
   This is the single biggest gap.
2. **2-class task.** Everything runs on a binary slice (airplane vs. automobile, 3072-dim; or 64-dim
   digits) instead of full ImageNet/CIFAR/MNIST. With one "other" class, untargeted ≡ targeted,
   sidestepping a real difficulty the papers handle.
3. **Loosened tolerances** (`binary_search tol=1e-2`, `g_dist tol=4e-2`) trade boundary precision for
   fewer queries.
4. **Surrogate assumed given** (trained independently), not derived from the victim — see §0.
5. **No multi-image / multi-seed averaging** of the kind papers use for stable reported numbers; the
   demo optimizes one image (and small per-budget/sweep averages).

---

## 6. What *is* faithful (so this is balanced)

- The **geometry and update rules** of every attack match the paper (orthogonal+radial walk,
  `g(θ)` minimization, sign-of-derivative trick, MC normal averaging, law of sines, surrogate
  blend, surrogate-gradient + fallback fusion).
- The **lineage ordering** and the **query-efficiency trend** (random → boundary → OPT/Sign-OPT →
  HSJ → Triangle → surrogate-assisted) reproduce the field's actual progression.
- The **SQBA decay behavior** (graceful fall-back to HopSkipJump as the surrogate weakens) is real
  and is what the weaken-surrogate sweep demonstrates.

> One-line summary for a talk: *"Mechanisms faithful, scale and a few variance-reduction/biasing
> refinements simplified; trends real, absolute numbers illustrative."*
