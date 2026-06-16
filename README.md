# Black-box adversarial attack lineage → SQBA

A single, CPU-only Python script ([`lineage.py`](./lineage.py)) that implements
**eleven adversarial attacks** on the same victim model and the same image, so
you can read the history of black-box adversarial attacks as one table — from
the simplest blind baseline up to **SQBA** — *Small-Query Black-Box Attack*, from
"Hard-label based Small Query Black-box Adversarial Attack", Jeonghwan Park, Paul
Miller & Niall McLaughlin (Queen's University Belfast), WACV 2024. The paper
reports SQBA reaching **~5× higher attack success rate** than the benchmarks at
small query budgets (100, 250). The paper is open-access on the
[CVF site](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf).

It's the standalone, from-scratch companion to
[`../SQBA_Attack_Walkthrough.ipynb`](../SQBA_Attack_Walkthrough.ipynb): the
notebook visualises the *real* `sqba.py` against VGG-16/ResNet-18, while this
script rebuilds the *whole family* in ~300 lines of numpy on an 8×8 digits model
you can run anywhere in a few seconds.

## Run it

```bash
uv run lineage.py
```

[`uv`](https://docs.astral.sh/uv/) reads the PEP-723 header at the top of the
script and provisions `numpy` + `scikit-learn` in an ephemeral environment —
nothing is installed globally, no virtualenv to manage. Runtime is a few seconds
on a CPU. (You can also run it with a normal `python lineage.py` if those two
packages are already installed.)

## The core idea

Black-box adversarial-attack research advanced along one axis: **how little the
attacker is allowed to know**. Each tier below relaxes one assumption of the tier
above. The script is organised the same way.

| Tier | Attacker assumes | Attacks (year) |
|---|---|---|
| **white-box** | full gradients (of a surrogate) | FGSM '14, PGD '17 |
| **transfer** | a surrogate + ~0 victim queries | Papernot '16 |
| **hard-label / decision-based** | only the victim's top-1 **label** | Boundary '18, OPT '19, Sign-OPT '20, HopSkipJump '20 |
| **surrogate-assisted** | label **+** a surrogate's gradient | Biased Boundary '19, **SQBA '24** |

**Hard-label** is the hard case: the oracle returns only the predicted class — no
probabilities. The output is a step function (flat, with a jump at the boundary),
so you can't differentiate it. Every hard-label attack is a different way to
extract a usable signal from the single bit *"did the label change?"*.

Two primitives are reused everywhere:

1. **Binary search to the boundary** — bisect the line between an adversarial
   point and the clean image to land exactly on the boundary (the closest
   adversarial point along that line). Turns a discrete label into a real-valued
   distance you can minimise.
2. **The boundary normal** — near the boundary, the fastest label-changing
   direction is its normal. Attacks differ in *how they estimate it*: Boundary
   Attack doesn't (random walk); HopSkipJump estimates it by **querying**
   (costly); SQBA reads it **for free** from a surrogate's gradient.

**SQBA** fuses the two branches: a decision-based boundary search whose normal
comes from a white-box surrogate. That's the "small query" win — the victim
queries HopSkipJump spends *estimating* the normal are replaced by a free
surrogate backward pass. When the surrogate stops helping, SQBA falls back to the
query-based estimate, gracefully decaying into HopSkipJump.

## Example output

```
BLACK-BOX ADVERSARIAL ATTACK LINEAGE  (one image; victim is hard-label only)
attack         yr  victim_q      L2  win   key idea
-- WHITE-BOX --------------------------------------------------------
fgsm          '14         0   2.042    Y   one gradient-sign step on surrogate
pgd           '17         0   2.046    Y   iterated FGSM on surrogate
-- TRANSFER ---------------------------------------------------------
transfer      '16         4   1.375    Y   grow eps on surrogate til victim flips
-- HARD-LABEL -------------------------------------------------------
random         -       1500   2.199    Y   blind noise (scaffold)
line           -          8   1.667    Y   aim at nearest class + binsearch (scaffold)
boundary      '18       500   1.764    Y   random walk along boundary
opt           '19       699   1.623    Y   minimize g(theta), zeroth-order
sign-opt      '20       274   1.528    Y   g(theta) via SIGN of dir. deriv
hopskipjump   '20       401   1.072    Y   MC boundary-normal estimate
-- SURROGATE --------------------------------------------------------
biased-bdry   '19       500   1.001    Y   surrogate-biased boundary walk
sqba          '24       314   0.893    Y   teaching: single surrogate normal + fallback
sqba-full     '24       550   0.920    Y   paper Algo 1: multi-gradient + beta switch
```

There are **two SQBA rows**: `sqba` is the simplified teaching version (one
surrogate gradient per step), and `sqba-full` is the paper-faithful Algorithm 1
(multi-gradient ∇H_w + β switch + sign-gradient init — see below). On this 64-dim
toy the full version costs *more* queries for a similar L2, because its
distinctive multi-gradient method exploits how the surrogate gradient rotates in
*high*-dimensional image space (paper Fig 2) — an effect that barely shows on 8×8
digits. The point of including it is fidelity to the algorithm, not a win here.

Columns: `victim_q` = queries to the black-box victim (the budget being
minimised); `L2` = perturbation size; `win` = did the final image fool the
victim. **White-box/transfer rows show `victim_q ≈ 0`** because they attack the
*surrogate* and only "win" if the example transfers.

How to read it as history:

- **FGSM → PGD** — the gradient mechanism (run on the surrogate). Reliable and
  strong but not L2-minimal (~2.0) and `victim_q=0` (they only transfer).
- **transfer** — the black-box *use* of those gradients: craft on the surrogate,
  grow ε until the victim flips. Cheap (4 queries) but no control over size.
- **Boundary → OPT → Sign-OPT → HopSkipJump** — the hard-label spine. Watch
  quality climb as the normal estimate improves: random walk → distance
  optimisation → cheap sign-gradient → Monte-Carlo normal (`1.76 → 1.62 → 1.53 →
  1.07`). Note Sign-OPT beats OPT on *both* L2 and queries — the "sign trick"
  needs one query per probe instead of a full search.
- **Biased Boundary → SQBA** — put the surrogate gradient *back into* the
  boundary search. The two branches fuse; SQBA reaches the smallest perturbation.

## The weaken-surrogate knob

`build_surrogate(frac=…, hidden=…, label_noise=…)` controls how well the
surrogate mimics the victim. The sweep runs SQBA with four surrogates of
decreasing quality (averaged over 5 images):

```
surrogate   agree%      L2  victim_q  white_only  fallbacks
strong       99.7   0.733       310     8.2/15        6.8
ok           92.5   0.950       352     6.6/15        8.4
weak         76.1   0.913       396     4.8/15       10.2
useless      41.7   1.065       475     2.0/15       13.0
```

As the surrogate degrades, `white_only` (free surrogate steps) **falls 8.2 → 2.0**,
`fallbacks` (paid query-based steps) **rise 6.8 → 13.0**, and `victim_q` **climbs
310 → 475**. This is SQBA detecting that the free gradient stopped helping and
paying for the Monte-Carlo estimate instead — **smoothly decaying into
HopSkipJump**. That degradation *is* the thesis of the paper: a good surrogate
buys query efficiency; a useless one costs nothing in attack quality but loses
the savings.

## Per-rung paper map

| Rung | Paper |
|---|---|
| FGSM | Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, 2014 |
| PGD | Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, 2017 |
| transfer | Papernot et al., *Practical Black-Box Attacks against Machine Learning*, 2016 |
| Boundary Attack | Brendel, Rauber & Bethge, *Decision-Based Adversarial Attacks*, ICLR 2018 |
| OPT | Cheng et al., *Query-Efficient Hard-label Black-box Attack*, ICLR 2019 |
| Sign-OPT | Cheng et al., *Sign-OPT: A Query-Efficient Hard-label Adversarial Attack*, ICLR 2020 |
| HopSkipJump | Chen, Jordan & Wainwright, *HopSkipJumpAttack*, IEEE S&P 2020 |
| Biased Boundary | Brunner et al., *Guessing Smart: Biased Sampling for Efficient Black-Box Adversarial Attacks*, ICCV 2019 |
| **SQBA** | **Park et al., *Hard-label based Small Query Black-box Adversarial Attack*, WACV 2024** |

`random` and `line` are pedagogical scaffolding, not published attacks — `line`
is really the *initialisation* step used inside Boundary Attack and OPT.

## Alignment with the SQBA paper

This was audited against the [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf).

**What matches** (the lineage and concepts are confirmed by the paper itself):

- SQBA = transfer + query-based, hard-label, built on **HopSkipJump's**
  gradient estimation — the paper states it "integrates the transfer based
  attack … and applies gradient-free optimisation introduced in [HSJA]".
- The paper's benchmarks are exactly four rungs here: **HSJA, Sign-OPT, Boundary
  Attack, Biased-BA**. It names **Biased-BA (Brunner)** as the closest prior
  setting — the precursor this ladder places right before SQBA.
- SQBA blends two gradients with a **boolean β switch** (Eq 7): β=1 uses the
  surrogate gradient `∇H_w`; on a local minimum β→0 and it uses the query-based
  `∇H_b` (the HSJA Monte-Carlo estimate). `attack_sqba` mirrors this:
  `surrogate_grad` is `∇H_w` (the "white_only" iters), `mc_normal` is `∇H_b` (the
  "fallback" iters), and a useless surrogate makes it **decay into HopSkipJump** —
  exactly β→0.

**Two implementations** are provided so you can have both a readable rung and a
reference:

- **`attack_sqba` (teaching)** — one surrogate gradient per step, "trust it while
  it makes progress, else fall back to a query-based estimate". Starts from the
  shared *loose* point so the ladder rungs share an origin. This is the version
  in the weaken-surrogate sweep.
- **`attack_sqba_full` (paper-faithful, Algorithm 1)** — implements the parts the
  teaching version omits, with helper functions mapped one-to-one to the paper's
  equations:
  - `grad_hw_multigrad` — the **multi-gradient method** (the paper's core novelty,
    Section 4.1, Eqs 10–11): sample surrogate gradients at several points
    `x + η·ṽ` along the perturbation path (`η ≥ 0.2`) and keep the candidate whose
    probe stays adversarial and lands closest to `x`, exploiting that the surrogate
    gradient rotates toward the optimal *perpendicular* direction as η grows (Fig 2).
  - `grad_hb_mc` — the HopSkipJump Monte-Carlo estimate `∇H_b` (Eq 6) with the
    paper's `pₜ = 10√(t+1)` schedule.
  - `eq8_eq9_update` — the Eq 8 α-line-search step + Eq 9 binary search.
  - a boolean **β switch** (Eq 7): use `∇H_w` while it makes healthy progress;
    on a local minimum (small α / no improvement) switch to `∇H_b`.
  - **initialisation** from `sign(∇surrogate)` then binary search (Algorithm 1).

**Still approximated** in `attack_sqba_full` (documented inline): the l∞ **Dual
Gradient Method** for the surrogate gradient is replaced by a unit-L2 cross-entropy
gradient (DGM details are in the paper's supplemental, not the main text), and the
multi-gradient path is sampled relative to the *current* iterate rather than a
fixed initial `ṽ`.

**Retrofitted into the teaching version**: the fallback probe size now uses the
paper's distance-scaled `δₜ = 0.01·‖x−x′‖` (Eq 6) instead of a fixed `0.01`. The
`pₜ` sample schedule is deliberately *not* retrofitted there — a fixed `B` keeps
the weaken-sweep trend clean to read.

In short: the **threat model, two-gradient structure, β-switch / decay into
HopSkipJump, multi-gradient estimation, and the whole lineage** are now all
represented faithfully in `attack_sqba_full`; `attack_sqba` stays deliberately
simple for reading.

## Honest caveats

This is a teaching model (64-dim digits), not a benchmark. Read the *mechanisms*
as faithful and the *exact numbers* as illustrative:

- **PGD ≈ FGSM in L2.** At a fixed L∞ radius both saturate the same ball, so
  PGD's advantage (reliability, success at smaller ε) doesn't show as a smaller
  L2 on this easy task.
- **`line` (scaffold) beats `boundary`/`opt` in L2** only because it initialises
  from the *nearest* class while rungs 3–5 deliberately start from a *loose* far
  point — so the comparison is about the optimisation, not the lucky start.
- **OPT/Sign-OPT shine in high dimensions.** On 64-dim digits their per-query
  advantage over HopSkipJump is muted; the lineage and query *trends* are
  faithful, the rankings between close methods are not a benchmark result.
- **One image is noisy**; the weaken sweep is averaged over 5 to make the trend
  readable.
- **`sqba-full` costs more than `sqba` here, and that's expected.** The
  multi-gradient method pays for extra probes per step to find a better direction,
  but its payoff (gradient rotation in high-dimensional image space, Fig 2) barely
  exists on 8×8 digits. On CIFAR-10/ImageNet — the paper's setting — that trade
  flips. Treat `sqba-full` as a faithful *algorithm*, not a faster *result* on
  this toy.

## Files

- [`lineage.py`](./lineage.py) — the script (heavily commented; every function
  states the paper and idea it implements). Contains both `attack_sqba` (teaching)
  and `attack_sqba_full` (paper-faithful Algorithm 1).
- [`README.md`](./README.md) — this file.
- The source paper (open-access):
  [Park, Miller & McLaughlin, *Hard-label based Small Query Black-box Adversarial Attack*, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf).
