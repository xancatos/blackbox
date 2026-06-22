# Black-box adversarial attack lineage → SQBA

CPU-only, from-scratch implementations of **twelve black-box adversarial attacks**
run on one hard-label victim, so you can read the history of the field as a single
table — from the simplest blind baseline up to **SQBA** — *Small-Query Black-Box
Attack*, from "Hard-label based Small Query Black-box Adversarial Attack",
Jeonghwan Park, Paul Miller & Niall McLaughlin (Queen's University Belfast),
WACV 2024 ([open-access PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf)).
The paper reports SQBA reaching **~5× higher attack success rate** than the
benchmarks at small query budgets (100, 250) — a result the CIFAR demo below
reproduces.

The attack ladder lives in [`attacks.py`](./attacks.py) (shared, model- and
data-agnostic), driven by two demos:

| Demo | Victim / data | Deps | Runtime | Best for |
|---|---|---|---|---|
| [`lineage.py`](./lineage.py) | sklearn MLP on 8×8 **digits** (64-dim) | `numpy`, `scikit-learn` | seconds | reading the whole ladder fast; the clean weaken-surrogate sweep |
| [`cifar_lineage.py`](./cifar_lineage.py) | tiny **CNNs** on 2-class **CIFAR-10** (3072-dim) | `numpy`, `torch` (CPU), `scipy` | ~minutes (cached after) | the paper's setting: ASR-vs-budget, cross-arch transfer, where SQBA's multi-gradient pays off |

## Run it

```bash
uv run lineage.py          # digits — fast, numpy + scikit-learn only
uv run cifar_lineage.py    # CIFAR-10 — first run downloads ~170 MB data + CPU torch
uv run diagrams/render_all.py   # regenerate the dark-theme slide diagrams (see diagrams/)

# run the visual tutorial notebook end-to-end:
uv run --with numpy,torch,scipy,matplotlib,lpips,jupyter,nbconvert,ipykernel \
  jupyter nbconvert --to notebook --execute --inplace lineage_tutorial.ipynb
```

[`uv`](https://docs.astral.sh/uv/) reads the PEP-723 header in each script and
provisions deps in an ephemeral environment — nothing installed globally.
`cifar_lineage.py` caches the CIFAR tar, the 2-class subset, and the trained nets
under `~/.cache/blackbox`, so only the **first** run pays the download/training
cost; later runs are ~1–2 min.

## The core idea

Black-box adversarial-attack research advanced along one axis: **how little the
attacker is allowed to know**. Each tier below relaxes one assumption of the tier
above. The script is organised the same way.

| Tier | Attacker assumes | Attacks (year) |
|---|---|---|
| **white-box** | full gradients (of a surrogate) | FGSM '14, PGD '17 |
| **transfer** | a surrogate + ~0 victim queries | Papernot '16 |
| **hard-label / decision-based** | only the victim's top-1 **label** | Boundary '18, OPT '19, Sign-OPT '20, HopSkipJump '20, **Triangle '22** |
| **surrogate-assisted** | label **+** a surrogate's gradient | Biased Boundary '19, **SQBA '24** |

**Hard-label** is the hard case: the oracle returns only the predicted class — no
probabilities. The output is a step function (flat, with a jump at the boundary),
so you can't differentiate it. Every hard-label attack is a different way to
extract a usable signal from the single bit *"did the label change?"*.

Three primitives are reused everywhere:

1. **Binary search to the boundary** — bisect the line between an adversarial
   point and the clean image to land exactly on the boundary (the closest
   adversarial point along that line). Turns a discrete label into a real-valued
   distance you can minimise.
2. **The boundary normal** — near the boundary, the fastest label-changing
   direction is its normal. Attacks differ in *how they estimate it*: Boundary
   Attack doesn't (random walk); HopSkipJump estimates it by **querying**
   (costly); SQBA reads it **for free** from a surrogate's gradient.
3. **Geometric search & subspace reduction** — Triangle Attack uses the law of
   sines on a triangle constructed between the original image and adversarial points
   in a low-frequency Discrete Cosine Transform (DCT) subspace, avoiding gradient estimation.

**SQBA** fuses the two branches: a decision-based boundary search whose normal
comes from a white-box surrogate. That's the "small query" win — the victim
queries HopSkipJump spends *estimating* the normal are replaced by a free
surrogate backward pass. When the surrogate stops helping, SQBA falls back to the
query-based estimate, gracefully decaying into HopSkipJump.

## Example output — digits (`lineage.py`)

```
DIGITS ATTACK LINEAGE  (one image; victim is hard-label only)
attack         yr  victim_q      L2  win   key idea
-- HARD-LABEL ----------------------------------------------------------
random         -       1500   2.199    Y   blind noise (scaffold)
line           -          8   1.667    Y   aim at nearest class + binsearch (scaffold)
boundary      '18       500   1.764    Y   random walk along boundary
opt           '19       699   1.623    Y   minimize g(theta), zeroth-order
sign-opt      '20       274   1.528    Y   g(theta) via SIGN of dir. deriv
hopskipjump   '20       401   1.072    Y   MC boundary-normal estimate
triangle      '22       107   1.257    Y   geometric law of sines in DCT subspace
-- SURROGATE -----------------------------------------------------------
biased-bdry   '19       500   1.016    Y   surrogate-biased boundary walk
sqba          '24       307   0.893    Y   teaching: single surrogate normal + fallback
sqba-full     '24       550   0.920    Y   paper Algo 1: multi-gradient + beta switch
```

Columns: `victim_q` = queries to the black-box victim (the budget being
minimised); `L2` = perturbation size; `win` = did the final image fool the victim.
Reading it as history along the hard-label spine — quality climbs as the
boundary-normal estimate improves: random walk → distance optimisation → cheap
sign-gradient → Monte-Carlo normal (`1.76 → 1.62 → 1.53 → 1.07`), then the
geometric Triangle Attack (`1.257`) achieves high query efficiency.
Finally, the surrogate-assisted rungs (Biased Boundary, SQBA) reach the smallest perturbation.
(FGSM/PGD/transfer need pixel gradients, so they're shown in the CIFAR demo.)

There are **two SQBA rows**: `sqba` (teaching: one surrogate gradient per step)
and `sqba-full` (paper-faithful Algorithm 1: multi-gradient ∇H_w + β switch +
sign-gradient init). On this 64-dim toy the full version costs *more* for a similar
L2 — its multi-gradient method exploits gradient rotation in *high* dimensions
(paper Fig 2), which barely exists on 8×8 digits. **On CIFAR below it wins.**

## CIFAR-10: the paper's setting (`cifar_lineage.py`)

Tiny CNNs on a 2-class slice (**airplane vs automobile**, 3072-dim). We train two surrogates: a **White-box Replica Surrogate** (identical architecture, 94.5% accuracy, 100% agreement) and a **Black-box Generic Surrogate** (different architecture, 90.5% accuracy, 92% agreement). 
Perturbation is the paper's relative budget `ρ = ‖x_adv − x‖ / ‖x‖`, success `ρ ≤ 0.25`. The whole ladder runs, and the **ASR-vs-query-budget** table (the shape of the paper's Tables 3–6) is the headline:

```
ASR vs QUERY BUDGET  (success = rho<=0.25 within budget; 8 images, strong surrogate)
attack           Q=100    Q=250   Q=1000
boundary           12%      12%      12%
sign-opt           12%      12%      62%
hopskipjump        25%      25%      38%
triangle           38%      62%      88%
biased-bdry       100%     100%     100%
sqba              100%     100%     100%
sqba-full          75%     100%     100%
```

This **reproduces the paper's central result**: at small budgets (100, 250) the
surrogate-assisted attacks reach ~100% ASR while the pure query-based methods
(HopSkipJump, Sign-OPT) are stuck at 12–25%. Crucially, the pure black-box
**Triangle Attack** outperforms other non-surrogate decision attacks by a massive margin
(62% ASR at Q=250, and 88% ASR at Q=1000) by utilizing its low-frequency DCT subspace search.

## The weaken-surrogate knob

`build_surrogate(frac=…, label_noise=…, …)` controls how well the surrogate mimics
the victim. The sweep runs SQBA with surrogates of decreasing quality. On **digits**
(10 classes, agreement spans a wide range) the trend is cleanest:

```
surrogate   agree%      L2  victim_q  white_only  fallbacks
strong       99.4   0.741       301     8.4/15        6.6
ok           92.2   0.853       258     9.8/15        5.2
weak         82.5   0.965       387     5.4/15        9.6
useless      44.2   1.076       419     3.6/15       11.4
```

As the surrogate degrades, `white_only` (free surrogate steps) **falls 8.4 → 3.6**,
`fallbacks` (paid query-based steps) **rise 6.6 → 11.4**, and `victim_q` **climbs
301 → 419**. This is SQBA detecting that the free gradient stopped helping and
paying for the Monte-Carlo estimate instead — **smoothly decaying into
HopSkipJump**.

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
| **Triangle Attack** | **Wang et al., *Triangle Attack: A Query-efficient Decision-based Adversarial Attack*, ECCV 2022** |
| Biased Boundary | Brunner et al., *Guessing Smart: Biased Sampling for Efficient Black-Box Adversarial Attacks*, ICCV 2019 |
| **SQBA** | **Park et al., *Hard-label based Small Query Black-box Adversarial Attack*, WACV 2024** |

`random` and `line` are pedagogical scaffolding, not published attacks — `line`
is really the *initialisation* step used inside Boundary Attack and OPT.

## Alignment with the SQBA and Triangle papers

This implementation was audited against the [SQBA paper](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf)
and the [Triangle Attack paper](https://arxiv.org/abs/2112.06569).

**What matches:**
- SQBA = transfer + query-based, hard-label, built on **HopSkipJump's** gradient estimation.
- The Triangle Attack combines the law of sines with Discrete Cosine Transform (DCT) to sample 2D subspaces. In [attacks.py](./attacks.py), [attack_triangle](./attacks.py#L764) implements the low-frequency sampling, Law of Sines candidate calculation, and binary search steps exactly matching Algorithm 1.

## Honest caveats

These are teaching models, not a benchmark. Read the *mechanisms* as faithful and
the *exact numbers* as illustrative.

- **Digits is tiny (64-dim, 10 classes).** OPT/Sign-OPT and the multi-gradient
  method shine in high dimensions, so on digits the rankings between close methods
  aren't a benchmark result — the *lineage and query trends* are the point. CIFAR
  is where the high-dimensional behaviour (incl. `sqba-full` winning) shows.
- **`line` (scaffold) can beat `boundary` in L2** only because it initialises from
  the *nearest* class while the decision-based rungs deliberately start from a
  *loose* far point — the comparison is about the optimisation, not the start.
- **CIFAR victim accuracy is 94.5%** (airplane vs automobile classification, chosen so the boundary is distinct and models reach >=90% accuracy). Attacks only target test images the victim classifies correctly, so success stays meaningful.
- **PGD ≈ FGSM in L∞ size**; PGD's edge is reliability, not a smaller perturbation
  at the same ε.
- **Few images, averaged** to keep CPU time low; the per-budget ASR is over 8
  images and the weaken sweep over 4–5, so treat single percentages as indicative.

## Files

- [`attacks.py`](./attacks.py) — the shared, model-/data-agnostic attack ladder:
  all 12 attacks plus `attack_sqba` (teaching) and `attack_sqba_full` (paper-faithful
  Algorithm 1). Heavily commented; every function names the paper and idea.
- [`lineage.py`](./lineage.py) — digits demo (numpy + scikit-learn, runs in seconds).
- [`cifar_lineage.py`](./cifar_lineage.py) — CIFAR-10 2-class demo (tiny CPU CNNs):
  single-image ladder, **ASR-vs-query-budget** table, and the weaken sweep.
- [`README.md`](./README.md) — this file.
- The source papers (open-access):
  - [Park, Miller & McLaughlin, *Hard-label based Small Query Black-box Adversarial Attack*, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf).
  - [Wang et al., *Triangle Attack: A Query-efficient Decision-based Adversarial Attack*, ECCV 2022](https://arxiv.org/abs/2112.06569).
