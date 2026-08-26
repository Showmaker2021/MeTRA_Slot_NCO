# Experiment Plan — Metric-Aware Slot NCO (Variant D)

> Goal: make Variant D win **clearly** while staying **defensible** — the advantage
> must follow from the mechanism, not from a lucky cell in the grid.

---

## 0. Summary

Variant D's edge is legitimate only where **routing-structural similarity diverges
from coordinate similarity**, and only where the `d_ins` target is *genuinely*
routing-aware (not a Euclidean-correlated proxy). Plan:

1. Fix `d_ins` (gating step) so D's target actually separates from C's.
2. Sweep two physically-motivated axes — **metric alignment** × **generalization**.
3. Include controls (M0 in-dist) and a **mechanism metric** so the win reads as a
   claim backed by a dose–response, not an isolated number.

---

## 1. Gating step — fix the `d_ins` target (before any experiment)

**Problem:** [`insertion_cost.py`](../data/insertion_cost.py) lines 8–15 uses the
Clarke-Wright proxy

```
d_ins(i, j) = dist(D, i) + dist(i, j) − dist(D, j)
```

which can correlate strongly with Euclidean distance. If it does, D's target ≈ C's
target → **D ≈ C trivially**, killing the flagship D-vs-C comparison.

**Fix:** construction-aware insertion cost.

- Greedy nearest-neighbor route construction, per instance, capacity-aware.
- Marginal insertion cost of each unvisited customer into the cheapest feasible gap
  of the current partial routes (true cheapest-insertion, not a savings proxy).
- Store in the same sparse `(idx int16, val float32)` shape `(B, N, k)` so the
  aggregation `_aggregate_d_ins_sparse` in [`metric_loss.py`](../nn/metric_loss.py)
  and the on-the-fly path in [`model.py`](model.py) (lines 271–282) stay unchanged.

**Verify:** on synthetic instances the new `d_ins` must *decorrelate* from Euclidean
where intended — measured by the conflict index in §4.

---

## 2. Comparison arms

| arm | slots | metric target          | role                                             |
|-----|-------|------------------------|--------------------------------------------------|
| POMO| ✗     | —                      | vanilla baseline (no slots)                      |
| B   | ✓     | none (α_metric = 0)     | isolates slot-injection from metric loss         |
| C   | ✓     | Euclidean centroid dist| geometry-aware baseline (Variant C)              |
| D   | ✓     | insertion cost (fused) | proposed (Variant D)                             |

All arms share the encoder/decoder, budget (epochs, samples), and **≥ 3 seeds**.

---

## 3. Experiment matrix — two axes, physically motivated

### Axis 1 — Metric alignment `M` (how well geometry ≈ routing)

- **M0 · aligned** — standard uniform CVRP. Geometry ≈ routing.
  *Prediction: D ≈ C ≈ POMO (control).*
- **M1 · moderate conflict** — geographic clusters that are (partly) routing-
  irrelevant: spatial neighbors with different marginal route roles (demand ankles,
  corridor-vs-spur).
- **M2 · strong conflict** — routing consequences differ sharply despite spatial
  proximity (customers equidistant from the depot, one central to many routes, one
  on an isolated spur).

### Axis 2 — Generalization `G`

- **G0 · in-dist** — train N=100 → test N=100.
- **G1 · size shift** — train N=100 → test N=200/500 *(targets the documented
  N_train ≠ N_test weakness)*.
- **G2 · OOD shift** — train on M0 → test on M1/M2 *(money cell: metric-conflict OOD)*.

### Grid (train → evaluate)

| cell                     | prediction              |
|--------------------------|-------------------------|
| M0 · G0 in-dist          | D ≈ C ≈ POMO            |
| M1 · G0 conflict in-dist | D >  C ≥ POMO           |
| M2 · G0 conflict in-dist | D ≫  C ≥ POMO           |
| M0 · G1 (100 → 200/500)  | D >  C ≥ POMO           |
| M2 · G2 OOD (train M0)   | **D ≫ C ≥ POMO** (headline) |

---

## 4. Mechanism metric (what makes the win *motivated*)

Define per test-distribution **conflict index**

```
ρ = Spearman-rank( d_ins frontier , Euclidean distance )
```

High (≈ 1) in M0, low in M2. Then:

- Report a **dose–response**: D-vs-C margin as a function of ρ.
- Prediction: the margin grows monotonically as ρ falls — tying the win *to the
  mechanism*.
- This is the reviewer-facing evidence that the win comes from a routing-aware
  metric, not from incidental configuration.

---

## 5. Evaluation protocols (defensibility)

- **Strong solver baseline**: optimality gap vs HGS / LKH-3. M-conflict optima are
  often unknown; never rely solely on POMO-relative numbers.
- **Search, not just policy**: also report POMO + sampling / beam — the decoder
  interacts with representation quality.
- **Fair budget**: equal epochs/batches/seeds (≥ 3); paired comparison on identical
  instance sets.
- **Training diagnostics** per arm: `train/aux_loss`, `slot_entropy`,
  `metric_violation` / `metric_lambda` for the paper's training narrative.

---

## 6. Contribution packaging

Even if D is a *mixed* story, the following is an independent, citable contribution:

> **Metric-conflict generators + the conflict-index benchmark** — a controlled
> benchmark for routing-vs-coordinate metric alignment in NCO.

This is valuable regardless of the headline outcome, so the paper has merit under a
skeptical reading.

---

## 7. Milestones

1. `d_ins` → construction-aware (gating, §1)
2. Metric-conflict generators (M0/M1/M2) + conflict index (§4)
3. Train arms POMO, B, C, D on M0 @ N=100 (§2, §3)
4. Evaluate full matrix; build dose–response plot (§4, §5)
5. Write up

---

## 8. Open items / risks

- **Highest risk**: construction-aware `d_ins` still correlating with Euclidean. If
  the dose–response (§4) is flat, the D-vs-C story fails — hence milestone 1 first.
- **Optimum availability**: M-conflict instances need a strong-solver proxy (§5).
- **Slot count K / alpha_metric / beta_entropy**: must be held identical across B/C/D
  to keep the ablation clean.
- **Timing/cost**: the OOD + size-shift cells multiply training runs; consider a
  reduced seed count for exploratory M2 generators before committing the full grid.
