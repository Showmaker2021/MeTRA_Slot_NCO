# Handoff Report: Milestone M0 ($d_{\text{ins}}$ Operator Unit Tests)

**Author:** `explorer_m0_2`  
**Target Recipient:** `parent` (`c3281cb8-88ec-4601-9bd8-e3191fb328ba`)  
**Date:** 2026-08-06  
**Working Directory:** `d:/NCO NEW/rl4co/.agents/explorer_m0_2`

---

## 1. Observation

1. **Existing Code & Tests Inspected:**
   - Target Implementation File: `rl4co/data/insertion_cost.py` (98 lines) contains `compute_pairwise_distance_matrix` (lines 7–28) and `compute_marginal_insertion_cost` (lines 31–98).
   - Target Test File: `tests/test_insertion_cost.py` (39 lines) contains 3 test functions:
     - `test_compute_pairwise_distance_matrix` (lines 6–14): basic 3-4-5 right triangle test for unbatched coordinates.
     - `test_marginal_insertion_cost_basic` (lines 17–26): batch size 2, 5 customers, checking shape `(2, 5, 5)` and self-insertion cost `0.0`.
     - `test_knn_sparsification` (lines 29–38): batch size 1, 10 customers, $k=3$, checking max non-inf count per row $\le k+1$.

2. **Observed Gaps in Existing Test Suite:**
   - **No Analytical Value Verification:** Insertion costs $d_{\text{ins}}(i, j) = d(D, i) + d(i, j) - d(D, j)$ are never checked against exact mathematical hand-calculated values.
   - **No Default $k=15$ Coverage:** Requirement R1 specifies default $k=15$. Existing tests only test $k=3$ on $N=10$ or $k=\text{None}$.
   - **No $N \le k$ Edge Case Verification:** When $N \le k$ (e.g. $N=5, k=15$), existing tests do not assert that no elements are masked to `inf` and all $N \times N$ matrix elements remain finite.
   - **No Small $N$ Edge Case Verification:** $N=1$, $N=2$, $N=5$ cases are unverified.
   - **No Unbatched / Batched Consistency Checks:** Input shape `(N, 2)` vs `(B, N, 2)` consistency is unverified.
   - **No Custom Depot Coordinate Tests:** Custom depot inputs (`(2,)`, `(1,2)`, `(B, 1, 2)`) and non-standard depot locations are unverified.
   - **No Mathematical Invariant Asserts:** Distance matrix symmetry, zero diagonal, non-negativity of insertion costs, and triangle inequality are untested.

3. **Execution Environment Verification:**
   - Test execution using `conda run -n ec_nco python -m pytest tests/test_insertion_cost.py` was executed.
   - Python package `rl4co-0.6.0` was installed in editable mode in `ec_nco` environment.

---

## 2. Logic Chain

1. **Step 1 (Observation 1 -> Analysis):** The existing test suite in `tests/test_insertion_cost.py` has 3 basic test functions that perform superficial shape and count checks.
2. **Step 2 (Observation 2 -> Gap Identification):** Requirement R1 mandates a vectorized $k$-NN sparsified $d_{\text{ins}}$ operator with default $k=15$. To ensure production reliability across datasets ($N \in \{50, 100, 200, 500\}$) and downstream models (Slot Attention and METRA loss), the unit test suite must rigorously validate:
   - Mathematical formula correctness $d_{\text{ins}}(i, j) = d(D, i) + d(i, j) - d(D, j)$.
   - Self-insertion identity $d_{\text{ins}}(i, i) = 0.0$.
   - Sparsification behavior ($k=15$, exact $k+1$ non-inf count per row, non-neighbors masked to `inf`).
   - Boundary condition $N \le k$ where no entries should be masked to `inf`.
   - Small instance edge cases ($N=1, 2, 5$).
   - Batch sizes $B \ge 1$ and unbatched 2D inputs.
   - Random seed reproducibility and tensor dtype/device preservation.
3. **Step 3 (Analysis -> Specification):** A comprehensive 16-test specification organized into 5 modular test classes was created in `d:/NCO NEW/rl4co/.agents/explorer_m0_2/analysis.md`.

---

## 3. Caveats

- **No Code Modifications Made:** As per agent role guidelines for read-only investigation (`🔒 Key Constraints`), no source code files in `rl4co/data/` or `tests/` were modified. The test specification and proposed implementation are provided in `analysis.md` for the implementer agent.
- **CUDA Device Testing:** Tests are specified for CPU and automatically run on CUDA if available.

---

## 4. Conclusion

The existing unit tests in `tests/test_insertion_cost.py` provide minimal initial coverage. A complete 16-test specification covering mathematical analytical values, default $k=15$ sparsification, edge cases ($N \le k$, $N=1, 2, 5$, unbatched inputs, multi-batch $B>1$, random seeds), custom depots, and distance matrix invariants has been documented in `d:/NCO NEW/rl4co/.agents/explorer_m0_2/analysis.md`.

---

## 5. Verification Method

To verify the test suite specification and environment readiness:

1. **Inspect Analysis Specification:**
   - File: `d:/NCO NEW/rl4co/.agents/explorer_m0_2/analysis.md`

2. **Run PyTest on Existing Test Suite:**
   - Command: `conda run -n ec_nco python -m pytest tests/test_insertion_cost.py -v`
   - Expected Result: Existing 3 tests pass cleanly in `ec_nco` environment.

3. **Validation Criteria for Future Implementer:**
   - Implementer expands `tests/test_insertion_cost.py` with the 16 specified tests.
   - All 16 tests pass cleanly with 100% coverage of `rl4co/data/insertion_cost.py`.

