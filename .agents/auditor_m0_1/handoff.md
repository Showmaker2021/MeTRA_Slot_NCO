# Handoff Report: Forensic Integrity Audit — Milestone M0 ($d_{\text{ins}}$ Operator & Unit Tests)

**Agent**: `auditor_m0_1`  
**Working Directory**: `d:/NCO NEW/rl4co/.agents/auditor_m0_1`  
**Target Work Product**: `rl4co/data/insertion_cost.py` & `tests/test_insertion_cost.py`  
**Integrity Mode**: `development` (per `ORIGINAL_REQUEST.md`)  
**Verdict**: **CLEAN**  

---

## 1. Observation

Direct observations and evidence collected during forensic audit:

1. **Source Code Integrity (`rl4co/data/insertion_cost.py`)**:
   - `compute_pairwise_distance_matrix(coords)` uses `torch.cdist(coords, coords, p=2.0)` to compute Euclidean distances natively without allocating intermediate broadcast subtraction tensors.
   - `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)` evaluates $d_{\text{ins}}(i, j) = c(D, i) + c(i, j) - c(D, j)$ via `dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)`.
   - `d_ins = torch.clamp(d_ins, min=0.0)` prevents numerical float32 underflow below 0.0.
   - Diagonal self-insertion is explicitly zeroed using `eye_mask` and `d_ins.masked_fill(eye_mask, 0.0)`.
   - $k$-NN sparsification uses `torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)` and `d_ins.masked_fill(~knn_mask, float("inf"))` to mask non-neighbor entries to `inf` when $k < N$.
   - No hardcoded test outputs, return constants, or facade logic were found in any function.

2. **Test Suite Integrity (`tests/test_insertion_cost.py`)**:
   - `test_compute_pairwise_distance_matrix`: Verifies 2D & 3D inputs using 3-4-5 right triangle geometry (`coords = [[0,0], [3,0], [0,4]]`) and scaled batch inputs.
   - `test_marginal_insertion_cost_basic`: Independently loops through batch, $i$, and $j$ to check $d_{\text{ins}}[b,i,j] == ||locs[b,i] - depot[b,0]|| + ||locs[b,i] - locs[b,j]|| - ||locs[b,j] - depot[b,0]||$, non-negativity $d_{\text{ins}} \ge 0.0$, and zero diagonal.
   - `test_knn_sparsification`: Asserts exact non-inf count per row ($k+1$) and inf count overall for $N=20, k=15$ and $N=10, k=3$.
   - `test_edge_cases`: Tests $N \le k$ ($N=5, k=15$), unbatched 2D inputs, $k=\text{None}$, and 1D/2D/3D depot tensors.
   - `test_customer_at_depot_and_colocation`, `test_gradient_flow_insertion_cost`, and `test_clustered_spatial_distribution`: Verify numerical robustness against zero distance, autograd backpropagation, and spatial cluster isolation.
   - Tests do NOT rely on self-certification or hardcoded expectations from implementation constants.

3. **PyTest Execution in `ec_nco` Environment**:
   - Command: `D:\Miniconda\miniconda3\envs\ec_nco\python.exe -m pytest tests/test_insertion_cost.py -v`
   - Verbatim Output:
     ```text
     ============================= test session starts =============================
     platform win32 -- Python 3.10.19, pytest-9.1.1, pluggy-1.6.0 -- D:\Miniconda\miniconda3\envs\ec_nco\python.exe
     cachedir: .pytest_cache
     rootdir: D:\NCO NEW\rl4co
     configfile: pyproject.toml
     plugins: anyio-4.14.2, hydra-core-1.3.5
     collecting ... collected 7 items

     tests/test_insertion_cost.py::test_compute_pairwise_distance_matrix PASSED [ 14%]
     tests/test_insertion_cost.py::test_marginal_insertion_cost_basic PASSED  [ 28%]
     tests/test_insertion_cost.py::test_knn_sparsification PASSED             [ 42%]
     tests/test_insertion_cost.py::test_edge_cases PASSED                     [ 57%]
     tests/test_insertion_cost.py::test_customer_at_depot_and_colocation PASSED [ 71%]
     tests/test_insertion_cost.py::test_gradient_flow_insertion_cost PASSED   [ 85%]
     tests/test_insertion_cost.py::test_clustered_spatial_distribution PASSED [100%]

     ============================== 7 passed in 3.19s ==============================
     ```

4. **Empirical Stress Test Execution**:
   - Command: `D:\Miniconda\miniconda3\envs\ec_nco\python.exe -c "import torch; from rl4co.data.insertion_cost import compute_marginal_insertion_cost; locs = torch.randn(10, 100, 2); d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15); assert d_ins.shape == (10, 100, 100); assert (torch.isinf(d_ins).sum(dim=-1) == 84).all(); print('Stress test 1 passed!'); locs2 = torch.randn(50, 2); d_ins2 = compute_marginal_insertion_cost(locs2, k_neighbors=15); assert d_ins2.shape == (50, 50); print('Stress test 2 passed!')"`
   - Output: `Stress test 1 passed! Stress test 2 passed!`

---

## 2. Logic Chain

1. **Step 1 (Hardcoded / Facade Analysis)**:
   - Evaluated `rl4co/data/insertion_cost.py` for fixed return values or missing logic.
   - Observation shows genuine tensor distance and marginal cost calculation using PyTorch operations (`cdist`, `norm`, `masked_fill`, `topk`). Zero facade patterns detected.

2. **Step 2 (Self-Certifying / Fake Test Check)**:
   - Evaluated `tests/test_insertion_cost.py` for circular assertions.
   - Observations show tests construct explicit ground-truth inputs (3-4-5 right triangles) or calculate expected values independently of `compute_marginal_insertion_cost`.

3. **Step 3 (Behavioral & Environment Verification)**:
   - Executed PyTest in `ec_nco` environment. All 7 unit tests passed cleanly in 3.19s.
   - Executed independent stress tests for $N=100, B=10$ and $N=50$ 2D input. Sparsification counts ($(N - (k+1)) = 84$ inf values per row) matched mathematical expectation.

4. **Step 4 (Dependency & Execution Audit)**:
   - Core deliverable uses PyTorch built-in operations. No external prohibited packages or execution shortcuts used.

---

## 3. Caveats

- **Device Execution**: Tests ran on CPU in the `ec_nco` conda environment. PyTorch `cdist` and `topk` are fully CUDA-compatible when tensors are moved to GPU.
- **No Caveats**: No integrity violations or missing coverage were detected.

---

## 4. Conclusion & Forensic Audit Report

### Forensic Audit Report

**Work Product**: `rl4co/data/insertion_cost.py` & `tests/test_insertion_cost.py` (Milestone M0)  
**Profile**: General Project (Integrity Mode: `development`)  
**Verdict**: **CLEAN**

#### Phase Results
- **Hardcoded test result detection**: PASS — No embedded test strings or constant outputs.
- **Facade detection**: PASS — Full mathematical PyTorch vectorized implementation.
- **Pre-populated artifact detection**: PASS — No stale result files or logs pre-existing in repo.
- **Self-certifying test detection**: PASS — Unit tests independently verify formula and geometric ground truths.
- **Behavioral & unit test execution**: PASS — 7/7 tests passed in `ec_nco` environment.
- **Dependency audit**: PASS — Uses PyTorch standard operations.

---

## 5. Verification Method

To independently verify Milestone M0 forensic audit results:

1. **Execute Unit Tests**:
   ```powershell
   D:\Miniconda\miniconda3\envs\ec_nco\python.exe -m pytest tests/test_insertion_cost.py -v
   ```
2. **Execute Stress Test**:
   ```powershell
   D:\Miniconda\miniconda3\envs\ec_nco\python.exe -c "import torch; from rl4co.data.insertion_cost import compute_marginal_insertion_cost; locs = torch.randn(10, 100, 2); d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15); assert (torch.isinf(d_ins).sum(dim=-1) == 84).all()"
   ```
3. **Inspect Code Files**:
   - `rl4co/data/insertion_cost.py`
   - `tests/test_insertion_cost.py`
