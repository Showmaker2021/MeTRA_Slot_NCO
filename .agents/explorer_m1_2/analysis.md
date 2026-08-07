# PyTorch Dataset Compatibility and Unit Test Specifications (`generate_slot_dataset.py`)

## Executive Summary
This document delivers a detailed investigation into PyTorch dataset conventions, `torch.load()` compatibility, and unit test specifications for Milestone M1 (`rl4co/data/generate_slot_dataset.py` and `tests/test_generate_slot_dataset.py`).

Milestone M1 establishes an offline caching CLI to precompute customer instances and $k$-NN sparsified marginal insertion cost matrices ($d_{\text{ins}}$) for $N \in \{50, 100, 200, 500\}$ across **Uniform** and **Clustered (GMM)** distributions, saved in `.pt` format.

---

## 1. `rl4co` Dataset Conventions & PyTorch `torch.load()` Compatibility

### A. Existing `rl4co` Dataset Infrastructure vs. `.pt` Requirement
1. **Existing `.npz` Pipeline (`rl4co/data/utils.py`)**:
   - Standard dataset generation in `rl4co/data/generate_data.py` uses `.npz` files (NumPy compressed arrays) loaded via `load_npz_to_tensordict()`.
   - `TensorDict` (from `tensordict`) is the primary data structure passed to models and environments.
2. **Milestone M1 `.pt` Caching Requirement**:
   - Requirement R1 explicitly specifies saving precomputed instances + sparsified $d_{\text{ins}}$ matrices to disk in PyTorch binary `.pt` format (`torch.save()` / `torch.load()`).
   - `.pt` serialization is native to PyTorch and eliminates NumPy-to-PyTorch tensor conversion overhead during model training and dataset loading.

### B. `torch.load()` Compatibility & Security Analysis
- **PyTorch 2.6+ `weights_only` Security Standard**:
  - Recent PyTorch releases enforce or default `weights_only=True` in `torch.load()` to prevent arbitrary code execution vulnerabilities from unpickling untrusted objects.
  - If a dataset is saved as a custom object or raw `TensorDict` object, `torch.load(path, weights_only=True)` can fail unless custom class globals are explicitly whitelisted.
  - **Optimal Compatibility Design**: Save datasets as standard Python dictionaries containing PyTorch Tensors (`dict[str, torch.Tensor]`):
    ```python
    dataset_dict = {
        "locs": locs_tensor,  # float32, (num_samples, N, 2)
        "depot": depot_tensor, # float32, (num_samples, 1, 2) or (num_samples, 2)
        "d_ins": d_ins_tensor, # float32, (num_samples, N, N)
    }
    torch.save(dataset_dict, filepath)
    ```
  - This guarantees `torch.load(filepath, weights_only=True, map_location="cpu")` loads seamlessly across all PyTorch versions (1.x and 2.x+) without security warnings or unpickling errors.

### C. Interoperability with `rl4co.data.dataset`
`rl4co/data/dataset.py` provides `FastTdDataset` and `TensorDictDataset`. Loaded `.pt` dictionary structures integrate directly with `TensorDict` and `FastTdDataset`:
```python
import torch
from tensordict import TensorDict
from rl4co.data.dataset import FastTdDataset, TensorDictDataset

data_dict = torch.load(file_path, weights_only=True, map_location="cpu")
td = TensorDict(data_dict, batch_size=data_dict["locs"].shape[0])
dataset = FastTdDataset(td)  # or TensorDictDataset(td)
```
Indexing `dataset[i]` returns an individual problem instance dictionary/TensorDict with keys `"locs"`, `"depot"`, and `"d_ins"`.

---

## 2. Dataset Generation Engine Specification (`generate_slot_dataset.py`)

### A. CLI Argument Interface
`rl4co/data/generate_slot_dataset.py` must expose the following command line interface:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_dir` | `str` | `"data/slot_datasets"` | Directory path where generated `.pt` files will be stored. |
| `--num_samples` | `int` | `1000` | Number of problem instances per dataset file. |
| `--graph_sizes` | `int` (nargs+) | `[50, 100, 200, 500]` | Problem sizes $N$ to generate. |
| `--distributions` | `str` (nargs+) | `["uniform", "clustered"]` | Data distributions to generate. |
| `--k_neighbors` | `int` | `15` | Number of nearest neighbors for $k$-NN sparsification of $d_{\text{ins}}$. |
| `--seed` | `int` | `1234` | Random seed for reproducibility. |
| `--overwrite` / `-f` | `flag` | `False` | Overwrite existing dataset files if present. |

### B. Distribution Generation Mechanics
1. **Uniform Distribution (`"uniform"`)**:
   - `locs`: Uniformly sampled in $[0, 1]^2$ with shape `(num_samples, N, 2)`.
   - `depot`: Uniformly sampled in $[0, 1]^2$ with shape `(num_samples, 1, 2)` or fixed center $(0.5, 0.5)$.
2. **Clustered GMM Distribution (`"clustered"`)**:
   - Samples cluster centers $C_k \sim \text{Uniform}([0.2, 0.8]^2)$ with number of clusters $M \in [3, 7]$.
   - Samples node coordinates around cluster centers with cluster standard deviation $\sigma \sim \text{Uniform}([0.03, 0.08])$.
   - Clamps/wraps node coordinates to $[0.0, 1.0]^2$.

### C. $d_{\text{ins}}$ Calculation & Sparsification
For each generated batch `(locs, depot)`:
- Calls `rl4co.data.insertion_cost.compute_marginal_insertion_cost(locs, k_neighbors=k_neighbors, depot_loc=depot)`.
- Produces tensor `d_ins` of shape `(num_samples, N, N)` where self-insertion entries `d_ins[b, i, i] == 0.0` and non-k-NN neighbor entries equal `float('inf')`.

---

## 3. Formulated Unit Test Suite Specification (`tests/test_generate_slot_dataset.py`)

The test suite in `tests/test_generate_slot_dataset.py` will validate CLI execution, file creation, dataset loading, tensor shapes, and mathematical values across 10 structured test cases:

```python
"""
Unit test suite for offline slot dataset generator CLI (rl4co/data/generate_slot_dataset.py)
Validates CLI argument parsing, file generation, torch.load compatibility, TensorDict integration,
tensor shapes, distribution properties, and d_ins mathematical invariants.
"""

import pytest
import os
import sys
import tempfile
import torch
import numpy as np
from tensordict import TensorDict
from rl4co.data.dataset import FastTdDataset, TensorDictDataset
from rl4co.data.generate_slot_dataset import generate_slot_datasets, main
```

### Test Inventory Breakdown

#### Test 1: CLI Argument Parsing & Runner (`test_cli_argument_parsing`)
- **Objective**: Verify that command-line options (`--output_dir`, `--num_samples`, `--graph_sizes`, `--distributions`, `--k_neighbors`, `--seed`, `--overwrite`) are parsed correctly.
- **Assertion**: Running main function with custom flags populates configuration parameters without error.

#### Test 2: File Generation & Overwrite Guard (`test_file_generation_and_overwrite`)
- **Objective**: Test dataset `.pt` file creation in designated directory and verify overwrite protection behavior.
- **Assertion**:
  - Output `.pt` files exist on disk after execution (e.g. `slot_uniform_N50_k15.pt`).
  - Running without `--overwrite` when file exists skips re-computation (matches timestamp).
  - Running with `--overwrite` re-creates and updates the file.

#### Test 3: PyTorch `torch.load()` Compatibility & Security (`test_torch_load_compatibility`)
- **Objective**: Ensure saved files load cleanly via `torch.load()` with `weights_only=True`.
- **Assertion**:
  - `torch.load(file_path, weights_only=True, map_location="cpu")` executes without raising `UnpicklingError` or warning.
  - Returned object is a `dict` with keys `"locs"`, `"depot"`, `"d_ins"`.

#### Test 4: TensorDict & `rl4co` Dataset Interoperability (`test_tensordict_dataset_loading`)
- **Objective**: Confirm loaded dataset converts seamlessly to `TensorDict` and works with `FastTdDataset` / `TensorDictDataset`.
- **Assertion**:
  - `td = TensorDict(data_dict, batch_size=num_samples)` succeeds.
  - `ds = FastTdDataset(td)` has `len(ds) == num_samples`.
  - Indexing `ds[0]` returns valid single-instance dictionary with correct key types.

#### Test 5: Tensor Shape Invariants (`test_dataset_tensor_shapes`)
- **Objective**: Validate tensor output shapes for $N \in \{50, 100, 200, 500\}$.
- **Assertion**:
  - `locs.shape == (num_samples, N, 2)`
  - `depot.shape == (num_samples, 1, 2)` or `(num_samples, 2)`
  - `d_ins.shape == (num_samples, N, N)`

#### Test 6: Distribution Properties - Uniform vs Clustered (`test_distribution_properties`)
- **Objective**: Validate values and spatial distributions for Uniform and Clustered GMM instances.
- **Assertion**:
  - All coordinates in `locs` and `depot` lie strictly within $[0.0, 1.0]$.
  - Uniform distribution has standard deviation close to $1/\sqrt{12} \approx 0.288$.
  - Clustered GMM distribution exhibits spatial clustering (higher local density variance, distinct cluster mode centers).

#### Test 7: Sparsified $d_{\text{ins}}$ Mathematical Invariants (`test_d_ins_tensor_values`)
- **Objective**: Assert mathematical properties of cached $d_{\text{ins}}$ matrix.
- **Assertion**:
  - Self-insertion cost `d_ins[:, i, i] == 0.0` for all $i \in [0, N-1]$.
  - Non-k-NN entries equal `float('inf')`.
  - For $N > k$, number of finite (non-`inf`) entries per row is exactly $k + 1$ (or $\le k + 1$).
  - Finite entries are non-negative (`d_ins[torch.isfinite(d_ins)] >= 0.0`).

#### Test 8: Seed Determinism & Reproducibility (`test_seed_reproducibility`)
- **Objective**: Verify that identical seeds produce identical dataset tensors, while different seeds produce distinct tensors.
- **Assertion**:
  - `torch.allclose(data_seed42["locs"], data_seed42_bis["locs"]) == True`
  - `torch.allclose(data_seed42["locs"], data_seed99["locs"]) == False`

#### Test 9: Edge Case - Small Graph Size $N \le k$ (`test_edge_case_n_less_than_k`)
- **Objective**: Test generation when $N=10$ and $k=15$.
- **Assertion**:
  - Execution completes without index out-of-bounds.
  - $d_{\text{ins}}$ contains zero `inf` values (dense matrix).

#### Test 10: Edge Case - Custom Output Directory Creation (`test_edge_case_nested_dir_creation`)
- **Objective**: Pass deeply nested non-existent directory path (`/tmp/nested/sub/dir`).
- **Assertion**:
  - Script automatically creates parent directories (`os.makedirs(..., exist_ok=True)`).
  - Dataset file is successfully written.

---

## 4. Summary Table of Unit Tests

| Test Name | Validated Aspect | Primary Assertion |
|---|---|---|
| `test_cli_argument_parsing` | CLI Runner | Correct parsing of `--graph_sizes`, `--k_neighbors`, `--distributions` |
| `test_file_generation_and_overwrite` | Disk I/O | File existence & skip/overwrite behavior |
| `test_torch_load_compatibility` | PyTorch I/O | Safe loading with `torch.load(..., weights_only=True)` |
| `test_tensordict_dataset_loading` | `rl4co` Integration | Compatible with `TensorDict` and `FastTdDataset` |
| `test_dataset_tensor_shapes` | Dimensionality | `locs`: `(B, N, 2)`, `d_ins`: `(B, N, N)` |
| `test_distribution_properties` | Value Ranges | Bounded in $[0,1]^2$; GMM spatial clustering signature |
| `test_d_ins_tensor_values` | Math Invariants | Self-insertion `0.0`, non-neighbors `inf`, finite values $\ge 0.0$ |
| `test_seed_reproducibility` | Seed Control | Identical seed $\to$ identical tensors |
| `test_edge_case_n_less_than_k` | Boundary Condition | $N \le k$ produces dense $d_{\text{ins}}$ without error |
| `test_edge_case_nested_dir_creation` | System Resilience | Auto-creates non-existent directory tree |

---

## 5. Verification Method
To verify the test suite:
1. Run pytest once `generate_slot_dataset.py` and `tests/test_generate_slot_dataset.py` are created:
   ```bash
   pytest tests/test_generate_slot_dataset.py -v
   ```
2. Check zero failure and 100% pass rate.
