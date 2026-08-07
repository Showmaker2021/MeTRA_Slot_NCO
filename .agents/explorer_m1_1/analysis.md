# Milestone M1 Analysis: Offline Dataset Generator CLI, GMM Clustering, and `.pt` Dataset Format

## 1. Executive Summary

This report delivers the architectural and interface design for `rl4co/data/generate_slot_dataset.py` as required by Milestone M1 (Requirement R1 of the Metric-Aware Slot Abstraction NCO project). 

The goal of `generate_slot_dataset.py` is to generate, compute, and cache offline datasets containing customer locations, depot coordinates, and precomputed $k$-NN sparsified insertion cost matrices $d_{\text{ins}}$ saved in PyTorch `.pt` format. These datasets support both **Uniform** and **Clustered (GMM)** distributions across graph sizes $N \in \{50, 100, 200, 500\}$.

Key design choices established in this analysis:
1. **CLI Design**: Standard `argparse` CLI supporting `--output_dir`, `--distributions`, `--graph_sizes`, `--num_samples`, `--k_neighbors`, `--seed`, `--overwrite`, and `--device`.
2. **GMM Coordinate Generation**: PyTorch-vectorized GMM generator with 3 to 5 Gaussian clusters per sample, cluster centers sampled in $[0.2, 0.8]^2$, per-cluster standard deviations sampled from $\text{Uniform}(0.05, 0.10)$, and strict clamping to $[0.0, 1.0]^2$.
3. **`.pt` File Schema**: PyTorch dictionary containing Float32 Tensors `locs` $(B, N, 2)$, `depot` $(B, 2)$, and $k$-NN sparsified `d_ins` $(B, N, N)$ (with non-neighbors set to `inf`), loadable into `TensorDict` or PyTorch `Dataset` classes.

---

## 2. CLI Interface Design (`rl4co/data/generate_slot_dataset.py`)

### 2.1 Argument Specifications

The command-line interface uses Python's standard `argparse` module, following `rl4co` conventions established in `rl4co/data/generate_data.py`.

| Argument | Type | Default | Choices / Allowed Range | Description |
|---|---|---|---|---|
| `--output_dir` | `str` | `"data/slot"` | Any valid directory path | Target root directory where generated `.pt` dataset files will be stored. |
| `--distributions` | `str` (nargs=`+`) | `["uniform", "clustered"]` | Subsets of `{"uniform", "clustered"}` | List of data distribution types to generate. |
| `--graph_sizes` | `int` (nargs=`+`) | `[50, 100, 200, 500]` | Positive integers ($N > 1$) | List of graph sizes $N$ (number of customer nodes per instance). |
| `--num_samples` | `int` | `10000` | $N_{\text{samples}} \ge 1$ | Number of problem instances per dataset file. |
| `--k_neighbors` | `int` | `15` | $k \ge 1$ | Number of nearest neighbors to retain for $d_{\text{ins}}$ sparsification. |
| `--seed` | `int` | `1234` | Any valid 32-bit int | Base random seed for reproducibility across PyTorch and NumPy random generators. |
| `-f`, `--overwrite` | `bool` (store_true) | `False` | `True` / `False` | If set, overwrites existing dataset files instead of skipping them. |
| `--device` | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | `"cpu"`, `"cuda"` | Hardware device for fast vectorized $d_{\text{ins}}$ computation prior to saving. |

### 2.2 CLI Invocation Examples

1. **Default Run (All standard sizes & distributions)**:
   ```bash
   python rl4co/data/generate_slot_dataset.py
   ```

2. **Custom Small Benchmark Dataset**:
   ```bash
   python rl4co/data/generate_slot_dataset.py --output_dir data/slot_small --distributions uniform clustered --graph_sizes 50 100 --num_samples 1000 --k_neighbors 15 --seed 42 -f
   ```

3. **Large Scale N=500 Generation**:
   ```bash
   python rl4co/data/generate_slot_dataset.py --output_dir data/slot --distributions clustered --graph_sizes 500 --num_samples 2000 --k_neighbors 15 --device cuda
   ```

### 2.3 Input Validation & Safety Checks
- **Graph size vs. $k$-NN check**: If $k \ge N$, log a warning and clamp $k = N - 1$ (or retain full dense $d_{\text{ins}}$).
- **Directory Creation**: `os.makedirs(output_dir, exist_ok=True)` called automatically before saving.
- **Overwrite Protection**: Check `os.path.exists(file_path)` before generation; skip unless `--overwrite` or `-f` flag is set.

---

## 3. GMM Clustered Coordinate Generation Logic

### 3.1 Mathematical Specification

For a problem instance with $N$ customer nodes and $B$ batch size:

1. **Cluster Count ($M$)**:
   For each batch sample (or batch item), the number of clusters $M$ is sampled uniformly from $\{3, 4, 5\}$.

2. **Cluster Centers ($\mu_m$)**:
   Each cluster center $\mu_m \in \mathbb{R}^2$ is sampled uniformly from $[0.2, 0.8]^2$:
   $$\mu_m \sim \text{Uniform}(0.2, 0.8)^2, \quad m = 1, \dots, M$$
   Sampling in $[0.2, 0.8]^2$ prevents cluster centers from placing substantial Gaussian mass outside $[0, 1]^2$.

3. **Cluster Standard Deviations ($\sigma_m$)**:
   Each cluster $m$ has an isotropic standard deviation $\sigma_m \sim \text{Uniform}(0.05, 0.10)$.

4. **Node-to-Cluster Assignments**:
   For $N$ customer nodes, nodes are assigned to the $M$ clusters via multinomial sampling with equal mixture probabilities $p_m = 1/M$, or via balanced partitioning:
   $$\text{count}_m = \lfloor N / M \rfloor + [m \le N \bmod M]$$

5. **Customer Coordinate Generation**:
   For node $i$ assigned to cluster $m$:
   $$x_i \sim \mathcal{N}(\mu_m, \sigma_m^2 I_2)$$

6. **Depot Coordinate Generation**:
   Depot coordinate $x_{\text{depot}}$ is sampled uniformly from $[0.0, 1.0]^2$:
   $$x_{\text{depot}} \sim \text{Uniform}(0, 1)^2$$

7. **Bounding & Normalization**:
   All customer coordinates are clamped to $[0.0, 1.0]^2$:
   $$x_i \leftarrow \text{clamp}(x_i, 0.0, 1.0)$$

### 3.2 Uniform Coordinate Generation Logic
For uniform distribution:
- Customer locations $x_i \sim \text{Uniform}(0, 1)^2$ for $i = 1, \dots, N$.
- Depot location $x_{\text{depot}} \sim \text{Uniform}(0, 1)^2$.

---

## 4. PyTorch `.pt` Dataset File Format & Sparsified $d_{\text{ins}}$ Integration

### 4.1 Data Schema

The dataset file saved via `torch.save(data_dict, file_path)` follows a standard PyTorch dictionary layout:

```python
data_dict = {
    "locs": torch.FloatTensor,    # Shape: (num_samples, N, 2), range [0, 1]
    "depot": torch.FloatTensor,   # Shape: (num_samples, 2), range [0, 1]
    "d_ins": torch.FloatTensor,   # Shape: (num_samples, N, N), non-KNN values are float('inf')
}
```

### 4.2 Integration with `rl4co/data/insertion_cost.py`

The $d_{\text{ins}}$ tensor is computed directly using `compute_marginal_insertion_cost`:

```python
from rl4co.data.insertion_cost import compute_marginal_insertion_cost

# locs: (B, N, 2), depot: (B, 2)
d_ins = compute_marginal_insertion_cost(
    locs=locs,
    k_neighbors=k_neighbors,
    depot_loc=depot,
)
```

`compute_marginal_insertion_cost` automatically:
1. Calculates customer-to-customer distances $d(i, j)$ and depot-to-customer distances $d(D, i)$.
2. Computes marginal insertion cost $d_{\text{ins}}(i, j) = d(D, i) + d(i, j) - d(D, j)$.
3. Clamps non-negative underflow to `0.0` and sets diagonal self-insertion $d_{\text{ins}}(i, i) = 0.0$.
4. Masks non-$k$-NN neighbors to `float('inf')` using `torch.topk(dist_customers, k=k_neighbors+1)`.

### 4.3 Storage & File Naming Conventions

Dataset file naming follows a deterministic format:
`{distribution}_n{graph_size}_k{k_neighbors}_seed{seed}.pt`

Example directory structure:
```
data/slot/
├── uniform_n50_k15_seed1234.pt
├── uniform_n100_k15_seed1234.pt
├── uniform_n200_k15_seed1234.pt
├── uniform_n500_k15_seed1234.pt
├── clustered_n50_k15_seed1234.pt
├── clustered_n100_k15_seed1234.pt
├── clustered_n200_k15_seed1234.pt
└── clustered_n500_k15_seed1234.pt
```

### 4.4 Estimated Dataset File Sizes (Float32)

| $N$ | Samples | `locs` Size | `depot` Size | `d_ins` Size | Total File Size |
|---|---|---|---|---|---|
| 50 | 10,000 | ~400 KB | ~80 KB | ~100 MB | **~100.5 MB** |
| 100 | 10,000 | ~800 KB | ~80 KB | ~400 MB | **~400.9 MB** |
| 200 | 10,000 | ~1.6 MB | ~80 KB | ~1.6 GB | **~1.60 GB** |
| 500 | 10,000 | ~4.0 MB | ~80 KB | ~10.0 GB | **~10.00 GB** |

*(Note: For large $N=500$, `num_samples` can be batched during generation to prevent CPU/GPU memory spike during `d_ins` tensor creation).*

### 4.5 Downstream Loading Compatibility

Dataset files saved with `torch.save` can be loaded cleanly and wrapped in RL4CO `TensorDictDataset` or `FastTdDataset`:

```python
import torch
from tensordict import TensorDict
from rl4co.data.dataset import FastTdDataset

loaded_data = torch.load("data/slot/clustered_n50_k15_seed1234.pt")
td = TensorDict(loaded_data, batch_size=loaded_data["locs"].shape[0])
dataset = FastTdDataset(td)
```

---

## 5. Implementation Code Structure Sketch

Below is the proposed Python implementation structure for `rl4co/data/generate_slot_dataset.py`:

```python
import argparse
import os
import torch
import numpy as np
from typing import Dict, List, Optional
from rl4co.data.insertion_cost import compute_marginal_insertion_cost
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def generate_gmm_clusters(
    batch_size: int,
    num_loc: int,
    min_clusters: int = 3,
    max_clusters: int = 5,
    min_std: float = 0.05,
    max_std: float = 0.10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate GMM clustered coordinates in [0, 1]^2."""
    coords = torch.zeros(batch_size, num_loc, 2, device=device)
    
    for b in range(batch_size):
        num_clusters = torch.randint(min_clusters, max_clusters + 1, (1,)).item()
        centers = 0.2 + 0.6 * torch.rand(num_clusters, 2, device=device)
        stds = min_std + (max_std - min_std) * torch.rand(num_clusters, device=device)
        
        # Partition num_loc nodes among clusters
        cluster_sizes = [num_loc // num_clusters] * num_clusters
        for i in range(num_loc % num_clusters):
            cluster_sizes[i] += 1
            
        cur_idx = 0
        for k in range(num_clusters):
            sz = cluster_sizes[k]
            pts = torch.normal(
                mean=centers[k].unsqueeze(0).expand(sz, 2),
                std=stds[k],
            )
            coords[b, cur_idx : cur_idx + sz] = pts
            cur_idx += sz
            
    return torch.clamp(coords, 0.0, 1.0)


def generate_slot_dataset(
    distribution: str,
    num_samples: int,
    graph_size: int,
    k_neighbors: int = 15,
    seed: int = 1234,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Generate locs, depot, and precomputed d_ins for a single dataset configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate locs
    if distribution == "uniform":
        locs = torch.rand(num_samples, graph_size, 2, device=device)
    elif distribution == "clustered":
        locs = generate_gmm_clusters(
            batch_size=num_samples,
            num_loc=graph_size,
            device=device,
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    # 2. Generate depot
    depot = torch.rand(num_samples, 2, device=device)
    
    # 3. Compute sparsified d_ins in sub-batches if necessary for large N
    batch_size_compute = 1000 if graph_size >= 200 else num_samples
    d_ins_list = []
    
    for i in range(0, num_samples, batch_size_compute):
        locs_b = locs[i : i + batch_size_compute]
        depot_b = depot[i : i + batch_size_compute]
        d_ins_b = compute_marginal_insertion_cost(
            locs=locs_b,
            k_neighbors=k_neighbors,
            depot_loc=depot_b,
        )
        d_ins_list.append(d_ins_b.cpu())
        
    d_ins = torch.cat(d_ins_list, dim=0)
    
    return {
        "locs": locs.cpu(),
        "depot": depot.cpu(),
        "d_ins": d_ins,
    }
```
