#!/usr/bin/env bash
# ============================================================
# Full Pipeline: Generate Data → M8 Gate → Full Training
# ============================================================
# Step 1: Generate datasets (run once — takes ~30 min for N=500)
#   conda run -n ec_nco python -m rl4co.data.generate_slot_dataset \
#       --num_locs 50  --dist both --n_train 100000 --n_val 1000 --out_dir ./data/slot_datasets
#   conda run -n ec_nco python -m rl4co.data.generate_slot_dataset \
#       --num_locs 100 --dist both --n_train 100000 --n_val 1000 --out_dir ./data/slot_datasets
#   conda run -n ec_nco python -m rl4co.data.generate_slot_dataset \
#       --num_locs 200 --dist both --n_train 100000 --n_val 1000 --out_dir ./data/slot_datasets
#   conda run -n ec_nco python -m rl4co.data.generate_slot_dataset \
#       --num_locs 500 --dist both --n_train 50000  --n_val 500  --out_dir ./data/slot_datasets

# Step 2: M8 Decision Gate (fast, ~10 min, no GPU needed)
#   conda run -n ec_nco python scripts/m8_decision_gate.py

# Step 3: Full ablation at N=50 (3 seeds, ~2-4h on 1 GPU)
#   conda run -n ec_nco python scripts/run_ablation.py --num_loc 50 --n_seeds 3

# Step 4: Full scale N=100/200 (main results, ~24-48h on 1-4 GPUs)
#   conda run -n ec_nco python scripts/run_ablation.py --num_loc 100 --n_seeds 5 --devices 1
#   conda run -n ec_nco python scripts/run_ablation.py --num_loc 200 --n_seeds 5 --devices 1

# Step 5: Single variant training (quick test)
#   conda run -n ec_nco python scripts/train.py --variant D --num_loc 100 --dist uniform --seed 42
echo "See comments above for full pipeline commands."
