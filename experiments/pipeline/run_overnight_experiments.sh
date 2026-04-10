#!/bin/bash
# 12-hour overnight experiment suite on GPU
# Runs all 4 progressive unfreezing experiments (A→D) on KE07 + RA95
#
# Experiment A: Frozen generators — search params only (100 rounds)
# Experiment B: Unfreeze ProteinMPNN — sequence optimization (100 rounds)
# Experiment C: Unfreeze RFdiffusion — backbone optimization (100 rounds)
# Experiment D: Unfreeze both — full end-to-end (100 rounds)
#
# Each experiment: 100 rounds × 5 designs × 4 seqs = 2000 designs
# With surrogate scoring: ~5 min/experiment
# With Rosetta validation every 20 rounds: ~30 min/experiment
#
# Total: ~4 hours for all 8 experiments (2 enzymes × 4 experiments)
# Remaining 8 hours: extended runs with 500 rounds on best config

set -e

HOME_DIR=/home/shaharh_quris_ai
cd ~/edit-enzymes

echo "=============================================="
echo "OVERNIGHT EXPERIMENT SUITE"
echo "Started: $(date)"
echo "=============================================="

# === Phase 1: Progressive Unfreezing on KE07 (Kemp eliminase) ===
for EXP in A B C D; do
    echo ""
    echo "=== KE07 Experiment $EXP ==="
    echo "Started: $(date)"

    python3 experiments/pipeline/run_rl_optimization.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --n-rounds 100 \
        --n-designs 5 \
        --experiment $EXP \
        --output-dir results/rl/KE07_exp${EXP}_100r \
        --rfdiffusion-dir $HOME_DIR/RFdiffusion \
        --proteinmpnn-dir $HOME_DIR/ProteinMPNN \
        --surrogate-dir results/surrogates \
        --validate-every 20 \
        2>&1 | tee ~/overnight_KE07_exp${EXP}.log

    echo "Finished KE07 Experiment $EXP: $(date)"
done

# === Phase 2: Progressive Unfreezing on RA95 (Retro-aldolase) ===
for EXP in A B C D; do
    echo ""
    echo "=== RA95 Experiment $EXP ==="
    echo "Started: $(date)"

    python3 experiments/pipeline/run_rl_optimization.py \
        --template data/pdb_clean/4A29.pdb \
        --constraint data/catalytic_sites/4A29.yaml \
        --n-rounds 100 \
        --n-designs 5 \
        --experiment $EXP \
        --output-dir results/rl/RA95_exp${EXP}_100r \
        --rfdiffusion-dir $HOME_DIR/RFdiffusion \
        --proteinmpnn-dir $HOME_DIR/ProteinMPNN \
        --surrogate-dir results/surrogates \
        --validate-every 20 \
        2>&1 | tee ~/overnight_RA95_exp${EXP}.log

    echo "Finished RA95 Experiment $EXP: $(date)"
done

# === Phase 3: Extended run on best config (500 rounds) ===
echo ""
echo "=== EXTENDED: KE07 Best Config (500 rounds) ==="
echo "Started: $(date)"

python3 experiments/pipeline/run_rl_optimization.py \
    --template data/pdb_clean/2RKX.pdb \
    --constraint data/catalytic_sites/2RKX.yaml \
    --n-rounds 500 \
    --n-designs 5 \
    --experiment A \
    --output-dir results/rl/KE07_expA_500r \
    --rfdiffusion-dir $HOME_DIR/RFdiffusion \
    --proteinmpnn-dir $HOME_DIR/ProteinMPNN \
    --surrogate-dir results/surrogates \
    --validate-every 50 \
    2>&1 | tee ~/overnight_KE07_extended.log

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "=============================================="

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in results/rl/*/reward_history.json; do
    dir=$(dirname $f)
    name=$(basename $dir)
    n_rounds=$(python3 -c "import json; print(len(json.load(open('$f'))))")
    best=$(python3 -c "import json; d=json.load(open('${dir}/best_design.json')); print(f\"reward={d['reward']:.3f}, RMSD={d.get('template_rmsd','?')}\")")
    echo "  $name: $n_rounds rounds, $best"
done
