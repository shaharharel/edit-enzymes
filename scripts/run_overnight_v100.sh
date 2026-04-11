#!/bin/bash
# V100 overnight experiment with auto-shutdown
set -e
cd ~/edit-enzymes
HOME_DIR=/home/shaharh_quris_ai

echo "=============================================="
echo "OVERNIGHT V100 EXPERIMENT SUITE"
echo "Started: $(date)"
echo "=============================================="

# KE07 experiments A-D
for EXP in A B C D; do
    echo ""
    echo "=== KE07 Experiment $EXP === $(date)"
    python3 experiments/pipeline/run_rl_optimization.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --n-rounds 50 --n-designs 5 --experiment $EXP \
        --output-dir results/rl/KE07_exp${EXP}_v100 \
        --rfdiffusion-dir $HOME_DIR/RFdiffusion \
        --proteinmpnn-dir $HOME_DIR/ProteinMPNN \
        --surrogate-dir results/surrogates \
        --validate-every 25 \
        2>&1 | tee ~/overnight_KE07_exp${EXP}.log
    echo "=== KE07 Experiment $EXP DONE === $(date)"
done

# RA95 experiments A-D
for EXP in A B C D; do
    echo ""
    echo "=== RA95 Experiment $EXP === $(date)"
    python3 experiments/pipeline/run_rl_optimization.py \
        --template data/pdb_clean/4A29.pdb \
        --constraint data/catalytic_sites/4A29.yaml \
        --n-rounds 50 --n-designs 5 --experiment $EXP \
        --output-dir results/rl/RA95_exp${EXP}_v100 \
        --rfdiffusion-dir $HOME_DIR/RFdiffusion \
        --proteinmpnn-dir $HOME_DIR/ProteinMPNN \
        --surrogate-dir results/surrogates \
        --validate-every 25 \
        2>&1 | tee ~/overnight_RA95_exp${EXP}.log
    echo "=== RA95 Experiment $EXP DONE === $(date)"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "=============================================="

# Auto-shutdown
echo "Shutting down instance..."
sudo shutdown -h now
