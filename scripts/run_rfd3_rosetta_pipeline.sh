#!/bin/bash
# RFD3 generate + Rosetta score pipeline
# Run on ai-gpu2 (V100)
set -e

cd ~/edit-enzymes
OUTPUT=results/rfd3_pipeline/full_pipeline
mkdir -p $OUTPUT

echo "=== RFD3 + ROSETTA PIPELINE ==="
echo "10 rounds × 8 designs = 80 total"
echo "Started: $(date)"

# Step 1: Generate all designs
for i in $(seq 0 9); do
    echo ""
    echo "--- Generating Round $i ---"
    sudo docker run --rm --gpus all \
        -v $HOME/edit-enzymes/$OUTPUT:/output \
        rosettacommons/foundry:latest \
        bash -c "cd /app/foundry/models/rfd3/docs && rfd3 design out_dir=/output/round_${i} inputs=enzyme_design.json" 2>&1 | grep -E "Finished|Error|seconds"
done

echo ""
echo "=== GENERATION COMPLETE: $(date) ==="
echo ""

# Step 2: Score all with Rosetta
echo "=== SCORING WITH ROSETTA ==="
python3 experiments/rl/run_rfd3_ppo.py \
    --template data/pdb_clean/2RKX.pdb \
    --constraint data/catalytic_sites/2RKX.yaml \
    --n-rounds 0 \
    --output-dir /dev/null 2>/dev/null &
# Just to init pyrosetta, kill immediately
sleep 2 && kill %1 2>/dev/null

python3 -c "
import sys, json, glob, numpy as np
sys.path.insert(0, '.')
from experiments.rl.run_rfd3_ppo import score_design, init_pyrosetta
init_pyrosetta()

all_rosetta = []
for rnd in range(10):
    cifs = sorted(glob.glob(f'$OUTPUT/round_{rnd}/*.cif.gz'))
    jsons = sorted(glob.glob(f'$OUTPUT/round_{rnd}/*.json'))
    round_scores = []
    for c, j in zip(cifs, jsons):
        s = score_design(c, j)
        r = s.get('rosetta_total', None)
        if r is not None and r == r:  # not NaN
            round_scores.append(r)
            all_rosetta.append(r)
    if round_scores:
        print(f'Round {rnd}: best={min(round_scores):.1f}, mean={np.mean(round_scores):.1f}, n={len(round_scores)}')

if all_rosetta:
    print()
    print(f'Overall: best={min(all_rosetta):.1f}, mean={np.mean(all_rosetta):.1f}, n={len(all_rosetta)}')
    json.dump(all_rosetta, open('$OUTPUT/all_rosetta_scores.json', 'w'))
"

echo ""
echo "=== PIPELINE COMPLETE: $(date) ==="
