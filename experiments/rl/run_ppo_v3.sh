#!/bin/bash
# PPO v3: Fixed host orchestrator
# - Docker: generate + PPO training (separate calls)
# - Host: Rosetta scoring
# - Checkpoint persists via ~/ppo_ckpt mounted as /ckpt
set -e

N_ROUNDS=${1:-30}
OUTPUT_DIR=~/rfd3_ppo_v3
CKPT_DIR=~/ppo_ckpt
SCRIPT_DIR=~/edit-enzymes/experiments/rl

mkdir -p $OUTPUT_DIR $CKPT_DIR

# Clear old checkpoint for fresh start
rm -f $CKPT_DIR/model.pt

echo "=============================================="
echo "PPO v3: FIXED WEIGHT PERSISTENCE"
echo "Rounds: $N_ROUNDS"
echo "Started: $(date)"
echo "=============================================="

for ROUND in $(seq 0 $((N_ROUNDS-1))); do
    ROUND_DIR=$OUTPUT_DIR/round_$(printf '%04d' $ROUND)
    mkdir -p $ROUND_DIR

    echo ""
    echo "--- Round $((ROUND+1))/$N_ROUNDS --- $(date)"

    # 1. GENERATE (Docker, uses persisted checkpoint for generation)
    T0=$(date +%s)
    sudo docker run --rm --gpus all \
        -v $ROUND_DIR:/output \
        -v $CKPT_DIR:/ckpt \
        rosettacommons/foundry:latest \
        bash -c "cd /app/foundry/models/rfd3/docs && rfd3 design out_dir=/output inputs=enzyme_design.json" \
        2>&1 | grep "Finished" || true
    T1=$(date +%s)
    N_DESIGNS=$(ls $ROUND_DIR/*.cif.gz 2>/dev/null | wc -l)
    echo "  Generated: $N_DESIGNS designs ($((T1-T0))s)"

    if [ "$N_DESIGNS" -eq 0 ]; then continue; fi

    # 2. ROSETTA SCORE (host)
    T2=$(date +%s)
    python3 -c "
import sys, json, glob, gzip, tempfile, os, numpy as np
import pyrosetta
import biotite.structure.io.pdbx as pdbx
pyrosetta.init('-mute all -ex1 -ex2')
sfxn = pyrosetta.get_score_function(True)

cifs = sorted(glob.glob('$ROUND_DIR/*.cif.gz'))
scores = []
for cif_path in cifs:
    try:
        with gzip.open(cif_path, 'rt') as f:
            cif = pdbx.CIFFile.read(f)
        aa = pdbx.get_structure(list(cif.values())[0], model=1)
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
            pdb_path = f.name
            for i in range(len(aa)):
                if aa.element[i] == '': continue
                f.write(f'ATOM  {i+1:5d}  {aa.atom_name[i]:<3s} {aa.res_name[i]:3s} {aa.chain_id[i]:1s}{aa.res_id[i]:4d}    {aa.coord[i,0]:8.3f}{aa.coord[i,1]:8.3f}{aa.coord[i,2]:8.3f}  1.00  0.00           {aa.element[i]:>2s}\n')
            f.write('END\n')
        with open(pdb_path) as f:
            lines = [l for l in f if l.startswith('ATOM') or l.startswith('END')]
        clean = pdb_path+'.c'
        with open(clean,'w') as f: f.writelines(lines)
        pose = pyrosetta.pose_from_pdb(clean)
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        tf=TaskFactory(); tf.push_back(RestrictToRepacking())
        pk=pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
        pk.task_factory(tf); pk.apply(pose)
        scores.append(float(sfxn(pose)))
        os.unlink(pdb_path); os.unlink(clean)
    except:
        scores.append(9999.0)

json.dump(scores, open('$ROUND_DIR/rosetta_scores.json','w'))
valid = [s for s in scores if s < 9000]
if valid:
    print(f'Rosetta: best={min(valid):.0f}, mean={np.mean(valid):.0f}, n={len(valid)}')
else:
    print('No valid scores')
"
    T3=$(date +%s)
    echo "  Scored: $((T3-T2))s"

    # 3. COMPUTE ADVANTAGES (host)
    CIF_ADV=$(python3 -c "
import json, numpy as np, glob
scores = json.load(open('$ROUND_DIR/rosetta_scores.json'))
cifs = sorted(glob.glob('$ROUND_DIR/*.cif.gz'))
# Filter valid
valid = [(c, s) for c, s in zip(cifs, scores) if s < 9000]
if not valid:
    print('[]')
else:
    rewards = [-s for _, s in valid]
    mean_r = np.mean(rewards)
    std_r = max(np.std(rewards), 1.0)
    advantages = [(-s - mean_r) / std_r for _, s in valid]
    # Map to Docker paths
    result = [['/input/' + c.split('/')[-1], a] for (c, _), a in zip(valid, advantages)]
    print(json.dumps(result))
")

    if [ "$CIF_ADV" == "[]" ]; then
        echo "  No valid designs for training"
        continue
    fi

    # 4. PPO UPDATE (Docker, with checkpoint persistence)
    T4=$(date +%s)
    sudo docker run --rm --gpus all \
        -v $ROUND_DIR:/input \
        -v $CKPT_DIR:/ckpt \
        -v $SCRIPT_DIR/ppo_train_docker.py:/workspace/train.py \
        rosettacommons/foundry:latest \
        python3 /workspace/train.py \
        --cif-advantages "$CIF_ADV" \
        --ppo-epochs 3 \
        --clip-epsilon 0.2 \
        --lr 1e-5 \
        2>&1 | grep "PPO_RESULT\|Loaded\|saved" || true
    T5=$(date +%s)
    echo "  PPO: $((T5-T4))s"

    # Track results
    echo "$ROUND" >> $OUTPUT_DIR/completed_rounds.txt
done

echo ""
echo "=============================================="
echo "PPO v3 COMPLETE: $(date)"
echo "=============================================="

# Summarize
python3 -c "
import json, glob, numpy as np
all_scores = []
for d in sorted(glob.glob('$OUTPUT_DIR/round_*/rosetta_scores.json')):
    scores = json.load(open(d))
    valid = [s for s in scores if s < 9000]
    if valid:
        all_scores.append(min(valid))

if len(all_scores) >= 6:
    f3 = np.mean(all_scores[:3])
    l3 = np.mean(all_scores[-3:])
    print(f'Rosetta trend: first3={f3:.0f} -> last3={l3:.0f} (delta={l3-f3:+.0f})')
    print(f'Best: {min(all_scores):.0f}')
json.dump(all_scores, open('$OUTPUT_DIR/rosetta_history.json', 'w'))
"
