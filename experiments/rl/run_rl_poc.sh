#!/bin/bash
# RL POC: RFD3 (Docker) + Rosetta (host) + training (Docker)
# Run on ai-gpu2 host
set -e

N_ROUNDS=30
OUTPUT_DIR=~/rfd3_rl_final
mkdir -p $OUTPUT_DIR

echo "=== RL POC: RFD3 + Rosetta ==="
echo "Rounds: $N_ROUNDS"
echo "Started: $(date)"

# Score designs with Rosetta on HOST
score_round() {
    local ROUND_DIR=$1
    python3 -c "
import sys, json, glob, gzip, tempfile, os, numpy as np
import pyrosetta
pyrosetta.init('-mute all -ex1 -ex2')
sfxn = pyrosetta.get_score_function(True)
import biotite.structure.io.pdbx as pdbx

cifs = sorted(glob.glob('${ROUND_DIR}/*.cif.gz'))
results = []
for cif_path in cifs:
    try:
        with gzip.open(cif_path, 'rt') as f:
            cif = pdbx.CIFFile.read(f)
        block = list(cif.values())[0]
        aa = pdbx.get_structure(block, model=1)
        # Write PDB
        tmp = tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False)
        for i in range(len(aa)):
            if aa.element[i] == '': continue
            tmp.write(f'ATOM  {i+1:5d}  {aa.atom_name[i]:<3s} {aa.res_name[i]:3s} {aa.chain_id[i]:1s}{aa.res_id[i]:4d}    {aa.coord[i,0]:8.3f}{aa.coord[i,1]:8.3f}{aa.coord[i,2]:8.3f}  1.00  0.00           {aa.element[i]:>2s}\n')
        tmp.write('END\n'); tmp.close()
        # Clean for Rosetta
        with open(tmp.name) as f:
            lines = [l for l in f if l.startswith('ATOM') or l.startswith('END')]
        clean = tmp.name + '.clean.pdb'
        with open(clean, 'w') as f: f.writelines(lines)
        pose = pyrosetta.pose_from_pdb(clean)
        # Repack
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        tf = TaskFactory(); tf.push_back(RestrictToRepacking())
        pk = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
        pk.task_factory(tf); pk.apply(pose)
        score = sfxn(pose)
        results.append({'cif': cif_path, 'rosetta': float(score), 'n_res': pose.total_residue()})
        os.unlink(tmp.name); os.unlink(clean)
    except Exception as e:
        results.append({'cif': cif_path, 'rosetta': 9999.0, 'error': str(e)})

json.dump(results, open('${ROUND_DIR}/rosetta_scores.json', 'w'), indent=2)
scores = [r['rosetta'] for r in results if r['rosetta'] < 9000]
if scores:
    print(f'Rosetta: best={min(scores):.0f}, mean={np.mean(scores):.0f}, n={len(scores)}')
else:
    print('No valid Rosetta scores')
"
}

# Generate designs with RFD3 in Docker
generate_round() {
    local ROUND=$1
    local ROUND_DIR=$OUTPUT_DIR/round_$(printf '%04d' $ROUND)
    mkdir -p $ROUND_DIR

    sudo docker run --rm --gpus all \
        -v $ROUND_DIR:/output \
        rosettacommons/foundry:latest \
        bash -c "cd /app/foundry/models/rfd3/docs && rfd3 design out_dir=/output inputs=enzyme_design.json" \
        2>&1 | grep -E "Finished|Error" | head -3

    echo "Generated: $(ls $ROUND_DIR/*.cif.gz 2>/dev/null | wc -l) designs"
}

# Train on best designs in Docker
train_on_best() {
    local ROUND=$1
    local ROUND_DIR=$OUTPUT_DIR/round_$(printf '%04d' $ROUND)

    # Get best 4 CIFs by Rosetta score
    BEST_CIFS=$(python3 -c "
import json
scores = json.load(open('${ROUND_DIR}/rosetta_scores.json'))
valid = [s for s in scores if s['rosetta'] < 9000]
valid.sort(key=lambda x: x['rosetta'])
for s in valid[:4]:
    print(s['cif'].replace('${ROUND_DIR}', '/input'))
")

    if [ -z "$BEST_CIFS" ]; then
        echo "No valid designs to train on"
        return
    fi

    # Write training script for this round
    cat > /tmp/train_round.py << PYEOF
import torch, sys, os, numpy as np
sys.stdout = open('/output/train_log.txt', 'w')
from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

conf = RFD3InferenceConfig(ckpt_path="/root/.foundry/checkpoints/rfd3_latest.ckpt",
    diffusion_batch_size=1, inference_sampler={"num_timesteps": 3})
engine = RFD3InferenceEngine(**conf)
engine._set_out_dir("/tmp/train")
engine.initialize()

class L(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DiffusionLoss(weight=4.0,sigma_data=16.0,lddt_weight=0.25,alpha_virtual_atom=1.0,
            alpha_polar_residues=1.0,alpha_ligand=10.0,lp_weight=0.0,unindexed_norm_p=1.0,
            alpha_unindexed_diffused=1.0,unindexed_t_alpha=0.75)
        self.s = SequenceLoss(weight=0.1,max_t=1)
    def forward(self,**kw):
        dl,dd=self.d(**kw); sl,sd=self.s(**kw); return dl+sl,{**dd,**sd}

engine.trainer.loss = L().cuda()
model = engine.trainer.state["model"]
model.train()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)

cif_paths = """$BEST_CIFS""".strip().split('\n')
total_loss = 0
n = 0
for cif in cif_paths:
    cif = cif.strip()
    if not cif or not os.path.exists(cif): continue
    try:
        result = parse(cif)
        aa = result["asym_unit"][0]
        initialize_chain_info_from_atom_array(aa)
        aa.set_annotation("chain_iid", aa.chain_id.copy())
        if "pn_unit_iid" not in aa.get_annotation_categories():
            aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
        for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
            if ann not in aa.get_annotation_categories():
                aa.set_annotation(ann, np.zeros(len(aa), dtype=bool))
        pr = engine.pipeline({"atom_array": aa, "example_id": f"best_{n}"})
        pr = engine.trainer.fabric.to_device(pr)
        for step in range(2):
            opt.zero_grad()
            engine.trainer.training_step(batch=[pr], batch_idx=step, is_accumulating=False)
            ret = engine.trainer._current_train_return
            if ret and "total_loss" in ret: total_loss += ret["total_loss"].item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            n += 1
    except Exception as e:
        print(f"Train failed on {cif}: {e}")

print(f"Trained on {n} steps, avg_loss={total_loss/max(n,1):.4f}")
PYEOF

    sudo docker run --rm --gpus all \
        -v $ROUND_DIR:/input \
        -v /tmp:/workspace \
        rosettacommons/foundry:latest \
        python3 /workspace/train_round.py 2>&1 | grep -v "WARNING\|Environment\|set it\|atomworks\|write\|Cached\|DEBUG\|conformer\|charges\|cuEquiv"

    cat $ROUND_DIR/train_log.txt 2>/dev/null
}

# Main loop
REWARD_HISTORY=""
ROSETTA_HISTORY=""

for ROUND in $(seq 0 $((N_ROUNDS-1))); do
    echo ""
    echo "=== ROUND $((ROUND+1))/$N_ROUNDS === $(date)"

    # 1. Generate
    generate_round $ROUND

    # 2. Score with Rosetta
    ROUND_DIR=$OUTPUT_DIR/round_$(printf '%04d' $ROUND)
    score_round $ROUND_DIR

    # 3. Train on best
    train_on_best $ROUND

    echo "--- Round $((ROUND+1)) complete ---"
done

echo ""
echo "=== ALL COMPLETE === $(date)"
echo "Results in $OUTPUT_DIR"
