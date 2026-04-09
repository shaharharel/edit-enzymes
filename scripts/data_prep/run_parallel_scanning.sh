#!/bin/bash
# Run mutation scanning on ai-chem with 4 parallel groups
# Each group processes ~10 proteins sequentially (PyRosetta can't multiprocess)

set -e
source ~/rosetta_env/bin/activate
cd ~/edit-enzymes

PDBS=($(ls data/pdb_clean/*.pdb | sort))
TOTAL=${#PDBS[@]}
N_MUTS=2500
OUTDIR=cache/mutation_scanning
N_GROUPS=4

mkdir -p $OUTDIR

echo "=== Mutation Scanning ==="
echo "PDBs: $TOTAL, mutations/protein: $N_MUTS, groups: $N_GROUPS"
echo "Estimated total: ~$((TOTAL * N_MUTS)) mutations"
echo ""

# Per-protein scanning script
cat > /tmp/scan_one.py << 'PYEOF'
import sys, json, time, os, numpy as np
pdb_path = sys.argv[1]
n_muts = int(sys.argv[2])
output_dir = sys.argv[3]
pdb_id = os.path.basename(pdb_path).replace(".pdb","").upper()
output_file = os.path.join(output_dir, f"{pdb_id}_mutations.json")

if os.path.exists(output_file):
    existing = json.load(open(output_file))
    if len(existing) >= n_muts * 0.8:
        print(f"{pdb_id}: already done ({len(existing)} mutations)")
        sys.exit(0)

import pyrosetta
pyrosetta.init("-mute all -ex1 -ex2")
sfxn = pyrosetta.get_score_function(True)

try:
    pose = pyrosetta.pose_from_pdb(pdb_path)
except Exception as e:
    print(f"{pdb_id}: failed to load - {e}")
    sys.exit(1)

wt_score = sfxn(pose)
sequence = pose.sequence()
L = pose.total_residue()
all_aas = "ACDEFGHIKLMNPQRSTVWY"
np.random.seed(42 + hash(pdb_id) % 10000)

# Spread mutations across all positions, 3 per position
positions = list(range(L))
np.random.shuffle(positions)

results = []
t0 = time.time()

from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, PreventRepackingRLT, OperateOnResidueSubset
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector, ResidueIndexSelector

for pos_0 in positions:
    pos_1 = pos_0 + 1
    if pos_1 > L:
        continue
    wt_aa = pose.residue(pos_1).name1()
    if wt_aa not in all_aas:
        continue

    candidates = [aa for aa in all_aas if aa != wt_aa]
    np.random.shuffle(candidates)

    for mut_aa in candidates[:3]:
        try:
            mut_pose = pose.clone()
            aa3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
                   "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
                   "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}[mut_aa]
            pyrosetta.rosetta.protocols.simple_moves.MutateResidue(pos_1, aa3).apply(mut_pose)

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            ms = ResidueIndexSelector(str(pos_1))
            ns = NeighborhoodResidueSelector(ms, 8.0, True)
            prevent = PreventRepackingRLT()
            tf.push_back(OperateOnResidueSubset(prevent, ns, True))
            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
            packer.task_factory(tf)
            packer.apply(mut_pose)

            mut_score = sfxn(mut_pose)
            results.append({
                "pdb_id": pdb_id, "position": pos_0, "wt_aa": wt_aa,
                "mut_aa": mut_aa, "ddg": float(mut_score - wt_score),
                "seq_length": L, "wt_score": float(wt_score),
                "mut_score": float(mut_score),
            })
        except:
            pass

        if len(results) >= n_muts:
            break
    if len(results) >= n_muts:
        break

elapsed = time.time() - t0
with open(output_file, "w") as f:
    json.dump(results, f)

ddgs = [r["ddg"] for r in results]
if ddgs:
    print(f"{pdb_id}: {len(results)} mutations in {elapsed:.0f}s, ddg=[{min(ddgs):.1f},{max(ddgs):.1f}]")
else:
    print(f"{pdb_id}: 0 mutations in {elapsed:.0f}s")
PYEOF

# Launch parallel groups
PER_GROUP=$(( (TOTAL + N_GROUPS - 1) / N_GROUPS ))

for i in $(seq 0 $((N_GROUPS - 1))); do
    START=$((i * PER_GROUP))
    END=$(( (i+1) * PER_GROUP ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    if [ $START -ge $TOTAL ]; then break; fi

    echo "Group $i: PDBs $START-$((END-1))"
    (for j in $(seq $START $((END-1))); do
        python3 /tmp/scan_one.py "${PDBS[$j]}" $N_MUTS $OUTDIR
    done) > ~/scan_group_${i}.log 2>&1 &
done

echo ""
echo "All groups launched. Monitor with:"
echo "  tail -f ~/scan_group_*.log"
echo "  ls cache/mutation_scanning/*_mutations.json | wc -l"
echo ""
echo "When done, combine with:"
echo "  python3 -c \"import json,glob; all=[]; [all.extend(json.load(open(f))) for f in glob.glob('cache/mutation_scanning/*_mutations.json')]; json.dump(all,open('cache/mutation_scanning/all_mutations.json','w')); print(f'{len(all)} total mutations')\""

wait
echo "=== ALL GROUPS FINISHED ==="
