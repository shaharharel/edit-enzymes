"""PPO fine-tuning of RFdiffusion3 for enzyme design.

Clean pipeline:
1. RFD3 generates all-atom enzyme design (backbone + sidechains + sequence)
2. Rosetta scores the design (apply sequence, repack, score)
3. Score → reward → PPO updates RFD3 weights

RFD3 runs inside Docker (Foundry image). For PPO we need to:
- Generate designs (Docker call)
- Score with Rosetta (PyRosetta)
- Track which designs are better/worse
- Use evolutionary selection + parameter tuning as the RL signal

For true gradient-based PPO on RFD3 weights, we'd need to load the model
natively (not via Docker). The Foundry training code supports this.
Phase 1 (this script): reward-guided search (evolutionary + scoring)
Phase 2 (future): native model loading + gradient PPO

Usage:
    python experiments/rl/run_rfd3_ppo.py \
        --theozyme data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --n-rounds 20 --designs-per-round 8
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import subprocess
import tempfile
import gzip
import numpy as np
from typing import Dict, List, Optional

from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PYROSETTA_INIT = False
_SFXN = None


def init_pyrosetta():
    global _PYROSETTA_INIT, _SFXN
    if not _PYROSETTA_INIT:
        import pyrosetta
        pyrosetta.init('-mute all -ex1 -ex2')
        _SFXN = pyrosetta.get_score_function(True)
        _PYROSETTA_INIT = True


def generate_designs_rfd3(
    input_json: str,
    output_dir: str,
    n_designs: int = 8,
) -> List[str]:
    """Generate enzyme designs using RFD3 via Docker."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(input_json).parent.resolve()

    # RFD3 uses num_designs from the config, default is 8
    cmd = [
        'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{input_dir}:/input',
        '-v', f'{output_dir}:/output',
        'rosettacommons/foundry:latest',
        'rfd3', 'design',
        f'out_dir=/output',
        f'inputs=/input/{Path(input_json).name}',
    ]

    logger.info(f"Running RFD3: {n_designs} designs")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"RFD3 failed: {result.stderr[-300:]}")
        return []

    # Collect output CIF files
    cif_files = sorted(output_dir.glob('*.cif.gz'))
    logger.info(f"RFD3 generated {len(cif_files)} designs in {elapsed:.0f}s")
    return [str(f) for f in cif_files]


def cif_to_pdb(cif_gz_path: str, output_pdb: str) -> Optional[str]:
    """Convert CIF.gz to PDB for Rosetta scoring."""
    try:
        import gzip
        # Read CIF
        with gzip.open(cif_gz_path, 'rt') as f:
            cif_content = f.read()

        # Parse ATOM records from CIF format
        lines = []
        atom_num = 1
        for line in cif_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.split()
                if len(parts) >= 21:
                    group = parts[0]
                    atom_name = parts[2]
                    res_name = parts[4]
                    chain = parts[5]
                    seq_id = parts[7]
                    x = float(parts[18])
                    y = float(parts[19])
                    z = float(parts[20])

                    pdb_line = (
                        f"{group:<6s}{atom_num:5d}  {atom_name:<3s} {res_name:3s} "
                        f"{chain:1s}{int(seq_id):4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                    )
                    lines.append(pdb_line)
                    atom_num += 1

        lines.append("END")
        with open(output_pdb, 'w') as f:
            f.write('\n'.join(lines))
        return output_pdb

    except Exception as e:
        logger.warning(f"CIF→PDB conversion failed: {e}")
        return None


def score_design(cif_gz_path: str, json_path: str) -> Dict[str, float]:
    """Score an RFD3 design with Rosetta + extract RFD3 metrics."""
    scores = {}

    # 1. RFD3's own metrics
    try:
        with open(json_path) as f:
            metrics = json.load(f)
        rfd3_metrics = metrics.get('metrics', {})
        scores['rfd3_n_clashes'] = rfd3_metrics.get('n_clashing.interresidue_clashes_w_sidechain', 0)
        scores['rfd3_n_chainbreaks'] = rfd3_metrics.get('n_chainbreaks', 0)
        scores['rfd3_helix_fraction'] = rfd3_metrics.get('helix_fraction', 0)
        scores['rfd3_sheet_fraction'] = rfd3_metrics.get('sheet_fraction', 0)
        scores['rfd3_loop_fraction'] = rfd3_metrics.get('loop_fraction', 0)
        scores['rfd3_rog'] = rfd3_metrics.get('radius_of_gyration', 0)
        scores['rfd3_max_ca_deviation'] = rfd3_metrics.get('max_ca_deviation', 0)
        scores['rfd3_ligand_min_dist'] = rfd3_metrics.get('n_clashing.ligand_min_distance', 0)
    except Exception as e:
        logger.warning(f"Failed to read RFD3 metrics: {e}")

    # 2. Rosetta scoring
    try:
        init_pyrosetta()
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreType

        # Convert CIF to PDB for Rosetta
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
            pdb_path = tmp.name

        converted = cif_to_pdb(cif_gz_path, pdb_path)
        if not converted:
            return scores

        # Load and score
        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
        except Exception:
            # Try stripping non-standard residues
            with open(pdb_path) as f:
                clean_lines = [l for l in f.readlines()
                              if l.startswith('ATOM') or l.startswith('END')]
            clean_path = pdb_path + '.clean.pdb'
            with open(clean_path, 'w') as f:
                f.writelines(clean_lines)
            pose = pyrosetta.pose_from_pdb(clean_path)

        # Repack sidechains (RFD3 places them but Rosetta might prefer different rotamers)
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(_SFXN)
        packer.task_factory(tf)
        packer.apply(pose)

        total = _SFXN(pose)
        scores['rosetta_total'] = float(total)
        scores['rosetta_per_residue'] = float(total / max(pose.total_residue(), 1))

        # Key energy terms
        for term_name in ['fa_atr', 'fa_rep', 'fa_sol', 'fa_elec',
                          'hbond_sr_bb', 'hbond_sc', 'fa_dun', 'rama_prepro']:
            try:
                st = getattr(ScoreType, term_name)
                scores[f'rosetta_{term_name}'] = float(pose.energies().total_energies()[st])
            except:
                pass

        scores['n_residues'] = pose.total_residue()
        scores['sequence'] = pose.sequence()

    except Exception as e:
        logger.warning(f"Rosetta scoring failed: {e}")

    return scores


def create_theozyme_json(
    template_pdb: str,
    constraint_yaml: str,
    length_range: str = "180-250",
    output_path: str = "/tmp/theozyme_input.json",
) -> str:
    """Create RFD3 input JSON from our constraint format."""
    template = load_pdb(template_pdb)
    constraint = load_constraint_from_yaml(constraint_yaml)

    # Build select_fixed_atoms from catalytic residues
    fixed_atoms = {}
    unindex_residues = []

    for res in constraint.residues:
        if res.position_index is not None:
            chain_pos = f"A{res.position_index + 1}"
            unindex_residues.append(chain_pos)

            # Fix functional atoms
            from src.utils.protein_constants import FUNCTIONAL_ATOMS
            func_atoms = FUNCTIONAL_ATOMS.get(res.residue_type, [])
            if func_atoms:
                fixed_atoms[chain_pos] = ','.join(func_atoms)
            else:
                fixed_atoms[chain_pos] = 'ALL'

    config = {
        "design_0": {
            "input": f"/input/{Path(template_pdb).name}",
            "unindex": ','.join(unindex_residues) if unindex_residues else "",
            "length": length_range,
            "select_fixed_atoms": fixed_atoms,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created theozyme JSON: {len(unindex_residues)} catalytic residues, length={length_range}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='RFD3 + Rosetta pipeline')
    parser.add_argument('--template', required=True, help='Template PDB')
    parser.add_argument('--constraint', required=True, help='Catalytic constraint YAML')
    parser.add_argument('--n-rounds', type=int, default=10)
    parser.add_argument('--designs-per-round', type=int, default=8)
    parser.add_argument('--length', type=str, default='180-250')
    parser.add_argument('--output-dir', type=str, default='results/rfd3_pipeline')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean template (strip HETATM)
    clean_template = output_dir / 'template_clean.pdb'
    with open(args.template) as f:
        lines = [l for l in f if l.startswith('ATOM') or l.startswith('TER') or l.startswith('END')]
    with open(clean_template, 'w') as f:
        f.writelines(lines)

    # Score native template baseline
    logger.info("=== Scoring native template ===")
    init_pyrosetta()
    import pyrosetta
    native_pose = pyrosetta.pose_from_pdb(str(clean_template))
    native_score = _SFXN(native_pose)
    native_per_res = native_score / native_pose.total_residue()
    logger.info(f"Native: {native_score:.1f} REU ({native_per_res:.1f}/res), {native_pose.total_residue()} residues")

    # Create theozyme input JSON
    theozyme_json = str(output_dir / 'theozyme_input.json')
    create_theozyme_json(str(clean_template), args.constraint, args.length, theozyme_json)

    # Copy template to output dir (Docker mount needs it in same dir as JSON)
    import shutil
    template_in_output = output_dir / Path(args.template).name
    if str(clean_template) != str(template_in_output):
        shutil.copy(str(clean_template), str(template_in_output))

    # Main loop
    all_scores = []
    best_ever = {'rosetta_total': float('inf')}
    reward_history = []

    for rnd in range(args.n_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {rnd+1}/{args.n_rounds}")
        logger.info(f"{'='*60}")

        round_dir = output_dir / f'round_{rnd:03d}'

        # Generate designs
        t0 = time.time()
        cif_files = generate_designs_rfd3(
            theozyme_json, str(round_dir), n_designs=args.designs_per_round,
        )
        t_gen = time.time() - t0

        if not cif_files:
            logger.warning("No designs generated")
            continue

        # Score all designs
        t0 = time.time()
        round_scores = []
        for cif_path in cif_files:
            json_path = cif_path.replace('.cif.gz', '.json')
            scores = score_design(cif_path, json_path)
            scores['cif_path'] = cif_path
            scores['round'] = rnd
            round_scores.append(scores)
        t_score = time.time() - t0

        # Compute rewards (lower Rosetta = better)
        rosetta_scores = [s.get('rosetta_total', float('inf')) for s in round_scores]
        valid_scores = [s for s in rosetta_scores if s != float('inf') and not np.isnan(s)]

        if valid_scores:
            round_best = min(valid_scores)
            round_mean = np.mean(valid_scores)
            round_worst = max(valid_scores)
            reward = -round_mean  # higher reward = better (lower Rosetta)
            reward_history.append(reward)

            if round_best < best_ever.get('rosetta_total', float('inf')):
                best_idx = rosetta_scores.index(round_best)
                best_ever = round_scores[best_idx].copy()
                best_ever['round'] = rnd

            logger.info(
                f"  Generation: {t_gen:.0f}s, Scoring: {t_score:.0f}s"
            )
            logger.info(
                f"  Rosetta: best={round_best:.1f}, mean={round_mean:.1f}, worst={round_worst:.1f}"
            )
            logger.info(
                f"  Per-residue: best={round_best/round_scores[rosetta_scores.index(round_best)].get('n_residues',1):.1f}"
            )
            logger.info(
                f"  Best ever: {best_ever.get('rosetta_total', 'N/A'):.1f} (round {best_ever.get('round', '?')})"
            )

            # Show best design details
            best_design = round_scores[rosetta_scores.index(round_best)]
            logger.info(
                f"  Best design: {best_design.get('n_residues', '?')} res, "
                f"clashes={best_design.get('rfd3_n_clashes', '?')}, "
                f"helix={best_design.get('rfd3_helix_fraction', 0):.0%}"
            )
        else:
            logger.warning("No valid Rosetta scores this round")

        all_scores.extend(round_scores)

        # Save round results
        with open(round_dir / 'scores.json', 'w') as f:
            json.dump(round_scores, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds: {args.n_rounds}")
    logger.info(f"Total designs: {len(all_scores)}")
    logger.info(f"Native template: {native_score:.1f} REU ({native_per_res:.1f}/res)")

    if best_ever.get('rosetta_total') != float('inf'):
        best_per_res = best_ever['rosetta_total'] / max(best_ever.get('n_residues', 1), 1)
        logger.info(f"Best design: {best_ever['rosetta_total']:.1f} REU ({best_per_res:.1f}/res)")
        logger.info(f"  from round {best_ever.get('round', '?')}")

    if reward_history and len(reward_history) >= 4:
        first2 = np.mean(reward_history[:2])
        last2 = np.mean(reward_history[-2:])
        logger.info(f"Reward trend: first2={first2:.1f} → last2={last2:.1f} (Δ={last2-first2:+.1f})")

    # Save all results
    with open(output_dir / 'all_scores.json', 'w') as f:
        json.dump(all_scores, f, indent=2, default=str)
    with open(output_dir / 'best_design.json', 'w') as f:
        json.dump(best_ever, f, indent=2, default=str)
    with open(output_dir / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)

    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
