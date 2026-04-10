"""End-to-end enzyme design pipeline.

Runs the full loop:
1. RFdiffusion: generate backbone variants (partial diffusion from template)
2. ProteinMPNN: design sequences with fixed catalytic residues
3. Score with BOTH learned surrogates AND Rosetta (compare them)
4. RL: select best designs, update search parameters
5. Repeat

Supports two modes:
- Active-site only: redesign ~30 residues around catalytic site
- Full protein: redesign entire backbone + sequence

Usage:
    # On ai-gpu (RFdiffusion + everything)
    python experiments/pipeline/run_end_to_end.py \
        --template data/pdb/1TIM.pdb \
        --constraint data/catalytic_sites/1TIM.yaml \
        --mode active_site \
        --n-designs 10 \
        --n-rounds 5

    # Full protein mode
    python experiments/pipeline/run_end_to_end.py \
        --template data/pdb/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --mode full_protein \
        --n-designs 20
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
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml, ActiveSiteSpec
from src.data.protein_structure import ProteinBackbone
from src.utils.protein_constants import AA_1TO3, BACKBONE_ATOMS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration."""
    # Paths
    template_pdb: str = ''
    constraint_yaml: str = ''
    rfdiffusion_dir: str = '~/RFdiffusion'
    proteinmpnn_dir: str = '~/ProteinMPNN'
    surrogate_checkpoint: Optional[str] = None
    output_dir: str = 'results/pipeline'

    # Mode
    mode: str = 'active_site'  # 'active_site' or 'full_protein'

    # RFdiffusion params
    partial_T: int = 15  # noise steps (controls deviation)
    n_designs: int = 10  # backbones per round
    diffusion_steps: int = 50

    # ProteinMPNN params
    n_sequences_per_backbone: int = 4
    sampling_temperature: float = 0.1

    # RL params
    n_rounds: int = 5
    top_k: int = 5  # keep top K designs per round

    # Device
    device: str = 'cuda'


def write_pdb(backbone: ProteinBackbone, path: str):
    """Write backbone to PDB file."""
    lines = []
    atom_num = 1
    for i in range(backbone.length):
        resname = 'ALA'
        if backbone.sequence and i < len(backbone.sequence):
            resname = AA_1TO3.get(backbone.sequence[i], 'ALA')
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            pos = backbone.coords[i, j]
            lines.append(
                f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname:3s} A{i+1:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            atom_num += 1
    lines.append("END")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def run_rfdiffusion(
    config: PipelineConfig,
    template_pdb: str,
    n_residues: int,
    output_prefix: str,
) -> List[str]:
    """Run RFdiffusion to generate backbone variants."""
    rfdiff_dir = Path(config.rfdiffusion_dir).expanduser()
    script = rfdiff_dir / 'scripts' / 'run_inference.py'
    ckpt = rfdiff_dir / 'models' / 'Base_epoch8_ckpt.pt'

    # Set LD_LIBRARY_PATH for DGL
    import os
    env = os.environ.copy()
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:" + env.get('LD_LIBRARY_PATH', '')

    cmd = [
        'python3', str(script),
        f'inference.output_prefix={output_prefix}',
        f'inference.input_pdb={template_pdb}',
        f'inference.ckpt_override_path={ckpt}',
        f'inference.num_designs={config.n_designs}',
        f'diffuser.T={config.diffusion_steps}',
        f'diffuser.partial_T={config.partial_T}',
    ]

    if config.mode == 'full_protein':
        cmd.append(f'contigmap.contigs=[{n_residues}-{n_residues}]')
    else:
        # Active site: keep full length but partial diffusion
        cmd.append(f'contigmap.contigs=[{n_residues}-{n_residues}]')

    logger.info(f"Running RFdiffusion: {config.n_designs} designs, partial_T={config.partial_T}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)

    if result.returncode != 0:
        logger.error(f"RFdiffusion failed:\n{result.stderr[-500:]}")
        return []

    # Collect output PDBs
    output_pdbs = sorted(Path(output_prefix).parent.glob(f'{Path(output_prefix).name}_*.pdb'))
    logger.info(f"Generated {len(output_pdbs)} backbones")
    return [str(p) for p in output_pdbs]


def run_proteinmpnn(
    config: PipelineConfig,
    pdb_path: str,
    fixed_positions: List[int],
    n_sequences: int,
) -> List[str]:
    """Run ProteinMPNN to design sequences."""
    mpnn_dir = Path(config.proteinmpnn_dir).expanduser()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Parse PDB
        parse_script = mpnn_dir / 'helper_scripts' / 'parse_multiple_chains.py'
        pdb_dir = tmpdir / 'pdbs'
        pdb_dir.mkdir()
        # Copy PDB
        import shutil
        shutil.copy(pdb_path, pdb_dir / Path(pdb_path).name)

        parsed = tmpdir / 'parsed.jsonl'
        subprocess.run(
            ['python3', str(parse_script),
             f'--input_path={pdb_dir}',
             f'--output_path={parsed}'],
            capture_output=True, text=True, check=True,
        )

        # Fixed positions
        fixed_jsonl = None
        if fixed_positions:
            fixed_jsonl = tmpdir / 'fixed.jsonl'
            # Load parsed to get total length
            with open(parsed) as f:
                pdb_data = json.loads(f.readline())
            total_len = sum(len(v) for k, v in pdb_data.items() if k.startswith('seq_chain'))

            non_fixed = [str(i + 1) for i in range(total_len) if i not in fixed_positions]
            make_fixed = mpnn_dir / 'helper_scripts' / 'make_fixed_positions_dict.py'
            subprocess.run(
                ['python3', str(make_fixed),
                 f'--input_path={parsed}',
                 f'--output_path={fixed_jsonl}',
                 '--chain_list', 'A',
                 '--position_list', ' '.join(non_fixed),
                 '--specify_non_fixed'],
                capture_output=True, text=True, check=True,
            )

        # Run ProteinMPNN
        output_dir = tmpdir / 'output'
        output_dir.mkdir()

        cmd = [
            'python3', str(mpnn_dir / 'protein_mpnn_run.py'),
            '--jsonl_path', str(parsed),
            '--out_folder', str(output_dir),
            '--num_seq_per_target', str(n_sequences),
            '--sampling_temp', str(config.sampling_temperature),
            '--seed', '42',
            '--path_to_model_weights', str(mpnn_dir / 'vanilla_model_weights'),
            '--model_name', 'v_48_010',
        ]
        if fixed_jsonl:
            cmd.extend(['--fixed_positions_jsonl', str(fixed_jsonl)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"ProteinMPNN failed:\n{result.stderr[-300:]}")
            return []

        # Parse output FASTA
        sequences = []
        for fasta_file in sorted((output_dir / 'seqs').glob('*.fa')):
            with open(fasta_file) as f:
                for line in f:
                    if not line.startswith('>'):
                        seq = line.strip()
                        if seq:
                            sequences.append(seq)

        return sequences[:n_sequences]


_SURROGATE_CACHE = {}
_ESM_MODEL = None
_ESM_ALPHABET = None


def _get_esm_model():
    """Lazy-load ESM-2 model."""
    global _ESM_MODEL, _ESM_ALPHABET
    if _ESM_MODEL is None:
        import esm
        _ESM_MODEL, _ESM_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _ESM_MODEL = _ESM_MODEL.eval().to(device)
        logger.info(f"Loaded ESM-2 on {device}")
    return _ESM_MODEL, _ESM_ALPHABET


@torch.no_grad()
def _compute_esm_embedding(sequence: str) -> Optional[torch.Tensor]:
    """Compute ESM-2 embedding for a sequence on-the-fly."""
    try:
        model, alphabet = _get_esm_model()
        device = next(model.parameters()).device
        batch_converter = alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("protein", sequence)])
        tokens = tokens.to(device)
        results = model(tokens, repr_layers=[33], return_contacts=False)
        emb = results["representations"][33][0, 1:len(sequence)+1].cpu()  # (L, 1280)
        return emb
    except Exception as e:
        logger.warning(f"ESM embedding failed: {e}")
        return None

def _load_surrogate(checkpoint_path: str, input_dim: int = 1320):
    """Load a trained surrogate model."""
    if checkpoint_path in _SURROGATE_CACHE:
        return _SURROGATE_CACHE[checkpoint_path]

    from experiments.scoring.train_surrogates_on_mutations import SurrogateModel
    model = SurrogateModel(input_dim, 1, [512, 256, 128])
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    _SURROGATE_CACHE[checkpoint_path] = model
    return model


def score_with_surrogate(
    backbone_pdb: str,
    sequence: str,
    surrogate_dir: Optional[str] = None,
    esm_cache_dir: str = 'cache/esm_embeddings',
) -> Dict[str, float]:
    """Score a design with trained surrogates.

    For each position, predicts the ΔΔG of the designed AA vs wild-type
    using the trained energy term surrogates.
    """
    if surrogate_dir is None or not Path(surrogate_dir).exists():
        return {'surrogate_total_ddg': float('nan')}

    surrogate_dir = Path(surrogate_dir)
    results = {}

    # Load surrogates
    surrogate_names = ['total_ddg', 'd_fa_atr', 'd_fa_sol', 'd_fa_elec', 'd_fa_dun',
                       'd_fa_rep', 'd_hbond_sc', 'd_hbond_bb_sc', 'd_rama_prepro']
    models = {}
    for name in surrogate_names:
        ckpt = surrogate_dir / f'surrogate_{name}.pt'
        if ckpt.exists():
            try:
                models[name] = _load_surrogate(str(ckpt))
            except Exception:
                pass

    if not models:
        return {'surrogate_total_ddg': float('nan')}

    # Get ESM embedding — compute on-the-fly if not cached
    try:
        from src.utils.feature_cache import FeatureCache, get_sequence_hash
        cache = FeatureCache(esm_cache_dir)
        seq_hash = get_sequence_hash(sequence)
        key = {'sequence_hash': seq_hash, 'model': 'esm2_t33_650M_UR50D'}
        if cache.has(key):
            esm_data = cache.load(key)
            esm_emb = esm_data[:-1]  # (L, 1280)
        else:
            # Compute ESM embedding on-the-fly
            esm_emb = _compute_esm_embedding(sequence)
            if esm_emb is None:
                # Fall back to template embedding if available
                return {'surrogate_total_ddg': float('nan')}
            # Cache for reuse
            cache_data = torch.cat([esm_emb, esm_emb.mean(dim=0, keepdim=True)], dim=0)
            from src.utils.feature_cache import CacheMetadata
            cache.save(key, cache_data, CacheMetadata(method='esm2_t33_650M_UR50D', source='designed'))
    except Exception as e:
        logger.debug(f"ESM embedding failed: {e}")
        return {'surrogate_total_ddg': float('nan')}

    # Score: average surrogate prediction across all positions
    L = min(len(sequence), esm_emb.shape[0])
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    with torch.no_grad():
        for name, model in models.items():
            preds = []
            for pos in range(L):
                aa = sequence[pos]
                if aa not in aa_to_idx:
                    continue
                # Feature: ESM embedding + one-hot WT (zeros) + one-hot mut
                pos_emb = esm_emb[pos]
                if isinstance(pos_emb, torch.Tensor):
                    pos_emb = pos_emb.float()
                else:
                    pos_emb = torch.tensor(pos_emb, dtype=torch.float32)
                wt_oh = torch.zeros(20)
                mut_oh = torch.zeros(20)
                mut_oh[aa_to_idx[aa]] = 1
                feat = torch.cat([pos_emb, wt_oh, mut_oh]).unsqueeze(0)
                pred = model(feat).item()
                preds.append(pred)
            if preds:
                results[f'surrogate_{name}'] = float(np.mean(preds))

    return results


_PYROSETTA_INIT = False

def score_with_rosetta(pdb_path: str, do_relax: bool = True) -> Dict[str, float]:
    """Score a design with actual Rosetta (ground truth comparison).

    If do_relax=True, performs FastRelax before scoring to resolve clashes
    and get meaningful energy values (critical for RFdiffusion-generated backbones).
    """
    try:
        import pyrosetta
        global _PYROSETTA_INIT
        if not _PYROSETTA_INIT:
            pyrosetta.init('-mute all -ex1 -ex2')
            _PYROSETTA_INIT = True

        sfxn = pyrosetta.get_score_function(True)
        pose = pyrosetta.pose_from_pdb(pdb_path)

        # Relax to resolve clashes from backbone generation
        if do_relax:
            from pyrosetta.rosetta.protocols.relax import FastRelax
            relax = FastRelax()
            relax.set_scorefxn(sfxn)
            relax.max_iter(100)  # quick relax
            relax.apply(pose)

        total = sfxn(pose)

        from pyrosetta.rosetta.core.scoring import ScoreType
        terms = {}
        for term_name in ['fa_atr', 'fa_rep', 'fa_sol', 'fa_elec', 'hbond_sr_bb',
                          'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'fa_dun', 'rama_prepro']:
            try:
                st = getattr(ScoreType, term_name)
                terms[f'rosetta_{term_name}'] = float(pose.energies().total_energies()[st])
            except:
                pass

        terms['rosetta_total'] = total
        return terms

    except ImportError:
        logger.warning("PyRosetta not available, skipping Rosetta scoring")
        return {'rosetta_total': float('nan')}


def evaluate_design(
    backbone_pdb: str,
    sequence: str,
    template: ProteinBackbone,
    constraint,
    config: PipelineConfig,
) -> dict:
    """Evaluate a single design with both surrogate and Rosetta."""
    bb = load_pdb(backbone_pdb)

    # Structural quality
    from src.utils.metrics import bond_geometry_metrics, clash_score
    coords_t = torch.tensor(bb.coords, dtype=torch.float32)
    geom = bond_geometry_metrics(coords_t)
    clash = clash_score(coords_t)

    # RMSD to template (CA atoms)
    from src.utils.geometry import kabsch_rmsd
    min_len = min(bb.length, template.length)
    ca_gen = torch.tensor(bb.ca_coords[:min_len], dtype=torch.float32)
    ca_tmpl = torch.tensor(template.ca_coords[:min_len], dtype=torch.float32)
    rmsd, _, _ = kabsch_rmsd(ca_tmpl, ca_gen)

    # Sequence recovery vs template
    recovery = 0.0
    catalytic_recovery = 0.0
    if template.sequence and sequence:
        matches = sum(1 for a, b in zip(sequence, template.sequence) if a == b)
        recovery = matches / min(len(sequence), len(template.sequence))

        # Catalytic residue recovery (should be high — these are fixed)
        cat_positions = [r.position_index for r in constraint.residues
                        if r.position_index is not None]
        if cat_positions:
            cat_matches = sum(1 for p in cat_positions
                            if p < len(sequence) and p < len(template.sequence)
                            and sequence[p] == template.sequence[p])
            catalytic_recovery = cat_matches / len(cat_positions)

    # Surrogate scores (using trained models)
    surr_scores = score_with_surrogate(
        backbone_pdb, sequence,
        surrogate_dir=config.surrogate_checkpoint,
    )

    # Rosetta scores (ground truth, with relax to resolve clashes)
    rosetta_scores = score_with_rosetta(backbone_pdb, do_relax=True)

    return {
        'backbone_pdb': backbone_pdb,
        'sequence': sequence[:50] + '...' if len(sequence) > 50 else sequence,
        'seq_length': len(sequence),
        'template_rmsd': float(rmsd),
        'sequence_recovery': recovery,
        'catalytic_recovery': catalytic_recovery,
        'clash_score': clash,
        'n_ca_bond_deviation': geom.get('n_ca_bond_deviation', 0),
        **surr_scores,
        **rosetta_scores,
    }


def main():
    parser = argparse.ArgumentParser(description='End-to-end enzyme design pipeline')
    parser.add_argument('--template', required=True, help='Template PDB path')
    parser.add_argument('--constraint', required=True, help='Catalytic constraint YAML')
    parser.add_argument('--mode', choices=['active_site', 'full_protein'], default='active_site')
    parser.add_argument('--n-designs', type=int, default=10)
    parser.add_argument('--n-sequences', type=int, default=4)
    parser.add_argument('--n-rounds', type=int, default=3)
    parser.add_argument('--partial-T', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default='results/pipeline')
    parser.add_argument('--rfdiffusion-dir', type=str, default='~/RFdiffusion')
    parser.add_argument('--proteinmpnn-dir', type=str, default='~/ProteinMPNN')
    parser.add_argument('--surrogate-ckpt', type=str, default=None)
    args = parser.parse_args()

    config = PipelineConfig(
        template_pdb=args.template,
        constraint_yaml=args.constraint,
        mode=args.mode,
        n_designs=args.n_designs,
        n_sequences_per_backbone=args.n_sequences,
        n_rounds=args.n_rounds,
        partial_T=args.partial_T,
        sampling_temperature=args.temperature,
        output_dir=args.output_dir,
        rfdiffusion_dir=args.rfdiffusion_dir,
        proteinmpnn_dir=args.proteinmpnn_dir,
        surrogate_checkpoint=args.surrogate_ckpt,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template and constraints
    template = load_pdb(config.template_pdb)
    constraint = load_constraint_from_yaml(config.constraint_yaml)
    fixed_positions = [r.position_index for r in constraint.residues if r.position_index is not None]

    logger.info(f"Template: {template.pdb_id}, {template.length} residues")
    logger.info(f"Catalytic residues: {len(constraint.residues)} at positions {fixed_positions}")
    logger.info(f"Mode: {config.mode}, {config.n_rounds} rounds × {config.n_designs} designs")

    # Clean template PDB (strip heteroatoms)
    clean_template = output_dir / 'template_clean.pdb'
    with open(config.template_pdb) as f:
        clean_lines = [l for l in f if l.startswith('ATOM') or l.startswith('TER') or l.startswith('END')]
    with open(clean_template, 'w') as f:
        f.writelines(clean_lines)

    all_results = []
    best_designs = []

    for round_num in range(config.n_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num + 1}/{config.n_rounds}")
        logger.info(f"{'='*60}")

        round_dir = output_dir / f'round_{round_num:02d}'
        round_dir.mkdir(exist_ok=True)

        # Step 1: Generate backbones
        t0 = time.time()
        backbone_prefix = round_dir / 'backbone'
        backbone_pdbs = run_rfdiffusion(
            config, str(clean_template), template.length, str(backbone_prefix),
        )
        t_backbone = time.time() - t0
        logger.info(f"Step 1 (RFdiffusion): {len(backbone_pdbs)} backbones in {t_backbone:.1f}s")

        if not backbone_pdbs:
            logger.error("No backbones generated, skipping round")
            continue

        # Step 2: Design sequences for each backbone
        round_designs = []
        t0 = time.time()
        for bb_pdb in backbone_pdbs:
            sequences = run_proteinmpnn(
                config, bb_pdb, fixed_positions, config.n_sequences_per_backbone,
            )
            for seq in sequences:
                round_designs.append({
                    'backbone_pdb': bb_pdb,
                    'sequence': seq,
                    'round': round_num,
                })
        t_sequence = time.time() - t0
        logger.info(f"Step 2 (ProteinMPNN): {len(round_designs)} sequences in {t_sequence:.1f}s")

        # Step 3: Score all designs (surrogate + Rosetta)
        t0 = time.time()
        round_results = []
        for design in round_designs:
            result = evaluate_design(
                design['backbone_pdb'],
                design['sequence'],
                template,
                constraint,
                config,
            )
            result['round'] = round_num
            round_results.append(result)

        t_scoring = time.time() - t0
        logger.info(f"Step 3 (Scoring): {len(round_results)} designs in {t_scoring:.1f}s")

        # Step 4: Rank and select top designs
        # Sort by Rosetta total (lower = better) if available, else surrogate
        valid_results = [r for r in round_results if not np.isnan(r.get('rosetta_total', float('nan')))]
        if valid_results:
            valid_results.sort(key=lambda x: x['rosetta_total'])
            top = valid_results[:config.top_k]
        else:
            top = round_results[:config.top_k]

        # Log round summary
        logger.info(f"\nRound {round_num + 1} Summary:")
        logger.info(f"  Designs evaluated: {len(round_results)}")
        logger.info(f"  Time: backbone={t_backbone:.1f}s, sequence={t_sequence:.1f}s, scoring={t_scoring:.1f}s")

        for i, r in enumerate(top[:3]):
            logger.info(
                f"  Top {i+1}: RMSD={r['template_rmsd']:.2f}Å, "
                f"recovery={r['sequence_recovery']:.1%}, "
                f"rosetta={r.get('rosetta_total', 'N/A')}"
            )

        all_results.extend(round_results)
        best_designs.extend(top)

        # Save round results
        round_file = round_dir / 'results.json'
        with open(round_file, 'w') as f:
            json.dump(round_results, f, indent=2, default=str)

    # Save all results
    final_file = output_dir / 'all_results.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    best_file = output_dir / 'best_designs.json'
    with open(best_file, 'w') as f:
        json.dump(best_designs, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total designs: {len(all_results)}")
    logger.info(f"Best designs saved to: {best_file}")

    if all_results:
        rmsds = [r['template_rmsd'] for r in all_results]
        logger.info(f"RMSD range: [{min(rmsds):.2f}, {max(rmsds):.2f}] Å")

        rosetta_scores = [r.get('rosetta_total', float('nan')) for r in all_results]
        valid_scores = [s for s in rosetta_scores if not np.isnan(s)]
        if valid_scores:
            logger.info(f"Rosetta score range: [{min(valid_scores):.1f}, {max(valid_scores):.1f}]")

            # Compare surrogate vs Rosetta
            logger.info(f"\n=== Surrogate vs Rosetta Comparison ===")
            surr = [r.get('surrogate_stability', 0) for r in all_results if not np.isnan(r.get('rosetta_total', float('nan')))]
            ros = [r['rosetta_total'] for r in all_results if not np.isnan(r.get('rosetta_total', float('nan')))]
            if len(surr) > 2:
                from scipy import stats
                corr, pval = stats.pearsonr(surr, ros)
                logger.info(f"Surrogate-Rosetta correlation: r={corr:.3f}, p={pval:.3e}")


if __name__ == '__main__':
    main()
