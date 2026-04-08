"""Wrapper for RFdiffusion pretrained backbone generator.

RFdiffusion (Baker lab) is a pretrained SE(3)-equivariant diffusion model
trained on the entire PDB. We use the ActiveSite checkpoint for catalytic
site scaffolding — generating backbones conditioned on catalytic residue
geometry.

RFdiffusion2 (Nature Methods 2025) extends this to atom-level active site
scaffolding, designing from functional group geometries without specifying
residue positions.

This wrapper provides a unified interface that:
1. Converts our CatalyticConstraint / ActiveSiteSpec to RFdiffusion's
   contig mapping and input PDB format
2. Runs RFdiffusion inference
3. Converts output back to our ProteinBackbone representation
4. Supports integration with our RL loop

Setup:
    git clone https://github.com/RosettaCommons/RFdiffusion.git external/RFdiffusion
    cd external/RFdiffusion && pip install -e .
    Download ActiveSite_ckpt.pt to external/RFdiffusion/models/

Usage:
    wrapper = RFdiffusionWrapper(model_dir='external/RFdiffusion')
    backbone = wrapper.generate(spec, n_designs=10)
"""

import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.data.protein_structure import ProteinBackbone
from src.data.catalytic_constraints import ActiveSiteSpec, CatalyticConstraint
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RFdiffusionConfig:
    """Configuration for RFdiffusion inference."""
    # Paths
    rfdiffusion_dir: str = 'external/RFdiffusion'
    checkpoint: str = 'ActiveSite_ckpt.pt'

    # Generation
    n_designs: int = 10
    n_steps: int = 50  # denoising steps (T)
    noise_scale_ca: float = 0.0  # 0 = deterministic from motif
    noise_scale_frame: float = 0.0

    # Scaffold sizing
    min_scaffold_length: int = 20  # residues before/after motif
    max_scaffold_length: int = 80

    # Partial diffusion (for template-conditioned generation)
    partial_T: Optional[int] = None  # if set, only diffuse for partial_T steps
    partial_noise_scale: float = 1.0


class RFdiffusionWrapper:
    """Wrapper around RFdiffusion for backbone generation.

    Provides a Pythonic interface for generating protein backbones
    conditioned on catalytic site constraints using pretrained RFdiffusion.
    """

    def __init__(self, config: Optional[RFdiffusionConfig] = None):
        if config is None:
            config = RFdiffusionConfig()
        self.config = config
        self.rfdiffusion_dir = Path(config.rfdiffusion_dir)

        # Verify installation
        self._verify_installation()

    def _verify_installation(self):
        """Check that RFdiffusion is installed and weights are available."""
        script = self.rfdiffusion_dir / 'scripts' / 'run_inference.py'
        if not script.exists():
            logger.warning(
                f"RFdiffusion not found at {self.rfdiffusion_dir}. "
                f"Install with: git clone https://github.com/RosettaCommons/RFdiffusion.git {self.rfdiffusion_dir}"
            )
            self._available = False
            return

        ckpt = self.rfdiffusion_dir / 'models' / self.config.checkpoint
        if not ckpt.exists():
            logger.warning(
                f"Checkpoint {self.config.checkpoint} not found. "
                f"Download from: https://huggingface.co/GlandVergil/RFdiffusion"
            )
            self._available = False
            return

        self._available = True
        logger.info(f"RFdiffusion ready at {self.rfdiffusion_dir}")

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        spec: ActiveSiteSpec,
        n_designs: Optional[int] = None,
    ) -> List[ProteinBackbone]:
        """Generate backbones conditioned on catalytic constraints.

        Args:
            spec: Active site specification with catalytic constraints
            n_designs: Number of designs to generate (overrides config)

        Returns:
            List of generated ProteinBackbone objects
        """
        if not self.is_available:
            raise RuntimeError("RFdiffusion not installed. See docstring for setup.")

        n = n_designs or self.config.n_designs

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Write input PDB with catalytic residues
            input_pdb = tmpdir / 'motif.pdb'
            self._write_motif_pdb(spec, input_pdb)

            # 2. Build contig string
            contigs = self._build_contigs(spec)

            # 3. Run RFdiffusion
            output_prefix = tmpdir / 'design'
            self._run_inference(input_pdb, contigs, output_prefix, n)

            # 4. Parse output PDBs
            backbones = []
            for i in range(n):
                out_pdb = tmpdir / f'design_{i}.pdb'
                if out_pdb.exists():
                    try:
                        from src.data.pdb_loader import load_pdb
                        bb = load_pdb(str(out_pdb))
                        bb.pdb_id = f'rfdiff_design_{i}'
                        backbones.append(bb)
                    except Exception as e:
                        logger.warning(f"Failed to parse design {i}: {e}")

            logger.info(f"Generated {len(backbones)}/{n} backbones")
            return backbones

    def generate_from_template(
        self,
        template: ProteinBackbone,
        spec: ActiveSiteSpec,
        partial_T: Optional[int] = None,
        n_designs: int = 10,
    ) -> List[ProteinBackbone]:
        """Generate backbones by partially diffusing from a template.

        This is our primary use case: start from an existing backbone
        and explore nearby conformational space while maintaining
        catalytic geometry.

        Args:
            template: Starting backbone structure
            spec: Catalytic constraints to maintain
            partial_T: Number of noising steps (controls deviation from template)
            n_designs: Number of designs

        Returns:
            List of generated ProteinBackbone objects
        """
        if not self.is_available:
            raise RuntimeError("RFdiffusion not installed.")

        pT = partial_T or self.config.partial_T or 10

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write template PDB
            input_pdb = tmpdir / 'template.pdb'
            self._write_backbone_pdb(template, input_pdb)

            # Contigs for partial diffusion (reference full chain)
            L = template.length
            contigs = f"'{L}/{L}'"  # keep same length

            output_prefix = tmpdir / 'design'

            cmd = self._build_command(
                input_pdb, contigs, output_prefix, n_designs,
                extra_args=[
                    f'diffuser.partial_T={pT}',
                    f'denoiser.noise_scale_ca={self.config.partial_noise_scale}',
                    f'denoiser.noise_scale_frame={self.config.partial_noise_scale}',
                ]
            )

            self._execute(cmd)

            backbones = []
            for i in range(n_designs):
                out_pdb = tmpdir / f'design_{i}.pdb'
                if out_pdb.exists():
                    try:
                        from src.data.pdb_loader import load_pdb
                        bb = load_pdb(str(out_pdb))
                        bb.pdb_id = f'rfdiff_partial_{i}'
                        backbones.append(bb)
                    except Exception as e:
                        logger.warning(f"Failed to parse design {i}: {e}")

            return backbones

    def _write_motif_pdb(self, spec: ActiveSiteSpec, path: Path):
        """Write catalytic residues as a minimal PDB for RFdiffusion input."""
        lines = []
        atom_num = 1
        for i, res in enumerate(spec.constraint.residues):
            res_num = res.position_index if res.position_index is not None else i + 1
            for atom_name, pos in res.atom_positions.items():
                lines.append(
                    f"ATOM  {atom_num:5d}  {atom_name:<3s} {res.residue_type:3s} A{res_num:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
                )
                atom_num += 1
        lines.append("END")
        path.write_text('\n'.join(lines))

    def _write_backbone_pdb(self, backbone: ProteinBackbone, path: Path):
        """Write a ProteinBackbone as PDB."""
        from src.utils.protein_constants import BACKBONE_ATOMS
        lines = []
        atom_num = 1
        for i in range(backbone.length):
            resname = 'ALA'  # default for backbone-only
            if backbone.sequence and i < len(backbone.sequence):
                from src.utils.protein_constants import AA_1TO3
                resname = AA_1TO3.get(backbone.sequence[i], 'ALA')

            for j, atom_name in enumerate(BACKBONE_ATOMS):
                pos = backbone.coords[i, j]
                lines.append(
                    f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname:3s} A{i+1:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
                )
                atom_num += 1
        lines.append("END")
        path.write_text('\n'.join(lines))

    def _build_contigs(self, spec: ActiveSiteSpec) -> str:
        """Build RFdiffusion contig string from ActiveSiteSpec."""
        min_len = self.config.min_scaffold_length
        max_len = self.config.max_scaffold_length

        # If we have specific residue positions, build around them
        if spec.constraint.residues and spec.constraint.residues[0].position_index is not None:
            positions = sorted(
                r.position_index for r in spec.constraint.residues
                if r.position_index is not None
            )
            # Simple: scaffold before + motif residues + scaffold after
            motif_start = min(positions)
            motif_end = max(positions)
            motif_str = f"A{motif_start}-{motif_end}"
            contig = f"'{min_len}-{max_len}/{motif_str}/{min_len}-{max_len}'"
        else:
            # No specific positions: just constrain total length
            total = min_len * 2 + 10  # rough estimate
            contig = f"'{total}-{total + max_len}'"

        return contig

    def _build_command(
        self,
        input_pdb: Path,
        contigs: str,
        output_prefix: Path,
        n_designs: int,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the RFdiffusion command."""
        script = self.rfdiffusion_dir / 'scripts' / 'run_inference.py'
        ckpt = self.rfdiffusion_dir / 'models' / self.config.checkpoint

        cmd = [
            'python', str(script),
            f'inference.output_prefix={output_prefix}',
            f'inference.input_pdb={input_pdb}',
            f'inference.ckpt_override_path={ckpt}',
            f'inference.num_designs={n_designs}',
            f'contigmap.contigs=[{contigs}]',
            f'diffuser.T={self.config.n_steps}',
            f'denoiser.noise_scale_ca={self.config.noise_scale_ca}',
            f'denoiser.noise_scale_frame={self.config.noise_scale_frame}',
        ]

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def _run_inference(
        self, input_pdb: Path, contigs: str, output_prefix: Path, n_designs: int
    ):
        """Run RFdiffusion inference."""
        cmd = self._build_command(input_pdb, contigs, output_prefix, n_designs)
        self._execute(cmd)

    def _execute(self, cmd: List[str]):
        """Execute a shell command."""
        logger.info(f"Running: {' '.join(cmd[:5])}...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            logger.error(f"RFdiffusion failed:\n{result.stderr}")
            raise RuntimeError(f"RFdiffusion failed: {result.stderr[:500]}")
        logger.info("RFdiffusion completed successfully")
