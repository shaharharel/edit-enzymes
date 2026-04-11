# CLAUDE.md - Project Guidelines for AI Assistants

## Environment Setup

**Use the `quris` conda environment for all Python operations.**

---

## Project Overview

edit-enzymes is a learnable enzyme design system that turns the htFuncLib filter-based pipeline into a generative system optimized end-to-end via reinforcement learning. Joint work with the Sarel Lab.

### Core Idea

Instead of generating many enzyme candidates and filtering most of them (phylogenetic → Rosetta → enumeration → ML classification), we train models to directly produce designs satisfying all objectives simultaneously:
- **Backbone Generator**: SE(3)-equivariant diffusion conditioned on template backbone + catalytic constraints
- **Sequence Generator**: ProteinMPNN-style graph NN for sequence design on generated backbones
- **Scoring Models**: Learned surrogates for Rosetta/PROSS (stability, packing, desolvation, activity)
- **RL Loop**: Multi-objective optimization with separate credit assignment for structure vs sequence

### Starting Point

We condition on an existing backbone template and search in a small space around it, rather than generating from scratch. This reduces the search space and improves reliability.

---

## IMPORTANT: Always Check the Plan

**Before implementing anything, read the plan files:**
- `ENZYME_DESIGN_PLAN_2026-04-07.md` — Architecture, design decisions, component specs
- `ENZYME_DESIGN_TRACKING_2026-04-07.md` — Progress, what's done, what's next

**The plan specifies:**
- REINFORCE for backbone generator (differentiable, not black-box)
- PPO for sequence generator (discrete autoregressive actions)
- Separated credit assignment (backbone gets geometry reward, sequence gets stability reward)
- Rosetta energy as primary reward signal
- Progressive unfreezing experiments A→D

---

## Code Organization Rules

### Strict Source Code Policy

**All code must be written within the project's source structure:**

```
src/           # Core library code (data loaders, models, layers, utils)
experiments/   # Experiment runners with dataclass configs
scripts/       # Data preprocessing and analysis scripts
tests/         # Unit tests
```

**Do NOT write code outside these directories.**

---

## Project Structure

```
src/
├── data/
│   ├── catalytic_constraints.py   # CatalyticResidue, CatalyticConstraint, ActiveSiteSpec
│   ├── protein_structure.py       # ProteinBackbone, ProteinGraph
│   ├── pdb_loader.py             # PDB → ProteinBackbone
│   └── dataset_builders.py       # Training datasets for each component
├── models/
│   ├── trainer.py                # Unified training loop
│   ├── backbone_generator/       # SE(3)-equivariant diffusion
│   ├── sequence_generator/       # ProteinMPNN-style sequence design
│   ├── scoring/                  # Learned Rosetta/PROSS surrogates
│   ├── rl/                       # Reinforcement learning loop
│   └── layers/                   # EGNN, IPA, protein graph convolution
└── utils/
    ├── protein_constants.py      # AA vocabulary, atom types, bond lengths
    ├── geometry.py               # Rigid body transforms, RMSD
    ├── so3_utils.py              # SO(3) diffusion utilities
    ├── feature_cache.py          # Universal caching for embeddings + Rosetta
    ├── metrics.py                # Structure quality + regression metrics
    └── logging.py                # Logger setup

experiments/     # Config-driven experiment runners
scripts/         # Data prep and analysis
data/            # Raw data (PDB, Rosetta scores, catalytic sites)
cache/           # Pre-computed features (NEVER committed)
results/         # Experiment outputs
```

---

## Key Patterns

- **Experiments**: Config-driven via dataclasses, placed in `experiments/{component}/`
- **Preprocessing**: Scripts go in `scripts/data_prep/`
- **Models**: Inherit from PyTorch Lightning `pl.LightningModule`
- **Caching**: ALL expensive computations (embeddings, Rosetta) go through `src/utils/feature_cache.py`
- **Data paths**: Default to `data/` subdirectories
- **Imports**: Use `sys.path.insert(0, str(project_root))` for absolute imports from `src/`

---

## Caching Policy

**Every expensive computation must be cached.** This includes:
- ESM-2 embeddings (keyed by sequence hash)
- Rosetta energy computations (keyed by PDB ID + chain + mutation set)
- Structural features (keyed by PDB ID + chain)

Cache lives in `cache/` and is never committed to git. The `feature_cache.py` module provides thread-safe, hash-based caching with metadata tracking.
