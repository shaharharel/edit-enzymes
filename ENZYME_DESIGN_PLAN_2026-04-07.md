# edit-enzymes: Learnable Enzyme Design System

**Date**: 2026-04-07
**Joint work with**: Sarel Lab
**Reference paper**: htFuncLib — "Designed active-site library reveals thousands of functional GFP variants" (Nature Comms 2023, DOI: 10.1038/s41467-023-38099-z)

---

## Motivation

The htFuncLib pipeline designs active-site libraries through sequential filters: phylogenetic filtering → Rosetta ΔΔG scoring → combinatorial enumeration → EpiNNet ML classification. This works but discards most candidates at each stage.

**Our idea**: Turn this filter pipeline into a learnable generative system. Instead of generate-then-filter, train models to directly produce designs satisfying all objectives (stability, packing, catalytic geometry, activity) simultaneously, using RL to optimize a multi-objective reward built from learned surrogates of Rosetta/PROSS.

---

## System Architecture

```
CatalyticConstraint (active site geometry)
        │
        ▼
┌─────────────────────┐
│  Backbone Generator  │  ← Conditioned diffusion on template backbone
│  (small deviations   │  ← Catalytic geometry constraints
│   from template)     │  ← Fold family (e.g. TIM barrel)
└─────────┬───────────┘
          │ ProteinBackbone (L, 4, 3)
          ▼
┌─────────────────────┐
│  Sequence Generator  │  ← ProteinMPNN-style graph NN
│  (fixed catalytic    │  ← Autoregressive decoding
│   residues)          │
└─────────┬───────────┘
          │ amino acid sequence
          ▼
┌─────────────────────┐
│  Scoring Models      │  ← Learned surrogates for Rosetta/PROSS
│  (stability, packing,│  ← Fast + differentiable
│   desolvation, act.) │
└─────────┬───────────┘
          │ multi-objective reward
          ▼
┌─────────────────────┐
│  RL Optimization     │  ← REINFORCE (backbone) / PPO (sequence)
│  (separate credit    │  ← Backbone gets geometry reward
│   assignment)        │  ← Sequence gets stability/activity reward
└─────────────────────┘
```

**Starting point**: Conditioned backbone generator. We condition on an existing backbone template and search in a much smaller space around it, rather than generating from scratch. This drastically reduces the search space and improves reliability.

---

## Project Structure

```
edit-enzymes/
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── pdb/                        # Raw PDB structures
│   ├── rosetta_scores/             # Rosetta/PROSS score datasets
│   ├── catalytic_sites/            # Catalytic constraint definitions
│   └── proteinmpnn/                # ProteinMPNN training data
├── cache/                          # ALL embeddings and Rosetta runs cached here
│   ├── structure_features/
│   ├── esm_embeddings/
│   └── rosetta_features/
├── results/
├── tests/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── catalytic_constraints.py  # CatalyticResidue, CatalyticConstraint, ActiveSiteSpec
│   │   ├── protein_structure.py      # ProteinBackbone, ProteinGraph
│   │   ├── pdb_loader.py            # PDB → ProteinBackbone
│   │   └── dataset_builders.py      # Datasets for each training phase
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Unified trainer (edit-gfp pattern)
│   │   ├── backbone_generator/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # AbstractBackboneGenerator
│   │   │   ├── diffusion_model.py   # SE3DiffusionGenerator (template-conditioned)
│   │   │   ├── noise_schedule.py    # Variance schedules
│   │   │   └── se3_layers.py        # EGNN layers (upgrade to IPA later)
│   │   ├── sequence_generator/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # AbstractSequenceGenerator
│   │   │   ├── mpnn_model.py        # ProteinMPNN-style model
│   │   │   └── graph_features.py    # Backbone → graph features
│   │   ├── scoring/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # AbstractScoringModel
│   │   │   ├── stability_scorer.py  # ΔΔG surrogate
│   │   │   ├── packing_scorer.py    # Packing quality surrogate
│   │   │   ├── desolvation_scorer.py # Active-site desolvation
│   │   │   ├── activity_scorer.py   # Activity proxy
│   │   │   └── multi_objective.py   # MultiObjectiveScorer
│   │   ├── rl/
│   │   │   ├── __init__.py
│   │   │   ├── reward.py            # RewardFunction
│   │   │   ├── backbone_policy.py   # RL wrapper for backbone gen
│   │   │   ├── sequence_policy.py   # RL wrapper for sequence gen
│   │   │   └── ppo_trainer.py       # PPO/REINFORCE implementation
│   │   └── layers/
│   │       ├── __init__.py
│   │       ├── egnn.py              # E(n)-equivariant GNN
│   │       ├── invariant_point_attention.py  # IPA (future upgrade)
│   │       └── protein_graph_conv.py # Message passing on protein graphs
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py              # Rigid body transforms, RMSD
│       ├── so3_utils.py             # SO(3) diffusion utilities
│       ├── feature_cache.py         # CRITICAL: cache all embeddings + Rosetta runs
│       ├── metrics.py               # Structure + regression metrics
│       ├── logging.py               # Logger setup
│       └── protein_constants.py     # AA vocab, atom types, bond lengths
├── scripts/
│   ├── data_prep/
│   │   ├── extract_pdb_backbones.py
│   │   ├── compute_rosetta_scores.py
│   │   ├── generate_proteinmpnn_data.py
│   │   ├── compute_esm_embeddings.py
│   │   └── extract_catalytic_sites.py
│   └── analysis/
│       └── generate_report.py
└── experiments/
    ├── backbone/
    │   ├── train_backbone_diffusion.py
    │   └── configs.py
    ├── sequence/
    │   ├── train_sequence_mpnn.py
    │   └── configs.py
    ├── scoring/
    │   ├── train_scoring_models.py
    │   └── configs.py
    └── rl/
        ├── run_rl_finetuning.py
        └── configs.py
```

---

## Implementation Phases

### Phase 0: Scaffolding
Create all directories, `__init__.py` files, CLAUDE.md, requirements.txt, .gitignore.

### Phase 1: Data Representations + Caching Infrastructure
Core data types and caching system that every component depends on.

- `src/utils/protein_constants.py` — AA vocabulary, backbone atom names, ideal geometry
- `src/utils/feature_cache.py` — **Universal cache for all embeddings and Rosetta computations**. Two-tier: raw results + derived features. Keyed by (PDB ID, chain, method, params hash). Every expensive computation must go through cache.
- `src/data/catalytic_constraints.py` — `CatalyticResidue`, `CatalyticConstraint`, `ActiveSiteSpec`
- `src/data/protein_structure.py` — `ProteinBackbone` (coords `(L,4,3)`, sequence, mask) with `.to_graph(k=30)` → `ProteinGraph`
- `src/data/pdb_loader.py` — BioPython PDB parsing → `ProteinBackbone`
- `src/data/dataset_builders.py` — Dataset classes for each training phase

### Phase 2: Backbone Generator (PRIORITY — start here)
Template-conditioned SE(3)-equivariant diffusion for backbone generation.

**Key insight**: Condition on existing backbone, search in small space around it. Template backbone = starting point for reverse diffusion, noise level controls deviation magnitude.

- `src/utils/geometry.py` — `rigid_from_3_points()`, `compose_rigid()`, `kabsch_rmsd()`
- `src/utils/so3_utils.py` — IGSO3 sampling, SO(3) score functions
- `src/models/layers/egnn.py` — E(n)-equivariant GNN (distance-based, simpler)
- `src/models/layers/invariant_point_attention.py` — IPA from AlphaFold2/FrameDiff (frame-aware, more powerful)
- `src/models/backbone_generator/noise_schedule.py` — Linear/cosine schedules, forward diffusion
- `src/models/backbone_generator/base.py` — `AbstractBackboneGenerator`
- `src/models/backbone_generator/diffusion_model.py` — Denoiser conditioned on:
  - Template backbone (injected as reference coordinates)
  - Catalytic constraint nodes (special graph nodes with constraint features)
  - Controllable noise level (how much deviation from template)
  - Optional fold family embedding
  - Supports both EGNN and IPA as the equivariant backbone (configurable)

### Phase 3: Scoring Models
Learned surrogates for Rosetta/PROSS, trained on cached Rosetta computations.

- Stability scorer (ΔΔG), packing scorer, desolvation scorer, activity proxy
- MLP architecture: `[input] → [512] → [256] → [128] → [1]` with GELU + dropout
- `MultiObjectiveScorer` combining all with configurable weights
- Training data generated by `scripts/data_prep/compute_rosetta_scores.py` and cached

### Phase 4: Sequence Generator
ProteinMPNN-style model for sequence design on generated/template backbones.

- 3 encoder + 3 decoder message passing layers
- Fixed catalytic residue mask
- Autoregressive decoding with temperature
- Cross-entropy loss on native sequences

### Phase 5: RL Optimization Loop
Connects all components.

- REINFORCE for backbone (geometry/feasibility reward)
- PPO for sequence (stability/activity reward)
- Separate credit assignment

---

## Caching Strategy

**Every expensive computation must be cached.** Cache is the foundation for future experiments.

```python
# Two-tier cache structure
cache/
├── esm_embeddings/          # Keyed by (sequence_hash)
│   └── {seq_hash}.pt        # Per-residue (L, 1280) + mean-pooled (1280,)
├── rosetta_features/        # Keyed by (pdb_id, chain, mutation_set)
│   └── {key_hash}.json      # All energy terms, per-residue + global
└── structure_features/      # Keyed by (pdb_id, chain)
    └── {key_hash}.pt        # SASA, SS, distances, etc.
```

Cache implementation in `src/utils/feature_cache.py`:
- Hash-based keys for deduplication
- Automatic save/load with torch and JSON
- Thread-safe for parallel computation scripts
- Metadata tracking (when computed, with what parameters)

---

## Relation to htFuncLib Paper

| htFuncLib Stage | Our Component |
|----------------|---------------|
| Phylogenetic filtering (PSSM) | Implicit in sequence generator's learned preferences |
| Rosetta ΔΔG scoring | Stability scoring model (learned surrogate) |
| Combinatorial enumeration | Backbone + sequence generators (sample directly) |
| EpiNNet classification | Multi-objective scorer (predicts quality directly) |
| Experimental FACS validation | Activity scorer refined with experimental feedback |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Start with | Conditioned backbone generator | Smaller search space, more reliable |
| SE(3) layers | Both EGNN and IPA | Implement both; EGNN simpler baseline, IPA more powerful (frame-aware); compare |
| Caching | Universal, hash-based | Every Rosetta/embedding run is expensive; cache everything for reuse |
| Backbone-sequence interface | `ProteinBackbone.to_graph()` | Clean separation |
| RL for backbone | REINFORCE | Continuous diffusion process |
| RL for sequence | PPO | Discrete autoregressive actions |
| Fold constraint | Template conditioning + noise level | Controls deviation from known fold |

---

## Verification Plan

1. **Phase 0-1**: Import all modules; create `ProteinBackbone` from test PDB; cache round-trip
2. **Phase 2**: Train backbone diffusion on PDB subset; verify generated backbones have valid bond geometry and low RMSD to template (< 2Å for constrained regions)
3. **Phase 3**: Train stability scorer; verify Pearson r > 0.8 with held-out Rosetta scores
4. **Phase 4**: Train sequence MPNN; verify sequence recovery > 30% on test backbones
5. **Phase 5**: RL loop increases reward over training; designs score better than random
6. **End-to-end**: Design conditioned on catalytic constraint; catalytic residue RMSD < 1Å

---

## Dependencies

```
torch, pytorch-lightning, torch-geometric  # Deep learning + graphs
fair-esm                                   # ESM-2 embeddings
biopython                                  # PDB parsing
pandas, numpy, scipy, scikit-learn         # Data + numerics
matplotlib, seaborn                        # Visualization
tqdm, pyyaml                               # Utilities
pyrosetta                                  # Score generation (optional, heavy)
```
