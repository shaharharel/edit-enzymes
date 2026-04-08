# edit-enzymes: Progress Tracking

**Started**: 2026-04-07
**Status**: ALL PHASES COMPLETE — Full system implemented and smoke-tested

---

## Phase Overview

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0 | Scaffolding | ✅ COMPLETE | Directory structure, configs, CLAUDE.md |
| 1 | Data Representations + Caching | ✅ COMPLETE | Core types, cache infra |
| 2 | Backbone Generator | ✅ COMPLETE | Both EGNN (580K) and IPA (693K) |
| 3 | Scoring Models | ✅ COMPLETE | 4 surrogates + multi-objective |
| 4 | Sequence Generator | ✅ COMPLETE | ProteinMPNN-style (757K) |
| 5 | RL Optimization | ✅ COMPLETE | REINFORCE + PPO with credit assignment |

**Total**: 51 Python files, ~1.7M parameters (combined system)

---

## System Summary

| Component | Parameters | Architecture |
|-----------|-----------|--------------|
| Backbone Generator (EGNN) | 579,759 | SE(3)-equivariant diffusion, template-conditioned |
| Backbone Generator (IPA) | 692,884 | Frame-aware attention (AlphaFold2-style) |
| Sequence Generator | 756,500 | ProteinMPNN-style encoder-decoder with autoregressive sampling |
| Stability Scorer | 197,633 | MLP [input→512→256→128→1] |
| Packing Scorer | 197,633 | Same architecture |
| Desolvation Scorer | 197,633 | Same architecture |
| Activity Scorer | 197,633 | Same architecture |

---

## Completed Phases

### Phase 0: Scaffolding
- [x] Directory structure, CLAUDE.md, requirements.txt, .gitignore, all __init__.py, experiment configs

### Phase 1: Data Representations + Caching
- [x] `src/utils/protein_constants.py` — AA vocab, atom types, bond lengths
- [x] `src/utils/feature_cache.py` — Thread-safe hash-based caching
- [x] `src/utils/logging.py` — Logger setup
- [x] `src/utils/metrics.py` — Regression + structure quality metrics
- [x] `src/data/catalytic_constraints.py` — CatalyticResidue, CatalyticConstraint, ActiveSiteSpec, YAML loader
- [x] `src/data/protein_structure.py` — ProteinBackbone, ProteinGraph with k-NN graph builder
- [x] `src/data/pdb_loader.py` — BioPython PDB parsing
- [x] `src/data/dataset_builders.py` — BackboneDiffusionDataset, SequenceDesignDataset, ScoringDataset

### Phase 2: Backbone Generator
- [x] `src/utils/geometry.py` — rigid_from_3_points, kabsch_rmsd, bond_length_loss
- [x] `src/utils/so3_utils.py` — SO(3)/R3 diffusion, axis-angle conversions
- [x] `src/models/layers/egnn.py` — EGNNLayer + EGNNStack
- [x] `src/models/layers/invariant_point_attention.py` — IPA + IPAStack
- [x] `src/models/layers/protein_graph_conv.py` — Edge-conditioned message passing
- [x] `src/models/backbone_generator/noise_schedule.py` — Linear/cosine/polynomial schedules
- [x] `src/models/backbone_generator/base.py` — AbstractBackboneGenerator
- [x] `src/models/backbone_generator/diffusion_model.py` — SE3BackboneDiffusion (configurable EGNN/IPA)
- [x] `experiments/backbone/configs.py` + `train_backbone_diffusion.py`

### Phase 3: Scoring Models
- [x] `src/models/scoring/base.py` — AbstractScoringModel with shared MSE training
- [x] `src/models/scoring/stability_scorer.py` — ΔΔG surrogate
- [x] `src/models/scoring/packing_scorer.py` — Packing quality
- [x] `src/models/scoring/desolvation_scorer.py` — Desolvation
- [x] `src/models/scoring/activity_scorer.py` — Activity proxy
- [x] `src/models/scoring/multi_objective.py` — MultiObjectiveScorer with configurable weights
- [x] `scripts/data_prep/compute_rosetta_scores.py` — Rosetta data generation (with PyRosetta fallback)
- [x] `experiments/scoring/configs.py` + `train_scoring_models.py`

### Phase 4: Sequence Generator
- [x] `src/models/sequence_generator/graph_features.py` — Enhanced graph features (CB, orientations)
- [x] `src/models/sequence_generator/base.py` — AbstractSequenceGenerator
- [x] `src/models/sequence_generator/mpnn_model.py` — ProteinMPNNModel with autoregressive decoding
- [x] `experiments/sequence/configs.py` + `train_sequence_mpnn.py`

### Phase 5: RL Optimization
- [x] `src/models/rl/reward.py` — RewardFunction with separated backbone/sequence credit
- [x] `src/models/rl/backbone_policy.py` — REINFORCE wrapper + value baseline
- [x] `src/models/rl/sequence_policy.py` — PPO wrapper + value head
- [x] `src/models/rl/ppo_trainer.py` — RLTrainer with rollout buffer, GAE, full training loop
- [x] `experiments/rl/configs.py` + `run_rl_finetuning.py`

---

## Data & Embeddings (2026-04-08)

### PDB Download — COMPLETE
- [x] Downloaded 41 PDB structures (0 failures)
- [x] 2B3P (GFP), 9 TIM barrels, 5 serine proteases, 2 cysteine proteases
- [x] 4 lipases, 6 glycosidases, 3 oxidoreductases, 2 kinases, 3 metalloenzymes
- [x] 3 designed enzymes (KE07, KE70, RA95), 3 classic textbook enzymes
- [x] `scripts/data_prep/download_pdbs.py` with full enzyme catalog

### ESM-2 Embeddings — COMPLETE (MPS)
- [x] 38 unique sequences embedded with esm2_t33_650M_UR50D (650M params)
- [x] Per-residue (L, 1280) + mean-pooled (1280,) cached per sequence
- [x] Total time: 10.3s on MPS
- [x] Cache: `cache/esm_embeddings/` (38 entries)

### Structural Features — COMPLETE (CPU)
- [x] 41 structures processed: CA distances, contact maps, local geometry, SASA proxy, SS proxy
- [x] All structures: 0 clashes, N-CA bonds ~1.46Å (ideal)
- [x] Cache: `cache/structure_features/` (41 entries)

### Catalytic Sites — PENDING
- [ ] Extract catalytic residue info and create YAML constraint files

### Rosetta Scoring — PENDING (tonight when CPU freed)
- [ ] Run `scripts/data_prep/compute_rosetta_scores.py` on all 41 structures

---

## Next Steps

1. **Catalytic sites**: Extract M-CSA catalytic residues → YAML constraints
2. **Rosetta scores**: Generate scoring model training data (tonight)
3. **Supervised pretraining**: Train backbone diffusion + sequence MPNN
4. **Train scoring surrogates**: Verify Pearson r > 0.8 vs held-out Rosetta
5. **RL fine-tuning**: Run full optimization loop
6. **TIM barrel case study**: Apply to a specific enzyme design task

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-07 | Start with conditioned backbone generator | Template → smaller search space |
| 2026-04-07 | Implement both EGNN and IPA | Compare distance-based vs frame-aware |
| 2026-04-07 | Cache everything | Reusable across experiments |
| 2026-04-07 | Center coords before EGNN | Numerical stability |
| 2026-04-07 | Clamp displacement to ±10Å | Prevents explosion in untrained models |
| 2026-04-08 | REINFORCE for backbone, PPO for sequence | Matches continuous vs discrete action spaces |
| 2026-04-08 | Separated credit assignment | Backbone gets geometry reward, sequence gets stability/activity |
