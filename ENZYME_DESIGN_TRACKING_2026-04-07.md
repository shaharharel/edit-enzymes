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

### Catalytic Sites — COMPLETE
- [x] 37/41 YAML constraint files in `data/catalytic_sites/`
- [x] 4 skipped due to chain/numbering mismatches

### PROSS Labels — COMPLETE (MPS)
- [x] `scripts/data_prep/compute_pross_labels.py` — ESM-2 pseudo-PSSM computation
- [x] 38 sequences processed, cached at `cache/pross_labels/`

### Rosetta Scoring — COMPLETE (local)
- [x] 41 structures scored in 90s, cached at `cache/rosetta_features/`

### Mutation Scanning — IN PROGRESS (local)
- [x] `scripts/data_prep/generate_mutation_scanning_data.py` — informative mutation sampling
- [ ] ~6K mutations (150/protein × 41), running locally
- [ ] Sampling strategy: 30% borderline PSSM, 25% active-site adjacent, 25% moderate conservation, 20% random
- [ ] Cache: `cache/mutation_scanning/`

---

## Foundation Model Integration (2026-04-08)

### RFdiffusion Wrapper — COMPLETE (code ready, needs install)
- [x] `src/models/backbone_generator/rfdiffusion_wrapper.py`
- [x] `RFdiffusionWrapper` with motif scaffolding and template-conditioned partial diffusion
- [ ] Install RFdiffusion to `external/RFdiffusion/`
- [ ] Download `ActiveSite_ckpt.pt` weights
- [ ] Test on one of our catalytic constraint YAMLs

### ProteinMPNN Wrapper — TESTED ON MPS
- [x] `src/models/sequence_generator/proteinmpnn_wrapper.py`
- [x] Installed to `external/ProteinMPNN/` with pretrained weights
- [x] Tested on MPS: 1TIM → 30% recovery, 2B3P → 75% recovery (real generalization)
- [x] No fine-tuning needed — fixed residue support built-in

### PROSS Scoring Components — COMPLETE
- [x] `src/models/scoring/pross_scorer.py` — separated from raw Rosetta scoring
  - PSSMScorer: phylogenetic conservation from ESM embeddings
  - PROSSDeltaGScorer: PROSS-style ΔΔG per mutation
  - MutationCompatibilityScorer: DeepSets-style epistasis predictor
  - PROSSCombinedScorer: integrates all three

---

## RL Strategy: Progressive Unfreezing (2026-04-09)

| Experiment | Trainable | Purpose |
|-----------|-----------|---------|
| **A**: Frozen generators | Search params (T, temp, noise) | Baseline: search alone |
| **B**: Unfreeze ProteinMPNN | Sequence weights | Does RL improve sequence design? |
| **C**: Unfreeze RFdiffusion | Backbone weights | Does RL improve structure generation? |
| **D**: Unfreeze both | All weights | Full end-to-end joint training |

---

## Pipeline Validation Results (2026-04-10)

End-to-end pipeline tested on 3 benchmark enzymes (no RL yet, just generate + score):

| Enzyme | PDB | Best Rosetta | RMSD Range | Recovery | Designs |
|--------|-----|-------------|-----------|----------|---------|
| KE07 (Kemp eliminase) | 2RKX | 381.0 REU | 1.48–2.23 Å | 9.2% | 120 |
| RA95 (Retro-aldolase) | 4A29 | 356.9 REU | 1.55–2.58 Å | 4.5% | 120 |
| GFP (htFuncLib) | 2B3P | 424.9 REU | 1.80–2.82 Å | 9.2% | 120 |

**Issues identified:**
- Surrogate scoring returns NaN (needs ESM embeddings of designed sequences, not just template)
- Rosetta scores positive (native ~-85 REU) → designs need Rosetta relax step
- Low sequence recovery → partial_T too high, backbone deviates too much

### Scoring Surrogates — TRAINED
- 10 models trained on 43,878 mutations across 41 proteins
- Best: d_fa_dun r=0.947, d_fa_atr r=0.903, d_fa_sol r=0.900
- Total ΔΔG: r=0.704

---

## Next Steps

1. **Fix surrogate integration**: compute ESM embeddings on-the-fly for designed sequences
2. **Add Rosetta relax**: relax designed structures before scoring
3. **Lower partial_T**: try 5-10 instead of 15 to stay closer to template
4. **RL implementation**: progressive unfreezing A→D
5. **PROSS labels from Sarel lab**: request real outputs
6. **Retrospective validation**: compare designs against known KE07→KE70 improvements

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
| 2026-04-08 | Use pretrained foundation models | 42 structures → memorization. RFdiffusion + ProteinMPNN trained on full PDB |
| 2026-04-08 | Separate PROSS from Rosetta scoring | PROSS = phylogenetic + energetic + combinatorial; Rosetta = raw physics energy |
| 2026-04-08 | Get real PROSS labels from Sarel lab | ESM-2 pseudo-PSSM is proxy; Sarel lab built PROSS and can provide real outputs |
| 2026-04-09 | Smart mutation sampling, not exhaustive | Borderline PSSM + active-site adjacent + conservation-weighted avoids trivial negatives |
| 2026-04-09 | RL progressive unfreezing (A→D) | Test frozen search → unfreeze seq → unfreeze backbone → both |
| 2026-04-09 | No fine-tuning of RFdiffusion/ProteinMPNN initially | Use out-of-the-box; partial_T + fixed residues already handle our use case |
