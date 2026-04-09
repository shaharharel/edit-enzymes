"""Train scoring surrogates on detailed mutation scanning data.

Uses 43,878 mutations with 52 Rosetta energy fields to train:
1. Total ΔΔG predictor (main surrogate)
2. Per-term predictors (fa_rep, fa_sol, fa_elec, etc.)
3. Site-level ΔΔG predictor (local effects)

Input features: ESM-2 embedding at mutation site + mutation identity
Target: Rosetta energy terms from detailed scanning

Usage:
    python experiments/scoring/train_surrogates_on_mutations.py
    python experiments/scoring/train_surrogates_on_mutations.py --epochs 100 --device cuda
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils.feature_cache import FeatureCache, get_sequence_hash
from src.utils.metrics import RegressionMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SurrogateTrainingConfig:
    mutation_cache: str = 'cache/mutation_scanning_detailed'
    esm_cache: str = 'cache/esm_embeddings'
    pdb_dir: str = 'data/pdb'
    output_dir: str = 'results/surrogates'
    hidden_dims: list = None
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    dropout: float = 0.2
    patience: int = 15
    device: str = 'cuda'
    seed: int = 42
    ddg_clip: float = 500.0  # clip extreme ΔΔG values

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class SurrogateModel(nn.Module):
    """MLP surrogate for predicting Rosetta energy terms."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.GELU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_mutation_data(config: SurrogateTrainingConfig) -> list:
    """Load all mutation scanning results."""
    files = sorted(glob.glob(f'{config.mutation_cache}/*_mutations.json'))
    all_data = []
    for f in files:
        data = json.load(open(f))
        all_data.extend(data)
    logger.info(f"Loaded {len(all_data):,} mutations from {len(files)} proteins")
    return all_data


def load_esm_embeddings(config: SurrogateTrainingConfig, pdb_sequences: dict) -> dict:
    """Load ESM embeddings for all sequences."""
    cache = FeatureCache(config.esm_cache)
    embeddings = {}
    for pdb_id, seq in pdb_sequences.items():
        seq_hash = get_sequence_hash(seq)
        key = {'sequence_hash': seq_hash, 'model': 'esm2_t33_650M_UR50D'}
        if cache.has(key):
            data = cache.load(key)
            embeddings[pdb_id] = data[:-1]  # (L, 1280), drop mean-pooled
    logger.info(f"Loaded ESM embeddings for {len(embeddings)} proteins")
    return embeddings


def build_features(mutations: list, esm_embeddings: dict, config: SurrogateTrainingConfig):
    """Build feature matrix and target vectors.

    Features per mutation:
    - ESM embedding at mutation position (1280)
    - One-hot WT amino acid (20)
    - One-hot mutant amino acid (20)
    Total: 1320

    Targets:
    - ddg (total ΔΔG)
    - site_ddg (per-residue ΔΔG)
    - Individual energy term deltas (d_fa_rep, d_fa_sol, etc.)
    """
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    features = []
    targets_ddg = []
    targets_site_ddg = []
    target_terms = {}
    term_names = [k for k in mutations[0].keys() if k.startswith('d_') and not k.startswith('d_y')]

    for term in term_names:
        target_terms[term] = []

    skipped = 0
    for mut in mutations:
        pdb_id = mut['pdb_id']
        pos = mut['position']

        # Skip extreme outliers
        if abs(mut['ddg']) > config.ddg_clip:
            skipped += 1
            continue

        # Get ESM embedding at position
        if pdb_id not in esm_embeddings:
            skipped += 1
            continue
        emb = esm_embeddings[pdb_id]
        if pos >= emb.shape[0]:
            skipped += 1
            continue

        pos_emb = emb[pos].numpy() if isinstance(emb, torch.Tensor) else emb[pos]

        # One-hot encode WT and mutant
        wt_onehot = np.zeros(20)
        mut_onehot = np.zeros(20)
        if mut['wt_aa'] in aa_to_idx:
            wt_onehot[aa_to_idx[mut['wt_aa']]] = 1
        if mut['mut_aa'] in aa_to_idx:
            mut_onehot[aa_to_idx[mut['mut_aa']]] = 1

        feat = np.concatenate([pos_emb, wt_onehot, mut_onehot])
        features.append(feat)
        targets_ddg.append(mut['ddg'])
        targets_site_ddg.append(mut.get('site_ddg', 0.0))

        for term in term_names:
            target_terms[term].append(mut.get(term, 0.0))

    logger.info(f"Built features: {len(features)} samples, {len(features[0])} dims, skipped {skipped}")

    X = np.array(features, dtype=np.float32)
    y_ddg = np.array(targets_ddg, dtype=np.float32)
    y_site = np.array(targets_site_ddg, dtype=np.float32)
    y_terms = {k: np.array(v, dtype=np.float32) for k, v in target_terms.items()}

    return X, y_ddg, y_site, y_terms, term_names


def train_one_surrogate(
    X_train, y_train, X_val, y_val,
    config: SurrogateTrainingConfig,
    name: str,
) -> dict:
    """Train a single surrogate model."""
    device = config.device

    input_dim = X_train.shape[1]
    model = SurrogateModel(input_dim, 1, config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_dataset = TensorDataset(
        torch.tensor(X_train, device=device),
        torch.tensor(y_train, device=device).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, device=device),
        torch.tensor(y_val, device=device).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(y_batch.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_loss = float(np.mean((val_preds - val_trues) ** 2))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            metrics = RegressionMetrics.compute_all(val_trues.flatten(), val_preds.flatten())
            logger.info(
                f"  [{name}] epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, r={metrics['pearson_r']:.3f}"
            )

        if patience_counter >= config.patience:
            logger.info(f"  [{name}] Early stopping at epoch {epoch}")
            break

    # Final eval with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val, device=device)).cpu().numpy()
    metrics = RegressionMetrics.compute_all(y_val, val_preds.flatten())

    return {
        'model': model,
        'state_dict': best_state,
        'metrics': metrics,
        'best_val_loss': best_val_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--ddg-clip', type=float, default=500.0)
    parser.add_argument('--output-dir', type=str, default='results/surrogates')
    args = parser.parse_args()

    config = SurrogateTrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        ddg_clip=args.ddg_clip,
        output_dir=args.output_dir,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    mutations = load_mutation_data(config)

    # Get sequences for ESM lookup
    from src.data.pdb_loader import load_pdb
    pdb_sequences = {}
    for pdb_path in sorted(Path(config.pdb_dir).glob('*.pdb')):
        try:
            bb = load_pdb(str(pdb_path))
            if bb.sequence:
                pdb_sequences[pdb_path.stem.upper()] = bb.sequence
        except:
            pass

    esm_embeddings = load_esm_embeddings(config, pdb_sequences)

    # Build features
    X, y_ddg, y_site, y_terms, term_names = build_features(mutations, esm_embeddings, config)

    # Split: 80% train, 20% val (stratified by protein)
    pdb_ids = [m['pdb_id'] for m in mutations if abs(m['ddg']) <= config.ddg_clip and m['pdb_id'] in esm_embeddings]
    # Simple random split (TODO: protein-level split for generalization testing)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=config.seed)

    X_train, X_val = X[train_idx], X[val_idx]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Features: {X.shape[1]}")
    logger.info(f"Device: {config.device}")

    # Train surrogates
    results = {}

    # 1. Total ΔΔG
    logger.info("\n=== Training Total ΔΔG Surrogate ===")
    ddg_result = train_one_surrogate(
        X_train, y_ddg[train_idx], X_val, y_ddg[val_idx], config, 'total_ddg'
    )
    results['total_ddg'] = ddg_result
    torch.save(ddg_result['state_dict'], output_dir / 'surrogate_total_ddg.pt')
    logger.info(f"Total ΔΔG: r={ddg_result['metrics']['pearson_r']:.3f}, "
                f"RMSE={ddg_result['metrics']['rmse']:.2f}")

    # 2. Site-level ΔΔG
    logger.info("\n=== Training Site ΔΔG Surrogate ===")
    site_result = train_one_surrogate(
        X_train, y_site[train_idx], X_val, y_site[val_idx], config, 'site_ddg'
    )
    results['site_ddg'] = site_result
    torch.save(site_result['state_dict'], output_dir / 'surrogate_site_ddg.pt')
    logger.info(f"Site ΔΔG: r={site_result['metrics']['pearson_r']:.3f}, "
                f"RMSE={site_result['metrics']['rmse']:.2f}")

    # 3. Key individual energy terms
    key_terms = ['d_fa_rep', 'd_fa_atr', 'd_fa_sol', 'd_fa_elec', 'd_fa_dun',
                 'd_hbond_sc', 'd_hbond_bb_sc', 'd_rama_prepro']
    for term in key_terms:
        if term in y_terms:
            logger.info(f"\n=== Training {term} Surrogate ===")
            y_t = y_terms[term]
            # Clip extreme values for this term too
            clip_val = np.percentile(np.abs(y_t), 99)
            mask = np.abs(y_t) <= clip_val
            t_train = train_idx[mask[train_idx]]
            t_val = val_idx[mask[val_idx]]

            if len(t_train) > 100 and len(t_val) > 20:
                term_result = train_one_surrogate(
                    X[t_train], y_t[t_train], X[t_val], y_t[t_val], config, term
                )
                results[term] = term_result
                torch.save(term_result['state_dict'], output_dir / f'surrogate_{term}.pt')
                logger.info(f"{term}: r={term_result['metrics']['pearson_r']:.3f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SURROGATE TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Models saved to: {output_dir}")
    for name, res in results.items():
        m = res['metrics']
        logger.info(f"  {name}: r={m['pearson_r']:.3f}, R²={m['r2']:.3f}, RMSE={m['rmse']:.2f}")

    # Save summary
    summary = {name: res['metrics'] for name, res in results.items()}
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
