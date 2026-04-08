"""Structure quality and regression metrics."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict
from scipy import stats


@dataclass
class RegressionMetrics:
    """Comprehensive regression metrics."""

    @staticmethod
    def compute_all(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute all regression metrics."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-8))

        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
        }


def bond_geometry_metrics(coords: torch.Tensor) -> Dict[str, float]:
    """Evaluate backbone geometry quality.

    Args:
        coords: (L, 4, 3) backbone coordinates [N, CA, C, O]

    Returns:
        Dictionary of geometry quality metrics.
    """
    from src.utils.protein_constants import BOND_LENGTHS, CA_CA_DISTANCE

    L = coords.shape[0]

    # N-CA bond lengths
    d_n_ca = torch.norm(coords[:, 1] - coords[:, 0], dim=-1)
    # CA-C bond lengths
    d_ca_c = torch.norm(coords[:, 2] - coords[:, 1], dim=-1)
    # C-N peptide bonds
    if L > 1:
        d_c_n = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1)
    else:
        d_c_n = torch.tensor([])

    # CA-CA distances between consecutive residues
    if L > 1:
        d_ca_ca = torch.norm(coords[1:, 1] - coords[:-1, 1], dim=-1)
    else:
        d_ca_ca = torch.tensor([])

    metrics = {
        'n_ca_bond_mean': float(d_n_ca.mean()),
        'n_ca_bond_std': float(d_n_ca.std()),
        'n_ca_bond_deviation': float((d_n_ca - BOND_LENGTHS[('N', 'CA')]).abs().mean()),
        'ca_c_bond_mean': float(d_ca_c.mean()),
        'ca_c_bond_std': float(d_ca_c.std()),
        'ca_c_bond_deviation': float((d_ca_c - BOND_LENGTHS[('CA', 'C')]).abs().mean()),
    }

    if L > 1:
        metrics['c_n_peptide_mean'] = float(d_c_n.mean())
        metrics['c_n_peptide_std'] = float(d_c_n.std())
        metrics['c_n_peptide_deviation'] = float(
            (d_c_n - BOND_LENGTHS[('C', 'N')]).abs().mean()
        )
        metrics['ca_ca_distance_mean'] = float(d_ca_ca.mean())
        metrics['ca_ca_distance_std'] = float(d_ca_ca.std())
        metrics['ca_ca_distance_deviation'] = float(
            (d_ca_ca - CA_CA_DISTANCE).abs().mean()
        )

    return metrics


def clash_score(coords: torch.Tensor, threshold: float = 2.0) -> float:
    """Count steric clashes (CA atoms closer than threshold).

    Args:
        coords: (L, 4, 3) backbone coordinates
        threshold: Minimum allowed distance in Angstroms

    Returns:
        Fraction of CA pairs with distance < threshold (excluding neighbors).
    """
    ca = coords[:, 1]  # (L, 3)
    dist = torch.cdist(ca, ca)  # (L, L)
    L = ca.shape[0]

    # Mask diagonal and immediate neighbors
    mask = torch.ones(L, L, dtype=torch.bool, device=coords.device)
    mask.fill_diagonal_(False)
    for i in range(L - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False

    clashes = ((dist < threshold) & mask).sum().float()
    total = mask.sum().float()

    return float(clashes / max(total, 1))
