"""Universal caching system for embeddings, Rosetta scores, and structural features.

Every expensive computation (ESM embeddings, Rosetta energy calculations, structural
feature extraction) must go through this cache to ensure reusability across experiments.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for a cached computation."""
    created_at: str = ""
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # e.g., PDB ID, sequence hash

    def to_dict(self) -> dict:
        return {
            'created_at': self.created_at,
            'method': self.method,
            'params': self.params,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CacheMetadata':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _compute_hash(key_parts: Dict[str, Any]) -> str:
    """Compute a deterministic hash from key parts."""
    key_str = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class FeatureCache:
    """Thread-safe, hash-based feature cache with metadata tracking.

    Supports torch tensors (.pt), numpy arrays (.npy), and JSON-serializable data (.json).

    Usage:
        cache = FeatureCache('cache/esm_embeddings')

        # Check and retrieve
        key = {'sequence': 'MVSK...', 'model': 'esm2_650M'}
        if cache.has(key):
            embedding = cache.load(key)
        else:
            embedding = compute_embedding(sequence)
            cache.save(key, embedding, metadata=CacheMetadata(
                method='esm2', source='GFP_WT'
            ))
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _key_to_hash(self, key: Dict[str, Any]) -> str:
        return _compute_hash(key)

    def _path_for(self, key_hash: str, suffix: str) -> Path:
        return self.cache_dir / f"{key_hash}{suffix}"

    def has(self, key: Dict[str, Any]) -> bool:
        """Check if a key exists in cache."""
        h = self._key_to_hash(key)
        return any(
            self._path_for(h, suffix).exists()
            for suffix in ['.pt', '.npy', '.json']
        )

    def save(
        self,
        key: Dict[str, Any],
        data: Any,
        metadata: Optional[CacheMetadata] = None,
    ) -> Path:
        """Save data to cache.

        Args:
            key: Dictionary of key parts for hashing
            data: Data to cache (torch.Tensor, np.ndarray, or JSON-serializable)
            metadata: Optional metadata about the computation

        Returns:
            Path to the saved file.
        """
        h = self._key_to_hash(key)

        if metadata is None:
            metadata = CacheMetadata()
        if not metadata.created_at:
            metadata.created_at = datetime.now().isoformat()

        with self._lock:
            # Save data
            if isinstance(data, torch.Tensor):
                path = self._path_for(h, '.pt')
                torch.save(data, path)
            elif isinstance(data, np.ndarray):
                path = self._path_for(h, '.npy')
                np.save(path, data)
            else:
                path = self._path_for(h, '.json')
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            # Save metadata
            meta_path = self._path_for(h, '.meta.json')
            meta_dict = metadata.to_dict()
            meta_dict['key'] = key
            with open(meta_path, 'w') as f:
                json.dump(meta_dict, f, indent=2, default=str)

        logger.debug(f"Cached {h} → {path}")
        return path

    def load(self, key: Dict[str, Any]) -> Any:
        """Load data from cache.

        Args:
            key: Dictionary of key parts for hashing

        Returns:
            Cached data (torch.Tensor, np.ndarray, or dict).

        Raises:
            FileNotFoundError if key not in cache.
        """
        h = self._key_to_hash(key)

        pt_path = self._path_for(h, '.pt')
        if pt_path.exists():
            return torch.load(pt_path, weights_only=False)

        npy_path = self._path_for(h, '.npy')
        if npy_path.exists():
            return np.load(npy_path, allow_pickle=True)

        json_path = self._path_for(h, '.json')
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        raise FileNotFoundError(f"Key not found in cache: {key}")

    def load_metadata(self, key: Dict[str, Any]) -> Optional[CacheMetadata]:
        """Load metadata for a cached entry."""
        h = self._key_to_hash(key)
        meta_path = self._path_for(h, '.meta.json')
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            d = json.load(f)
        return CacheMetadata.from_dict(d)

    def delete(self, key: Dict[str, Any]) -> None:
        """Remove a cached entry."""
        h = self._key_to_hash(key)
        with self._lock:
            for suffix in ['.pt', '.npy', '.json', '.meta.json']:
                path = self._path_for(h, suffix)
                if path.exists():
                    path.unlink()

    def list_entries(self) -> list:
        """List all cached entries with their metadata."""
        entries = []
        for meta_path in sorted(self.cache_dir.glob('*.meta.json')):
            with open(meta_path) as f:
                meta = json.load(f)
            entries.append(meta)
        return entries

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(list(self.cache_dir.glob('*.meta.json')))

    def __repr__(self) -> str:
        return f"FeatureCache({self.cache_dir}, entries={len(self)})"


def get_sequence_hash(sequence: str) -> str:
    """Compute a hash for a protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]
