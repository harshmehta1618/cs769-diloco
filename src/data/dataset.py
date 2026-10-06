"""
Dataset and data-loading utilities.

Shard modes (controlled by DataConfig.shard_mode):
  "iid"          – randomly permuted equal-size shards (one per worker)
  "domain_skew"  – workers see different domains of the corpus
  "clone"        – all workers see identical data (Clone-DiLoCo control)
  "shuffle"      – independent shuffles of the same corpus

For the smoke test / local debugging, a "synthetic" corpus mode generates
random token tensors without any real data dependency.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Synthetic corpus (for smoke tests / CI)
# ---------------------------------------------------------------------------

class SyntheticTokenDataset(IterableDataset):
    """
    Infinite stream of random token sequences.
    Deterministic given (vocab_size, seq_len, worker_seed).
    """

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.seed       = seed

    def __iter__(self) -> Iterator[Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        while True:
            tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=rng)
            yield tokens


# ---------------------------------------------------------------------------
# Token-stream dataset (real corpora loaded via HuggingFace datasets)
# ---------------------------------------------------------------------------

class TokenStreamDataset(IterableDataset):
    """
    Streams pre-tokenized token arrays from a HuggingFace dataset column.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        A HuggingFace dataset with an "input_ids" column (list[int]).
    seq_len : int
        Context window length. The dataset packs examples back-to-back.
    seed : int
        Shuffle seed for this worker's shard.
    """

    def __init__(self, hf_dataset, seq_len: int, seed: int = 0) -> None:
        self._ds      = hf_dataset
        self.seq_len  = seq_len
        self._seed    = seed

    def __iter__(self) -> Iterator[Tensor]:
        rng = random.Random(self._seed)
        indices = list(range(len(self._ds)))
        rng.shuffle(indices)
        buf: list[int] = []
        while True:
            rng.shuffle(indices)
            for idx in indices:
                buf.extend(self._ds[idx]["input_ids"])
                while len(buf) >= self.seq_len + 1:
                    chunk = buf[: self.seq_len + 1]
                    buf   = buf[self.seq_len + 1 :]
                    yield torch.tensor(chunk, dtype=torch.long)


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

def build_worker_datasets(
    corpus: str,
    tokenizer_name: str,
    seq_len: int,
    M_workers: int,
    shard_mode: str,
    shard_seed: int,
    vocab_size: int,
    token_budget: int = 0,
) -> List[IterableDataset]:
    """
    Return a list of M datasets — one per worker.

    Parameters
    ----------
    corpus       : "synthetic" | "c4" | "dolma" | "fineweb"
    M_workers    : number of replicas (1 for SNOO/SW-AdamW)
    shard_mode   : "iid" | "domain_skew" | "clone" | "shuffle"
    shard_seed   : base seed; each worker offset by worker index
    """

    if corpus == "synthetic":
        datasets = []
        for w in range(M_workers):
            # clone mode → all workers share seed 0
            worker_seed = shard_seed if shard_mode == "clone" else shard_seed + w
            datasets.append(SyntheticTokenDataset(vocab_size, seq_len, seed=worker_seed))
        return datasets

    # Real corpus path
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install `datasets` to use real corpora: pip install datasets")

    if corpus == "c4":
        base_ds = load_dataset("c4", "en", split="train", streaming=True)
    elif corpus == "fineweb":
        base_ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    else:
        raise ValueError(f"Unsupported corpus: {corpus!r}")

    datasets = []
    for w in range(M_workers):
        worker_seed = shard_seed if shard_mode == "clone" else shard_seed + w
        datasets.append(TokenStreamDataset(base_ds, seq_len=seq_len, seed=worker_seed))

    return datasets


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a worker dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Batch unpacking helper
# ---------------------------------------------------------------------------

def unpack_batch(batch: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Given a batch of shape (B, seq_len+1), return:
      input_ids  : (B, seq_len)   — tokens 0..T-1
      targets    : (B, seq_len)   — tokens 1..T   (next-token labels)
    """
    batch = batch.to(device)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
