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
    Streams and packs tokens from a HuggingFace dataset (in-memory or streaming).

    Supports:
      - Datasets with "text" column (tokenizes on-the-fly with tokenizer_name)
      - Pre-tokenized datasets with "input_ids" column
      - Packing variable-length sequences into fixed-length (seq_len + 1) blocks
      - Infinite cyclic stream or bounded token stream

    Parameters
    ----------
    hf_dataset : datasets.Dataset or datasets.IterableDataset
        A HuggingFace dataset yielding dicts with "text" or "input_ids".
    tokenizer_name : str
        HuggingFace tokenizer identifier (e.g. "gpt2").
    seq_len : int
        Context window length. Emits chunks of size seq_len + 1.
    seed : int
        Shuffle seed for this worker's shard.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer_name: str = "gpt2",
        seq_len: int = 512,
        seed: int = 0,
    ) -> None:
        self._ds            = hf_dataset
        self.tokenizer_name = tokenizer_name
        self.seq_len        = seq_len
        self._seed          = seed
        self._tokenizer     = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore
                tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                self._tokenizer = tok
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer '{self.tokenizer_name}'. "
                    f"Ensure `transformers` is installed. Error: {e}"
                )
        return self._tokenizer

    def __iter__(self) -> Iterator[Tensor]:
        tokenizer = None
        buf: list[int] = []

        # Apply shuffle if streaming dataset supports it
        ds_iter = self._ds
        if hasattr(ds_iter, "shuffle"):
            try:
                ds_iter = ds_iter.shuffle(seed=self._seed, buffer_size=10_000)
            except Exception:
                pass

        while True:
            for item in ds_iter:
                if "input_ids" in item:
                    tok_ids = item["input_ids"]
                elif "text" in item:
                    if tokenizer is None:
                        tokenizer = self._get_tokenizer()
                    text = item["text"]
                    if not text:
                        continue
                    tok_ids = tokenizer.encode(text)
                    if tokenizer.eos_token_id is not None:
                        tok_ids.append(tokenizer.eos_token_id)
                else:
                    continue

                buf.extend(tok_ids)
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
    tokenizer_name : "gpt2" or huggingface tokenizer name
    seq_len      : sequence length
    M_workers    : number of replicas (1 for SNOO/SW-AdamW)
    shard_mode   : "iid" | "domain_skew" | "clone" | "shuffle"
    shard_seed   : base seed; each worker offset by worker index
    vocab_size   : vocabulary size
    token_budget : total training tokens
    """

    if corpus == "synthetic":
        datasets = []
        for w in range(M_workers):
            # clone mode -> all workers share identical seed
            worker_seed = shard_seed if shard_mode == "clone" else shard_seed + w
            datasets.append(SyntheticTokenDataset(vocab_size, seq_len, seed=worker_seed))
        return datasets

    # Real corpus path
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install `datasets` to use real corpora: pip install datasets")

    if corpus == "c4":
        dataset_name = "allenai/c4"
        dataset_subset = "en"
    elif corpus == "fineweb":
        dataset_name = "HuggingFaceFW/fineweb"
        dataset_subset = None
    else:
        raise ValueError(f"Unsupported corpus: {corpus!r}")

    datasets = []
    for w in range(M_workers):
        worker_seed = shard_seed if shard_mode == "clone" else shard_seed + w
        worker_shard_idx = 0 if shard_mode == "clone" else w

        # Load streaming split
        if dataset_subset:
            ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)

        # Worker sharding for streaming dataset
        if M_workers > 1 and shard_mode != "clone" and hasattr(ds, "shard"):
            try:
                ds = ds.shard(num_shards=M_workers, index=worker_shard_idx)
            except Exception:
                pass

        datasets.append(
            TokenStreamDataset(
                hf_dataset=ds,
                tokenizer_name=tokenizer_name,
                seq_len=seq_len,
                seed=worker_seed,
            )
        )

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
