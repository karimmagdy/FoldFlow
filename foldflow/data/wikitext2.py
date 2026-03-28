"""WikiText-2 data loading using HuggingFace datasets + GPT-2 tokenizer.

Provides IDENTICAL data pipeline for all models (fair comparison).
"""

import torch
from torch.utils.data import DataLoader, Dataset


class WikiText2Dataset(Dataset):
    """Pre-tokenized WikiText-2 dataset."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]  # input_ids, labels


def load_wikitext2(
    seq_len: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """Load WikiText-2 with GPT-2 tokenizer.

    Returns:
        train_loader, val_loader, vocab_size
    """
    import os
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    # Use local cache to avoid network hangs
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    vocab_size = tokenizer.vocab_size

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_split(split_name: str) -> torch.Tensor:
        texts = dataset[split_name]["text"]
        # Join all text, filter empty
        full_text = "\n".join(t for t in texts if t.strip())
        tokens = tokenizer.encode(full_text)
        return torch.tensor(tokens, dtype=torch.long)

    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")

    print(f"WikiText-2 train tokens: {len(train_tokens):,}")
    print(f"WikiText-2 val tokens: {len(val_tokens):,}")

    train_dataset = WikiText2Dataset(train_tokens, seq_len)
    val_dataset = WikiText2Dataset(val_tokens, seq_len)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, vocab_size
