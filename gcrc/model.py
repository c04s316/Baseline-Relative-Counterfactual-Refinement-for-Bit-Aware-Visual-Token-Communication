"""The lightweight masked-prior interface used by the GCR-C selector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .representation import unpatchify


class MaskedPrior(nn.Module):
    """Masked-token Transformer prior used by the reference experiments."""

    def __init__(self, vocab: int, n_tokens: int, dim: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.vocab = int(vocab)
        self.mask_id = int(vocab)
        self.token = nn.Embedding(vocab + 1, dim)
        self.position = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("token sequence is longer than the model position table")
        return self.norm(self.encoder(self.token(tokens) + self.position[:, : tokens.shape[1]]))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(tokens))


def masked_batch(tokens: torch.Tensor, vocab: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length = tokens.shape
    ratio = torch.empty(batch, 1, device=tokens.device).uniform_(0.25, 0.75, generator=generator)
    mask = torch.rand((batch, length), device=tokens.device, generator=generator) < ratio
    mask[:, 0] = True
    inputs = tokens.clone()
    inputs[mask] = vocab
    return inputs, mask


def train_masked_prior(
    tokens: torch.Tensor,
    vocab: int,
    epochs: int = 10,
    batch_size: int = 256,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> MaskedPrior:
    """Train a prior from caller-provided token tensors."""
    target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = MaskedPrior(vocab=vocab, n_tokens=tokens.shape[1]).to(target_device)
    loader = DataLoader(TensorDataset(tokens.long()), batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=target_device).manual_seed(seed + 19)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(target_device)
            inputs, mask = masked_batch(batch, vocab, generator)
            logits = model(inputs)
            loss = F.cross_entropy(logits[mask], batch[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def load_masked_prior(
    checkpoint: str | Path,
    device: torch.device | str | None = None,
    dim: int = 96,
    layers: int = 2,
    heads: int = 4,
) -> MaskedPrior:
    """Load a caller-supplied prior checkpoint without assuming a path layout."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    payload = torch.load(Path(checkpoint), map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or "position" not in state or "token.weight" not in state:
        raise ValueError("checkpoint must contain MaskedPrior state with token.weight and position")
    vocab = int(state["token.weight"].shape[0] - 1)
    n_tokens = int(state["position"].shape[1])
    inferred_dim = int(state["token.weight"].shape[1])
    model = MaskedPrior(vocab=vocab, n_tokens=n_tokens, dim=inferred_dim or dim, layers=layers, heads=heads)
    model.load_state_dict(state)
    return model.to(target_device).eval()


@torch.no_grad()
def batch_reconstruct(
    model: MaskedPrior,
    token_batch: torch.Tensor,
    selected_lists: Sequence[Sequence[int]],
    codebook: torch.Tensor,
) -> torch.Tensor:
    """One-pass deterministic reconstruction for candidate rollouts."""
    batch, n_tokens = token_batch.shape
    known = torch.zeros(batch, n_tokens, dtype=torch.bool, device=token_batch.device)
    for row, selected in enumerate(selected_lists):
        if selected:
            indices = torch.tensor(list(selected), dtype=torch.long, device=token_batch.device)
            known[row, indices] = True
    inputs = token_batch.clone()
    inputs[~known] = model.mask_id
    predicted = model(inputs).argmax(dim=-1)
    inputs[~known] = predicted[~known]
    return unpatchify(codebook.to(token_batch.device)[inputs].float())


@torch.no_grad()
def reconstruct(
    model: MaskedPrior,
    tokens: torch.Tensor,
    selected: Sequence[int],
    codebook: torch.Tensor,
    rounds: int = 1,
) -> torch.Tensor:
    """Reconstruct an image using deterministic confidence-based filling."""
    inputs = tokens.clone()
    known = torch.zeros_like(inputs, dtype=torch.bool)
    if selected:
        indices = torch.tensor(list(selected), dtype=torch.long, device=tokens.device)
        known[:, indices] = True
    for round_index in range(rounds):
        unknown = ~known
        if not unknown.any():
            break
        logits = model(inputs.masked_fill(~known, model.mask_id))
        probabilities = F.softmax(logits, dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        remaining = int(unknown.sum())
        fill = remaining if round_index == rounds - 1 else max(1, math.ceil(remaining / (rounds - round_index)))
        take = torch.topk(confidence.masked_fill(~unknown, -1), k=min(fill, remaining), dim=1).indices
        for index in take[0].tolist():
            inputs[0, index] = predictions[0, index]
            known[0, index] = True
    return unpatchify(codebook.to(tokens.device)[inputs].float())
