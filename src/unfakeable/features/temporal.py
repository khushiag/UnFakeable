"""Temporal feature encoder (LSTM, or GRU as the lighter alternative).

The encoder is a *sequence autoencoder* trained UNSUPERVISED on training-split embedding sequences only: labels are
never used, no classification head exists, and nothing is fitted on held-out data. Its final hidden state is the
temporal feature vector handed to the classical classifiers, which remain the sole decision makers.

With ``input_mode='diff'`` the encoder sees frame-to-frame embedding changes, so its latent focuses on temporal
dynamics (flicker, jitter, inconsistency) rather than on the static identity/content already captured spatially.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from unfakeable.config import TemporalCfg
from unfakeable.utils.device import resolve_device
from unfakeable.utils.log import get_logger

log = get_logger(__name__)


def _rnn(cell: str):
    if cell == "lstm":
        return nn.LSTM
    if cell == "gru":
        return nn.GRU
    raise ValueError(f"temporal.cell must be 'lstm' or 'gru', got {cell!r}")


class SeqAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, cell: str):
        super().__init__()
        self.cell = cell
        self.enc = _rnn(cell)(input_dim, hidden, layers, batch_first=True)
        self.dec = _rnn(cell)(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.enc(x)
        if self.cell == "lstm":
            h = h[0]
        return h[-1]

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        y, _ = self.dec(z.unsqueeze(1).expand(-1, x.shape[1], -1).contiguous())
        return self.out(y), z


def prepare_sequences(seqs: np.ndarray, input_mode: str) -> np.ndarray:
    if seqs.ndim != 3:
        raise ValueError(f"expected (N,T,D) sequences, got shape {seqs.shape}")
    if input_mode == "diff":
        if seqs.shape[1] < 3:
            raise ValueError("temporal encoding needs >= 3 frames per video")
        return np.diff(seqs, axis=1)
    if input_mode == "raw":
        if seqs.shape[1] < 2:
            raise ValueError("temporal encoding needs >= 2 frames per video")
        return seqs
    raise ValueError(f"temporal.input_mode must be 'diff' or 'raw', got {input_mode!r}")


class TemporalEncoder:
    def __init__(self, model: SeqAutoencoder, mean: np.ndarray, std: np.ndarray, meta: Dict, device: str = "cpu"):
        self.model, self.mean, self.std, self.meta = model.eval(), mean, std, meta
        self.device = torch.device(resolve_device(device))
        self.model.to(self.device)

    @property
    def dim(self) -> int:
        return int(self.meta["hidden_size"])

    @torch.inference_mode()
    def transform(self, seqs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        x = (prepare_sequences(seqs, self.meta["input_mode"]) - self.mean) / self.std
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size])).float().to(self.device)
            out.append(self.model.encode(xb).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def save(self, path) -> None:
        torch.save({"state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "mean": torch.from_numpy(self.mean), "std": torch.from_numpy(self.std), "meta": self.meta},
                   str(path))

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TemporalEncoder":
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        m = ck["meta"]
        model = SeqAutoencoder(m["input_dim"], m["hidden_size"], m["num_layers"], m["cell"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, ck["mean"].numpy(), ck["std"].numpy(), m, device)


def fit_temporal_encoder(seqs: np.ndarray, cfg: TemporalCfg, seed: int) -> Tuple[TemporalEncoder, List[float]]:
    """Fit on TRAINING sequences only (N,T,D). Returns the encoder and the per-epoch reconstruction loss."""
    x = prepare_sequences(seqs, cfg.input_mode).astype(np.float32)
    _rnn(cfg.cell)  # validate early
    d = x.shape[-1]
    mean = x.reshape(-1, d).mean(0)
    std = x.reshape(-1, d).std(0) + 1e-6
    xs = torch.from_numpy((x - mean) / std).float()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = torch.device(resolve_device(cfg.device))
    model = SeqAutoencoder(d, cfg.hidden_size, cfg.num_layers, cfg.cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs), batch_size=cfg.batch_size,
                                         shuffle=True, generator=gen)  # num_workers=0: safe on Windows
    history: List[float] = []
    model.train()
    for epoch in range(cfg.epochs):
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = torch.mean((recon - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach()) * len(xb)
            n += len(xb)
        history.append(total / max(n, 1))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == cfg.epochs:
            log.info("temporal %s epoch %d/%d recon_mse=%.5f", cfg.cell, epoch + 1, cfg.epochs, history[-1])
    meta = {"input_dim": d, "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers, "cell": cfg.cell,
            "input_mode": cfg.input_mode, "n_train_sequences": int(len(x))}
    return TemporalEncoder(model, mean.astype(np.float32), std.astype(np.float32), meta, cfg.device), history
