from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mingpt import GPT


@dataclass(frozen=True)
class TokenLMConfig:
    vocab_size: int
    block_size: int
    sos_token: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class TokenGPTAuthenticator(nn.Module):
    """
    Autoregressive token language model used for continuous authentication.

    Training:
      - fit on genuine windows only (one-class)
      - maximize likelihood p(tokens | user)

    Inference:
      - score(window) = -NLL(tokens)
      - threshold picked on val, applied to test
    """

    def __init__(self, cfg: TokenLMConfig):
        super().__init__()
        self.cfg = cfg
        self.gpt = GPT(
            vocab_size=cfg.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            embd_pdrop=cfg.embd_pdrop,
            resid_pdrop=cfg.resid_pdrop,
            attn_pdrop=cfg.attn_pdrop,
        )

    def _build_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens as (B, L), got {tokens.shape}")
        bsz, seq_len = tokens.shape
        if seq_len != self.cfg.block_size:
            raise ValueError(f"Token length {seq_len} != block_size {self.cfg.block_size}")
        sos = torch.full((bsz, 1), int(self.cfg.sos_token), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([sos, tokens[:, :-1]], dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._build_inputs(tokens)
        logits, _ = self.gpt(x)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))

    @torch.no_grad()
    def nll_per_sample(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self(tokens)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens.reshape(-1),
            reduction="none",
        ).view(tokens.shape[0], -1)
        return per_token.mean(dim=1)

    @torch.no_grad()
    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        # larger is more genuine
        return -self.nll_per_sample(tokens)


def build_token_lm(
    *,
    num_codebook_vectors: int,
    block_size: int,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    dropout: float = 0.1,
    sos_token: Optional[int] = None,
) -> Tuple[TokenGPTAuthenticator, TokenLMConfig]:
    vocab_size = int(num_codebook_vectors) + 1
    sos = int(vocab_size - 1 if sos_token is None else sos_token)
    cfg = TokenLMConfig(
        vocab_size=vocab_size,
        block_size=int(block_size),
        sos_token=sos,
        n_layer=int(n_layer),
        n_head=int(n_head),
        n_embd=int(n_embd),
        embd_pdrop=float(dropout),
        resid_pdrop=float(dropout),
        attn_pdrop=float(dropout),
    )
    return TokenGPTAuthenticator(cfg), cfg

