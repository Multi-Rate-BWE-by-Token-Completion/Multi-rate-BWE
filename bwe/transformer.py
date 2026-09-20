"""Transformer for bandwidth extension by token prediction.

Ported from the token-prediction transformer of our earlier work, itself derived
from DAC's `transup` model. The model is codec-agnostic -- it consumes and produces
integer RVQ codes -- so the port is mostly mechanical. Two things differ:

  * SpectroStream runs at 25 Hz rather than DAC's 100 Hz, so sequences are 4x
    shorter for the same audio duration but the RVQ stack is deeper for the same
    bitrate. n_pred_tokens is therefore typically larger here.
  * The original computed every output head and zeroed the inactive ones before
    summing (which keeps all parameters in the autograd graph, avoiding DDP
    unused-parameter errors). With more codebooks that wastes real compute -- each
    head is ninp x ntoken_output -- so this version selects the active head
    directly. If you later wrap this in DDP, pass find_unused_parameters=True.
"""
import math

import torch
import torch.nn as nn
from audiotools.ml import BaseModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, applied to [T, B, D] tensors."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x, scale):
        return self.dropout(x * scale + self.pe[: x.size(0), :])


class TransformerModel(BaseModel):
    """Predicts one RVQ codebook at a time from a stack of input codebooks.

    Input codes are pooled per codebook position, so the model sees a single
    sequence of length T regardless of how many codebooks are supplied. A learned
    step embedding tells it which codebook it is currently predicting, and a
    dedicated output head is used per step.

    Parameters
    ----------
    n_input_embs
        Number of input codebooks the model can be given: n_codebooks of the
        band-limited signal plus n_pred_tokens - 1 fed back predictions.
    input_pool
        "concat" (used by the paper) embeds each codebook into emb_dim dims,
        concatenates and projects to ninp. "sum" adds ninp-dim embeddings.
    input_norm
        True (used by the paper) applies a LayerNorm to the pooled embedding;
        False scales it by sqrt(ninp) instead.
    norm_first
        False is post-LN, True pre-LN with a final LayerNorm.

    The defaults are the legacy ones ("sum", no norm), so older checkpoints
    still load; the configs in conf/bwe/ set the values the paper used.
    """

    def __init__(
        self,
        n_input_embs: int = 47,
        ntoken_input: int = 1024,
        ntoken_output: int = 1024,
        ninp: int = 1024,
        nhead: int = 8,
        nhid: int = 4096,
        nlayers: int = 6,
        n_pred_tokens: int = 24,
        dropout: float = 0.0,
        input_pool: str = "sum",
        emb_dim: int = 64,
        input_norm: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        if input_pool not in ("sum", "concat"):
            raise ValueError(f"input_pool must be 'sum' or 'concat', got {input_pool!r}")
        # pre-LN (norm_first=True) needs the final LayerNorm below; post-LN does not.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout,
            norm_first=norm_first
        )
        self.norm_first = norm_first
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers,
            norm=nn.LayerNorm(ninp) if norm_first else None)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.n_input_embs = n_input_embs
        self.input_pool = input_pool
        self.emb_dim = emb_dim if input_pool == "concat" else ninp
        self.input_embs = nn.ModuleList(
            [nn.Embedding(ntoken_input, self.emb_dim) for _ in range(n_input_embs)]
        )
        # Built only for the non-legacy settings, so a legacy checkpoint's
        # state_dict still matches this module exactly.
        self.input_proj = (
            nn.Linear(n_input_embs * emb_dim, ninp) if input_pool == "concat" else None
        )
        self.input_ln = nn.LayerNorm(ninp) if input_norm else None
        self.step_emb = nn.Embedding(n_pred_tokens, ninp)
        self.output_heads = nn.ModuleList(
            [nn.Linear(ninp, ntoken_output, bias=False) for _ in range(n_pred_tokens)]
        )

        self.ninp = ninp
        self.n_pred_tokens = n_pred_tokens
        self.ntoken_input = ntoken_input
        self.ntoken_output = ntoken_output
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for emb in self.input_embs:
            nn.init.uniform_(emb.weight, -initrange, initrange)
        if self.input_proj is not None:
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        nn.init.uniform_(self.step_emb.weight, -initrange, initrange)
        for head in self.output_heads:
            nn.init.uniform_(head.weight, -initrange, initrange)

    def prepare_input(self, src: torch.Tensor) -> torch.Tensor:
        """src [B, N_CB, T] -> [T, B, ninp], pooling one embedding per codebook."""
        B, n_cb, T = src.size()
        if n_cb > self.n_input_embs:
            raise ValueError(
                f"{n_cb} input codebooks but only {self.n_input_embs} embedding "
                f"tables; raise TransformerModel.n_input_embs"
            )
        if self.input_pool == "sum":
            emb = None
            for q in range(n_cb):
                e = self.input_embs[q](src[:, q, :].long())
                emb = e if emb is None else emb + e
        else:
            # Codebooks beyond n_cb are levels this prediction step has not
            # reached yet. They stay zero, which gives the projection an explicit
            # "absent" signal -- the summed form cannot express that at all.
            parts = src.new_zeros((B, T, self.n_input_embs, self.emb_dim),
                                  dtype=self.input_proj.weight.dtype)
            for q in range(n_cb):
                parts[:, :, q] = self.input_embs[q](src[:, q, :].long())
            emb = self.input_proj(parts.reshape(B, T, -1))

        if self.input_ln is not None:
            emb = self.input_ln(emb)
            scale = 1.0
        else:
            scale = math.sqrt(self.ninp)
        emb = emb.permute(1, 0, 2).contiguous()
        return self.pos_encoder(emb, scale)

    def forward(self, src: torch.Tensor, pred_step) -> torch.Tensor:
        """Returns logits [B, T, ntoken_output] for codebook `pred_step`."""
        step = int(torch.as_tensor(pred_step).view(-1)[0].item())
        x = self.prepare_input(src)
        x = x + self.step_emb(
            torch.tensor([step], device=src.device)
        ).view(1, 1, self.ninp)
        enc = self.transformer(x)                       # [T, B, ninp]
        logits = self.output_heads[step](enc)           # [T, B, ntoken_output]
        return logits.transpose(0, 1)                   # [B, T, ntoken_output]
