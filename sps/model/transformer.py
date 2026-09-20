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

    Two aspects of the pooling are configurable. BOTH DEFAULT TO THE LEGACY
    BEHAVIOUR so that checkpoints trained before they existed still load.

    input_pool
        "sum" (legacy) embeds each codebook into ninp dims and adds them. With
        n_input_embs=47 that superposes 47 symbols in one 1024-d vector, leaving
        each codebook roughly 1024/47 = 22 effective dimensions to separate 1024
        entries.

        "concat" embeds each codebook into emb_dim dims, concatenates, and
        projects to ninp. Summing is the special case where that projection is a
        tiled identity, so this is strictly more expressive -- and at emb_dim=64
        it is also SMALLER: 47*1024*64 + (47*64)*1024 = 6.2M against the 49M the
        summed tables use.

    input_norm
        False (legacy) scales the pooled embedding by sqrt(ninp) before adding
        the positional encoding, which is the "Attention is All You Need" recipe.
        That recipe assumes ONE embedding initialised at std ~ d^-0.5. Here 47
        tables initialised at uniform(-0.1, 0.1) are summed first, so the pooled
        content arrives at rms 0.0577*sqrt(47)*32 = 12.7 against the positional
        encoding's 0.707 -- a ratio of 18:1 where the recipe intends 1.4:1. `pe`
        is registered with requires_grad=False, so the model cannot amplify it;
        it can only shrink all 47 tables together, against the content pathway.
        Whatever position information layer 1 fails to resolve is unrecoverable,
        which is a candidate explanation for how little time context has bought
        (5.36% top-1 at 63 frames against 6.35% at 251).

        True applies a LayerNorm after pooling and drops the scale to 1.0. That
        also removes a second-order wart: the sum runs over 24+k embeddings at
        prediction step k, so the legacy input norm drifts by sqrt(47/24) = 1.40
        across the RVQ depth, and PyTorch's TransformerEncoderLayer is post-norm
        so nothing rescales it before the first attention.
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
        # norm_first=False (legacy) is POST-LN, which nn.TransformerEncoderLayer
        # defaults to and which nothing here previously overrode. Post-LN puts the
        # residual stream outside the LayerNorm, so activation magnitude compounds
        # with depth and the gradient through the early layers is scaled by the
        # product of the later layers' Jacobians. At 6 layers it trains; at 12 it
        # is the standard divergence case, and it is why the original Transformer
        # needed a warmup schedule. A deeper model in HP-codecX collapsed, which
        # is consistent with this being the cause rather than anything about the
        # data.
        #
        # norm_first=True is PRE-LN: each sublayer normalises its input and the
        # residual stream stays unnormalised end to end, so gradients reach layer
        # 0 without that compounding factor and depth is stable without warmup.
        # It needs a FINAL LayerNorm, because the last block's output never passes
        # through one otherwise -- nn.TransformerEncoder leaves norm=None if not
        # given, which for pre-LN would emit an unnormalised residual stream.
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
