"""TransformerModel + an explicit input-bandwidth (cutoff) embedding.

The counterpart to the implicit multi-rate arm. There the cutoff is only inferable
from the input codes -- the spectrum goes quiet above it -- and the model must
work out which bandwidth it is looking at. Here it is told, via a learned
embedding indexed by the rate's position in `cutoff_rates`, added to the pooled
input exactly as the step embedding is.

Worth having both: if implicit matches explicit, the model infers bandwidth from
the codes and the embedding is unnecessary complexity; if explicit wins, the codes
alone are ambiguous and that is a fact about the representation. Either outcome is
reportable, and the pair is what makes it a controlled claim rather than a guess.

The base TransformerModel is untouched: a predictor trained without a rate
embedding behaves exactly as before, and scripts/train_bwe.py picks the class
from the `rate_emb` config switch.
"""
import torch
import torch.nn as nn

from .transformer import TransformerModel


class RateTransformerModel(TransformerModel):
    """Adds a learned embedding over a fixed set of input bandwidths."""

    def __init__(
        # EVERY inherited parameter listed EXPLICITLY. argbind binds only named
        # keyword parameters, so a (*args, **kwargs) signature would hide these and
        # silently fall back to the base defaults -- sum pooling and no input
        # LayerNorm, discarding the +0.0975 ViSQOL fix while the config appeared to
        # set it. That bug already happened once in CondTransformerModel.
        self,
        n_input_embs: int = 95,
        ntoken_input: int = 1024,
        ntoken_output: int = 1024,
        ninp: int = 1024,
        nhead: int = 8,
        nhid: int = 4096,
        nlayers: int = 6,
        n_pred_tokens: int = 48,
        dropout: float = 0.0,
        input_pool: str = "sum",
        emb_dim: int = 64,
        input_norm: bool = False,
        norm_first: bool = False,
        n_rates: int = 4,
    ):
        super().__init__(
            n_input_embs=n_input_embs, ntoken_input=ntoken_input,
            ntoken_output=ntoken_output, ninp=ninp, nhead=nhead, nhid=nhid,
            nlayers=nlayers, n_pred_tokens=n_pred_tokens, dropout=dropout,
            input_pool=input_pool, emb_dim=emb_dim, input_norm=input_norm,
            norm_first=norm_first)
        self.n_rates = n_rates
        self.rate_emb = nn.Embedding(n_rates, ninp) if n_rates > 0 else None
        if self.rate_emb is not None:
            nn.init.uniform_(self.rate_emb.weight, -0.1, 0.1)

    def forward(self, src: torch.Tensor, pred_step, rate_idx=None):
        if (rate_idx is None) != (self.rate_emb is None):
            raise ValueError(
                f"n_rates={self.n_rates} but rate_idx was "
                f"{'not ' if rate_idx is None else ''}supplied. The conditioning must "
                "be consistent between construction and every call, or rate_emb "
                "silently never receives gradient."
            )
        x = self.prepare_input(src)
        step = int(torch.as_tensor(pred_step).view(-1)[0].item())
        x = x + self.step_emb(
            torch.tensor([step], device=src.device)).view(1, 1, self.ninp)
        if self.rate_emb is not None:
            r = int(torch.as_tensor(rate_idx).view(-1)[0].item())
            if not 0 <= r < self.n_rates:
                raise ValueError(f"rate_idx {r} outside 0..{self.n_rates - 1}")
            x = x + self.rate_emb(
                torch.tensor([r], device=src.device)).view(1, 1, self.ninp)
        return self.output_heads[step](self.transformer(x)).transpose(0, 1)
