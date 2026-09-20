"""TransformerModel with an explicit embedding of the input cutoff.

Used by the "+ rate emb." ablation: instead of inferring the input bandwidth
from the codes, the model is told which rate it is seeing. `rate_idx` is the
position of the cutoff in `cutoff_rates`, and must be supplied on every call
(scripts/bwe/synthesize.py --rate_idx auto does this at synthesis time).

Selected by `rate_emb: true` in the config; a predictor without it behaves
exactly as TransformerModel.
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
