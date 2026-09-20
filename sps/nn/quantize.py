from typing import Union
from typing import List
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from sps.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1, bias = False)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1, bias = False)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        #print('input quantizer', z.size())
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        #print('input Nearest N', z_e.size())
        z_q, indices = self.decode_latents(z_e)
        #print('quantization', z_q.size())

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)
        #print('output_quantizer', z_q.size())

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):

        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 64,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
        quantizer_bypass_prob: float = 0.0
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.RVQ = nn.ModuleList([VectorQuantize(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)])

        # Fraction of the batch that gets a randomly truncated number of
        # active quantizers each step (SpectroStream Sec. 3.1.1). The rest
        # of the batch always uses all `n_codebooks` quantizers.
        self.quantizer_dropout = quantizer_dropout

        # Per-example probability of skipping the entire quantizer during
        # training (SpectroStream Sec. 3.1.2 / DAC): the decoder sees the
        # raw continuous encoder output for that example instead of the
        # quantized (straight-through) one, giving the encoder unaltered
        # gradients on some fraction of examples.
        self.quantizer_bypass_prob = quantizer_bypass_prob

    def _sample_truncation_levels(self, batch_size: int, device) -> torch.Tensor:
        """Sample a per-example truncation level r in {1, ..., R} using the
        biased ("quasi-exponentially decreasing") distribution from Sec. 3.1.1:
        levels in the first quarter of the range are 2x as likely as the
        second quarter, and 4x as likely as the second half.
        """
        R = self.n_codebooks
        r1 = max(1, R // 4)
        r2 = max(r1, R // 2)

        weights = torch.ones(R, device=device)
        weights[:r1] = 4.0
        weights[r1:r2] = 2.0
        # levels[r2:] keep weight 1.0

        levels = torch.multinomial(weights, batch_size, replacement=True) + 1  # 1-indexed
        return levels.float()

    def sample_rate_plan(self, batch_size: int, device):
        """Draw the per-example (truncation level, bypass) plan for one step.

        Factored out of forward() so a caller that owns several quantizers can
        draw ONE plan and apply it to all of them. In a residual cascade the
        branches must agree: the high branch codes the low branch's residual, so
        letting each sample its own level trains combinations that are never
        decoded (n_lf=2 with n_hf=31 spends 31 codebooks refining the residual of
        a nearly-destroyed low band) and makes the high branch's input
        distribution depend on a latent variable it is never told.

        Returns (n_quantizers, bypass), both (B,) float tensors, in exactly the
        form forward() would have produced for itself.
        """
        # Sentinel n_codebooks + 1 means "never truncated" (all quantizers active).
        n_quantizers = torch.full(
            (batch_size,), self.n_codebooks + 1, dtype=torch.float32, device=device
        )
        if self.quantizer_dropout > 0:
            sampled_levels = self._sample_truncation_levels(batch_size, device)
            n_dropout = int(batch_size * self.quantizer_dropout)
            n_quantizers[:n_dropout] = sampled_levels[:n_dropout]

        bypass = torch.zeros(batch_size, device=device)
        if self.quantizer_bypass_prob > 0:
            bypass = (torch.rand(batch_size, device=device) < self.quantizer_bypass_prob).float()

        return n_quantizers, bypass

    def rescale_plan(self, n_quantizers: torch.Tensor, from_n_codebooks: int):
        """Map a truncation plan drawn against `from_n_codebooks` onto this stack.

        A no-op when the two stacks are the same depth, which is the current
        configuration (32 + 32). Kept so an unequal split stays coherent rather
        than silently truncating to the wrong depth.
        """
        if from_n_codebooks == self.n_codebooks:
            return n_quantizers
        scaled = torch.round(n_quantizers * (self.n_codebooks / from_n_codebooks))
        # Preserve the "never truncated" sentinel, and never truncate below 1.
        never = n_quantizers > from_n_codebooks
        scaled = scaled.clamp(min=1.0)
        return torch.where(never, torch.full_like(scaled, self.n_codebooks + 1), scaled)

    def forward(self, z, *, n_quantizers=None, bypass=None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        Note: during training, a per-example truncation level is sampled
            according to `self.quantizer_dropout` (SpectroStream Sec. 3.1.1)
            and quantizers beyond that level are masked out of the sum and
            the losses for that example. All quantizers are always run
            forward (for simplicity/vectorization); only their contribution
            is masked.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """

        batch_size = z.shape[0]

        # An injected plan is authoritative and applies in eval() too: that is
        # what lets a caller impose one shared rate on both branches of a
        # cascade, and what makes inference-time rate control possible. When
        # nothing is injected, fall back to sampling our own -- training only,
        # exactly as before.
        if n_quantizers is None:
            n_quantizers, sampled_bypass = (
                self.sample_rate_plan(batch_size, z.device)
                if self.training
                else (
                    torch.full((batch_size,), self.n_codebooks + 1,
                               dtype=torch.float32, device=z.device),
                    torch.zeros(batch_size, device=z.device),
                )
            )
            if bypass is None:
                bypass = sampled_bypass
        elif bypass is None:
            bypass = torch.zeros(batch_size, device=z.device)

        z_q = 0
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        residual = z

        for i, quantizer in enumerate(self.RVQ):

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            mask = (i < n_quantizers).float() * (1.0 - bypass)  # (B,), 1.0 if this level is active

            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss = commitment_loss + (commitment_loss_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        # Bypassed examples get the raw (unquantized) encoder output instead,
        # so the decoder receives unaltered gradients for them.
        z_q = z_q + bypass[:, None, None] * z

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)
