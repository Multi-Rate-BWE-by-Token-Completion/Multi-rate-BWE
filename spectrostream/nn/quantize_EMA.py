from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from spectrostream.nn.layers import WNConv1d


def _flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, "b d t -> (b t) d")


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        x.pow(2).sum(dim=1, keepdim=True)
        - 2 * x @ y.t()
        + y.pow(2).sum(dim=1, keepdim=True).t()
    )


def _sample_rows(rows: torch.Tensor, max_rows: Optional[int]) -> torch.Tensor:
    if max_rows is None or rows.size(0) <= max_rows:
        return rows
    indices = torch.randperm(rows.size(0), device=rows.device)[:max_rows]
    return rows.index_select(0, indices)


def _run_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    num_iters: int,
) -> torch.Tensor:
    if points.size(0) == 0:
        raise ValueError("Cannot initialize EMA codebook from an empty batch")

    if points.size(0) >= num_clusters:
        init_indices = torch.randperm(points.size(0), device=points.device)[:num_clusters]
        centroids = points.index_select(0, init_indices).clone()
    else:
        # Fewer data points than codebook entries (e.g. batch 16 x 33 frames =
        # 528 points for a 1024-entry codebook), so some centroids must reuse the
        # same point. Exact duplicates are permanently dead under EMA: identical
        # centroids give identical distances, argmin always resolves to the first,
        # so the copies are never assigned and never move. Jitter them apart so
        # every entry is at least reachable. Without this, measured codebook usage
        # sat at ~330-610 of 1024 per level -- almost exactly the point count.
        repeat = (num_clusters + points.size(0) - 1) // points.size(0)
        centroids = points.repeat(repeat, 1)[:num_clusters].clone()
        scale = centroids.std() if centroids.numel() > 1 else torch.ones((), device=points.device)
        centroids = centroids + 0.01 * scale * torch.randn_like(centroids)

    for _ in range(num_iters):
        distances = _euclidean_distance(points, centroids)
        assignments = distances.argmin(dim=1)

        counts = torch.bincount(assignments, minlength=num_clusters).to(points.dtype)
        summed = torch.zeros_like(centroids)
        summed.index_add_(0, assignments, points)

        updated = centroids.clone()
        active = counts > 0
        updated[active] = summed[active] / counts[active].unsqueeze(1)
        centroids = updated

    return centroids


class VectorQuantizeEMA(nn.Module):
    """Vector quantizer with EMA-updated codebook and optional k-means warm start."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        kmeans_max_points: Optional[int] = 4096,
        drop_unused_after_steps: int = 0,
        dead_code_threshold: float = 1.0,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.kmeans_max_points = kmeans_max_points
        self.drop_unused_after_steps = drop_unused_after_steps
        self.dead_code_threshold = dead_code_threshold

        # L2-normalizing latents and codebook before the distance computation
        # turns nearest-neighbour lookup into cosine similarity (the ViT-VQGAN /
        # "Improved VQGAN" trick, inherited here via DAC). SpectroStream cites
        # SoundStream for its RVQ and uses neither that nor factorized codes, so
        # this defaults off. It is also actively inconsistent with EMA training:
        # assignment would happen in normalized space while the EMA centroids --
        # and the returned z_q, and the commitment loss -- all live in
        # unnormalized space.
        self.l2_normalize = l2_normalize

        # Mirrors the `initialized` buffer as a plain Python bool purely to avoid
        # a GPU->CPU sync. Reading the buffer with .item() on every forward
        # stalls the pipeline once per quantizer, which at 64 quantizers is 64
        # syncs per step. The buffer remains the source of truth across
        # checkpoint load (see _sync_init_flag below).
        self._initialized_cache = False

        # k-means needs at least as many data points as it has clusters, or it
        # must reuse points and the duplicate centroids stay tied forever
        # (identical distances -> argmin always picks the first -> never
        # assigned, never updated). One batch does not supply enough: at batch 16
        # a 1.28 s example yields ~9 latent frames, so ~144 vectors against a
        # 1024-entry codebook. The paper's batch of 128 yields ~4096, comfortably
        # more than the codebook, which is why it does not hit this.
        #
        # Measured at codebook_dim=256, initializing from a single batch left
        # only 4-5 of 1024 codes reachable. So accumulate latents across the
        # first few steps and initialize once there are genuinely enough. Costs
        # a couple of dozen steps of quantizing against the random init.
        self._init_buffer = []
        self._init_points_needed = min(
            kmeans_max_points if kmeans_max_points else 2 * codebook_size,
            2 * codebook_size,
        )

        if input_dim != codebook_dim:
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1, bias=False)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1, bias=False)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook.weight.requires_grad_(False)

        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    # These three methods only mutate registered buffers under torch.no_grad()
    # (k-means init, EMA codebook update, dead-code recycling) -- no gradient
    # ever flows back through them into the compiled tensor graph. But they do
    # data-dependent branching on `.item()` and ops dynamo can't trace under
    # dynamic shapes (F.one_hot, boolean-mask scatter into buffers), which
    # crashes torch.compile(generator) with "Cannot call sizes() on tensor
    # with symbolic sizes/strides" -- the old gradient-based RVQ had no
    # EMA-update step at all, so compiling SpS never exercised this before.
    # torch._dynamo.disable() makes dynamo treat each as an opaque eager call
    # (a graph break at the call site) instead of trying to trace into it,
    # while the surrounding forward (in_proj/decode_latents/out_proj) still compiles.
    def _sync_init_flag(self) -> None:
        """Refresh the Python-side cache from the buffer (one sync, not per-step).

        Called once lazily; also picks up the restored value after a checkpoint
        load, so resuming a run does not re-run k-means over the codebook.
        """
        self._initialized_cache = bool(self.initialized.item())

    @torch._dynamo.disable()
    def _init_codebook(self, latents: torch.Tensor) -> None:
        if not self.kmeans_init:
            return
        # Fast path: a plain Python bool, so the common case (already
        # initialized, i.e. every step after the first) costs no GPU sync.
        if self._initialized_cache:
            return
        # Cache says uninitialized -- confirm against the buffer, which may have
        # been restored from a checkpoint since this module was constructed.
        self._sync_init_flag()
        if self._initialized_cache:
            return

        # Accumulate across steps until there are enough points to seed every
        # centroid distinctly (see _init_points_needed above).
        self._init_buffer.append(latents.detach())
        n_points = sum(t.shape[0] for t in self._init_buffer)
        if n_points < self._init_points_needed:
            return

        with torch.no_grad():
            points = torch.cat(self._init_buffer, dim=0)
            self._init_buffer = []          # release; init happens exactly once
            centroids = _run_kmeans(
                _sample_rows(points, self.kmeans_max_points),
                self.codebook_size,
                self.kmeans_iters,
            )
            # Under DDP every rank runs k-means on its own accumulated points and
            # would otherwise land on different centroids, permanently desyncing
            # the codebooks. Take rank 0's. Safe from deadlock because every rank
            # sees the same batch size, so they all reach this line on the same
            # step. (Which rank's data seeds the init is immaterial.)
            if _ddp_active():
                dist.broadcast(centroids, src=0)
            self.codebook.weight.data.copy_(centroids)
            self.ema_codebook.data.copy_(centroids)
            self.ema_cluster_size.data.fill_(1.0)
            self.initialized.fill_(True)
            self._initialized_cache = True

    @torch._dynamo.disable()
    def _update_ema(self, latents: torch.Tensor, indices: torch.Tensor) -> None:
        with torch.no_grad():
            encodings = F.one_hot(indices.reshape(-1), self.codebook_size).type_as(latents)
            cluster_size = encodings.sum(dim=0)
            embed_sum = encodings.t() @ latents

            # The codebook is trained by EMA, not by gradients, so DDP does
            # nothing for it: `codebook.weight` is a Parameter with
            # requires_grad=False (never gradient-synced) and the ema_* buffers
            # would just be overwritten by rank 0 under broadcast_buffers. Sum
            # the sufficient statistics here instead, so every rank applies the
            # identical global update and the codebooks stay bit-identical --
            # and, unlike buffer broadcasting, every rank's data contributes.
            if _ddp_active():
                dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
            self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

            total_count = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (total_count + self.codebook_size * self.epsilon)
                * total_count.clamp_min(self.epsilon)
            )
            updated_codebook = self.ema_codebook / smoothed_cluster_size.unsqueeze(1).clamp_min(self.epsilon)
            self.codebook.weight.data.copy_(updated_codebook)

    @torch._dynamo.disable()
    def _recycle_dead_codes(self, latents: torch.Tensor) -> None:
        if self.drop_unused_after_steps <= 0:
            return
        if int(self.step.item()) < self.drop_unused_after_steps:
            return

        dead_mask = self.ema_cluster_size < self.dead_code_threshold
        if not bool(dead_mask.any().item()):
            return

        with torch.no_grad():
            candidate_count = int(dead_mask.sum().item())
            candidates = _sample_rows(latents, candidate_count)
            if candidates.size(0) == 0:
                return
            if candidates.size(0) < candidate_count:
                repeat = (candidate_count + candidates.size(0) - 1) // candidates.size(0)
                candidates = candidates.repeat(repeat, 1)
            candidates = candidates[:candidate_count]

            self.codebook.weight.data[dead_mask] = candidates
            self.ema_codebook.data[dead_mask] = candidates
            self.ema_cluster_size.data[dead_mask] = self.dead_code_threshold

    def forward(self, z: torch.Tensor, update_ema: bool = True):
        z_e = self.in_proj(z)
        latents = _flatten_latents(z_e)

        self._init_codebook(latents)

        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q.detach(), z_e.detach(), reduction="none").mean([1, 2])

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        if self.training and update_ema:
            self._update_ema(latents, indices)
            self._recycle_dead_codes(latents)
            self.step.add_(1)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encodings = _flatten_latents(latents)
        codebook = self.codebook.weight

        if self.l2_normalize:
            encodings = F.normalize(encodings, dim=1)
            codebook = F.normalize(codebook, dim=1)

        distances = _euclidean_distance(encodings, codebook)
        indices = rearrange((-distances).max(dim=1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantizeEMA(nn.Module):
    """Residual vector quantizer made of EMA-updated quantizer blocks."""

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 64,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        kmeans_max_points: Optional[int] = 4096,
        drop_unused_after_steps: int = 0,
        dead_code_threshold: float = 1.0,
        quantizer_dropout: float = 0.0,
        quantizer_bypass_prob: float = 0.0,
        l2_normalize: bool = False,
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        # Fraction of the batch that gets a randomly truncated number of
        # active quantizers each step (SpectroStream Sec. 3.1.1). Mirrors
        # ResidualVectorQuantize's masking exactly; only the RVQ-level
        # contribution to z_q/losses is masked -- each VectorQuantizeEMA's own
        # EMA codebook update below is unconditional (all examples keep the
        # codebook "warm" regardless of whether their output was dropped).
        self.quantizer_dropout = quantizer_dropout

        # Per-example probability of skipping the entire quantizer during
        # training (SpectroStream Sec. 3.1.2 / DAC): bypassed examples get the
        # raw encoder output added back in instead of the quantized one.
        self.quantizer_bypass_prob = quantizer_bypass_prob

        self.RVQ = nn.ModuleList(
            [
                VectorQuantizeEMA(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    decay=decay,
                    epsilon=epsilon,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    kmeans_max_points=kmeans_max_points,
                    drop_unused_after_steps=drop_unused_after_steps,
                    dead_code_threshold=dead_code_threshold,
                    l2_normalize=l2_normalize,
                )
                for _ in range(n_codebooks)
            ]
        )

    def _sample_truncation_levels(self, batch_size: int, device) -> torch.Tensor:
        # Same biased ("quasi-exponentially decreasing") distribution as
        # ResidualVectorQuantize._sample_truncation_levels (SpectroStream
        # Sec. 3.1.1): levels in the first quarter of the range are 2x as
        # likely as the second quarter, and 4x as likely as the second half.
        R = self.n_codebooks
        r1 = max(1, R // 4)
        r2 = max(r1, R // 2)

        weights = torch.ones(R, device=device)
        weights[:r1] = 4.0
        weights[r1:r2] = 2.0

        levels = torch.multinomial(weights, batch_size, replacement=True) + 1  # 1-indexed
        return levels.float()

    def sample_rate_plan(self, batch_size: int, device):
        """Draw the per-example (truncation level, bypass) plan for one step.

        Factored out of forward() so a caller owning several quantizers can draw
        ONE plan and apply it to all of them. In a residual cascade the branches
        must agree: the high branch codes the low branch's residual, so letting
        each sample its own level trains combinations that are never decoded
        (n_lf=2 with n_hf=31 spends 31 codebooks refining the residual of a
        nearly-destroyed low band) and makes the high branch's input distribution
        depend on a latent variable it is never told.

        Returns (n_quantizers, bypass), both (B,) float tensors, in exactly the
        form forward() would have produced for itself.
        """
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
        never = n_quantizers > from_n_codebooks
        scaled = scaled.clamp(min=1.0)
        return torch.where(never, torch.full_like(scaled, self.n_codebooks + 1), scaled)

    def from_codes(self, codes: torch.Tensor):
        """Reconstruct the quantized latent from integer codes alone.

        codes : Tensor[B x N x T]  (N <= n_codebooks; a prefix of the RVQ stack,
                                    which is what truncating the bitrate means)
        returns Tensor[B x D x T]

        Needed to turn *predicted* tokens back into audio -- forward() only ever
        produces z_q as a by-product of encoding real input, so there was no path
        from codes to a waveform.
        """
        z_q = 0.0
        for i in range(codes.shape[1]):
            q = self.RVQ[i]
            z_q_i = q.out_proj(q.decode_code(codes[:, i, :]))
            z_q = z_q + z_q_i
        return z_q

    def forward(self, z: torch.Tensor, *, n_quantizers=None, bypass=None,
                passthrough: bool = True, update_ema: bool = True):
        """passthrough=False omits the `+ bypass * z` term at the end.

        update_ema=False suppresses the EMA codebook update and the dead-code
        recycling for this call. ParallelHPQuantizer needs it when it runs an
        INACTIVE section: that section is executed only so its parameters
        receive a (zero) gradient and DDP's reducer does not stall, and its
        codebooks must not absorb statistics from a component they are not
        meant to model.

        Only ParallelHPQuantizer passes False for passthrough. It runs two of these stacks over
        the same latent and sums them, so with the default each bypassed example
        would receive the raw encoder output TWICE (2*z instead of z). The
        parallel wrapper adds the term once itself. Default True keeps every
        existing single-stack caller bit-identical.
        """
        batch_size = z.shape[0]

        # An injected plan is authoritative and applies in eval() too: that is
        # what lets a caller impose one shared rate on both branches of a
        # cascade, and what makes inference-time rate control possible. When
        # nothing is injected, fall back to sampling our own -- training only,
        # exactly as before. Sentinel n_codebooks + 1 means "never truncated".
        if n_quantizers is None:
            if self.training:
                n_quantizers, sampled_bypass = self.sample_rate_plan(batch_size, z.device)
            else:
                n_quantizers = torch.full(
                    (batch_size,), self.n_codebooks + 1, dtype=torch.float32, device=z.device
                )
                sampled_bypass = torch.zeros(batch_size, device=z.device)
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
                residual, update_ema=update_ema)

            mask = (i < n_quantizers).float() * (1.0 - bypass)  # (B,), 1.0 if this level is active

            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss = commitment_loss + (commitment_loss_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        # Bypassed examples get the raw (unquantized) encoder output instead,
        # so the decoder receives unaltered gradients for them.
        if passthrough:
            z_q = z_q + bypass[:, None, None] * z

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss



if __name__ == "__main__":
    rvq = ResidualVectorQuantizeEMA(n_codebooks=2, codebook_size=16, codebook_dim=8)
    x = torch.randn(4, 512, 80)
    y = rvq(x)
    print(y[0].shape, y[1].shape, y[2].shape)