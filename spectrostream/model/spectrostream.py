import math
from typing import List
from typing import Union
from typing import Tuple

from einops import rearrange
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint_sequential
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
from torch.nn.utils import weight_norm

from .base import CodecMixin
from spectrostream.nn.layers import Snake1d
from spectrostream.nn.layers import WNConv1d
from spectrostream.nn.layers import WNConvTranspose1d
from spectrostream.nn.quantize_EMA import ResidualVectorQuantizeEMA



def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None :
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None :
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None :
            nn.init.constant_(m.bias, 0)


class EncoderBlock(nn.Module):
    def __init__(self, N_in:int = 16, N_out:int = 16, stride: Tuple[int] = (1,1)):
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.stride = stride
        self.k_size_2 = (max(3,2*self.stride[0]), max(3,2*self.stride[1]))
        self.padding_2 = ((self.stride[0]+1)//2, (self.stride[1]+1)//2)
        
        self.block = nn.Sequential(
            nn.ELU(),
            weight_norm(nn.Conv2d(in_channels=self.N_in, out_channels=self.N_in, kernel_size=(3,3), padding=(1,1))),
            nn.ELU(),
            weight_norm(nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=self.k_size_2, stride=self.stride, padding=self.padding_2)),
            )
        
        if self.stride == (1,1) :
            self.pooling = nn.Identity()
        else :
            self.pooling = nn.AvgPool2d(kernel_size=self.k_size_2, stride=self.stride, padding=self.padding_2)

        if self.N_in == self.N_out :
            self.projection = nn.Identity()
        else :
            self.projection = nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=(1,1))

    def forward(self, x):
        return self.block(x) + self.projection(self.pooling(x))

class BottleneckBlock(nn.Module):
    def __init__(self, N_in:int = 16, N_out:int = 16):
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.N_interm = max(self.N_in, self.N_out)
        self.block = nn.Sequential(
            nn.ELU(),
            weight_norm(nn.Conv1d(in_channels=self.N_in, out_channels=self.N_interm, kernel_size=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(in_channels=self.N_interm, out_channels=self.N_out, kernel_size=1)),
            )
        
        self.projection = nn.Conv1d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=1)

    def forward(self, x):
        return self.block(x) + self.projection(x)

class Encoder(nn.Module):
    def __init__(
        self,
        N_in_factors: List[int] = [1,2,2,4,4,4,8,8],
        N_out_factors: List[int] = [2,2,4,4,4,8,8,8],
        strides: List[Tuple[int]] = [(2,1),(2,1),(3,1),(2,1),(2,1),(2,2),(1,2),(1,1)],
        base_conv_depth: int = 32,
        bottleneck_N_in_factor : int = 40,
        bottleneck_N_out_dim : int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        # Create first convolution
        self.N_in_factors = N_in_factors
        self.N_out_factors = N_out_factors
        self.strides = strides
        self.base_conv_depth = base_conv_depth
        self.bottleneck_N_in_factor = bottleneck_N_in_factor
        self.bottleneck_N_out_dim = bottleneck_N_out_dim
        self.use_checkpoint = use_checkpoint

        encoder_blocks = [weight_norm(nn.Conv2d(in_channels=2, out_channels=self.base_conv_depth, kernel_size=(7,7), padding=(3,3)))]

        for i_block in range(len(strides)) :

            N_in = self.base_conv_depth * self.N_in_factors[i_block]
            N_out = self.base_conv_depth * self.N_out_factors[i_block]

            encoder_blocks += [EncoderBlock(N_in, N_out, stride=strides[i_block])]

        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        self.bottleneck_block = BottleneckBlock(self.base_conv_depth * self.bottleneck_N_in_factor, self.bottleneck_N_out_dim)

    def forward(self, x):

        if self.use_checkpoint and self.training:
            # Recompute activations during backward instead of storing them --
            # trades ~30% more compute for a large activation-memory reduction.
            # One segment per block (finest granularity = max memory savings).
            # use_reentrant=False correctly tracks gradients into the blocks'
            # conv weights even though `x` itself has requires_grad=False.
            y = checkpoint_sequential(self.encoder_blocks, len(self.encoder_blocks), x, use_reentrant=False)
        else:
            y = self.encoder_blocks(x)

        y = rearrange(y, "b c h t -> b (c h) t")

        y = self.bottleneck_block(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(self, N_in:int = 16, N_out:int = 16, stride: Tuple[int] = (1,1)):
        super().__init__()
        self.N_in=N_in
        self.N_out=N_out
        self.stride=stride
        self.k_size_2 = (max(3,2*self.stride[0]), max(3,2*self.stride[1]))
        self.padding_2 = ((self.stride[0]+1)//2, (self.stride[1]+1)//2)
        self.output_padding = (max(0, (self.stride[0] - 1)//2), max(0, (self.stride[1] - 1)//2))
        
        if self.stride == (1,1) :
            self.block = nn.Sequential(
                nn.ELU(),
                weight_norm(nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=(3,3), padding=(1,1))),
                nn.ELU(),
                weight_norm(nn.Conv2d(in_channels=self.N_out, out_channels=self.N_out, kernel_size=(3,3), padding=(1,1))),
                )
        else :
        
            self.block = nn.Sequential(
                nn.ELU(),
                weight_norm(nn.ConvTranspose2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=self.k_size_2, stride=self.stride, padding=self.padding_2, output_padding=self.output_padding)),
                nn.ELU(),
                weight_norm(nn.Conv2d(in_channels=self.N_out, out_channels=self.N_out, kernel_size=(3,3), padding=(1,1)))
            )
                

        
        if self.N_in != self.N_out :
            self.projection = nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=(1,1))
        else :
            self.projection = nn.Identity()

    def forward(self, x):

        y = self.block(x)

        if self.stride != (1,1) :
            res = y + nn.functional.interpolate(self.projection(x), size = y.size()[-2:], mode="bilinear")#'nearest')
        else :
            res = y + self.projection(x)
        return res

class Decoder(nn.Module):
    def __init__(
        self,
        N_in_factors: List[int] = [8,8,8,4,4,4,2,2],
        N_out_factors: List[int] = [8,8,4,4,4,2,2,1],
        strides: List[Tuple[int]] = [(1,1), (1,2), (2,2), (2,1), (2,1), (3,1), (2,1), (2,1)],
        base_conv_depth: int = 64,
        bottleneck_N_in_dim : int = 256,
        bottleneck_N_out_factor : int = 40,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.N_in_factors = N_in_factors
        self.N_out_factors = N_out_factors
        self.strides = strides
        self.base_conv_depth = base_conv_depth
        self.bottleneck_N_in_dim = bottleneck_N_in_dim
        self.bottleneck_N_out_factor = bottleneck_N_out_factor
        self.use_checkpoint = use_checkpoint

        self.bottleneck_block = BottleneckBlock(self.bottleneck_N_in_dim, self.base_conv_depth * self.bottleneck_N_out_factor)

        decoder_blocks = []

        for i_block in range(len(strides)) :

            N_in = self.base_conv_depth * self.N_in_factors[i_block]
            N_out = self.base_conv_depth * self.N_out_factors[i_block]

            decoder_blocks += [DecoderBlock(N_in, N_out, stride=strides[i_block])]
        
        decoder_blocks += [nn.ELU()]
        decoder_blocks += [nn.Conv2d(in_channels=self.base_conv_depth * self.N_out_factors[-1], out_channels=2, kernel_size=(7,7), padding=(3,3))]

        self.decoder_blocks = nn.Sequential(*decoder_blocks)

    def forward(self, x):

        y=x

        y = self.bottleneck_block(x)

        compressed_dim = self.bottleneck_N_out_factor // self.N_in_factors[0]

        y = rearrange(y, "b (c h) t -> b c h t", h = compressed_dim)

        if self.use_checkpoint and self.training:
            y = checkpoint_sequential(self.decoder_blocks, len(self.decoder_blocks), y, use_reentrant=False)
        else:
            y = self.decoder_blocks(y)

        return y



class SpS(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_N_in_factors: List[int] = [1,2,2,4,4,4,8,8],
        encoder_N_out_factors: List[int] = [2,2,4,4,4,8,8,8],
        encoder_strides: List[Tuple[int]] = [[2,1],[2,1],[3,1],[2,1],[2,1],[2,2],[1,2],[1,1]],
        encoder_base_conv_depth: int = 32,
        encoder_bottleneck_N_in_factor : int = 40,
        encoder_bottleneck_N_out_dim : int = 256,
        decoder_N_in_factors: List[int] = [8,8,8,4,4,4,2,2],
        decoder_N_out_factors: List[int] = [8,8,4,4,4,2,2,1],
        decoder_strides: List[Tuple[int]] = [[1,1], [1,2], [2,2], [2,1], [2,1], [3,1], [2,1], [2,1]],
        decoder_base_conv_depth: int = 64,
        decoder_bottleneck_N_in_dim : int = 256,
        decoder_bottleneck_N_out_factor : int = 40,
        n_quantizers: int = 64,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 1.0,
        quantizer_bypass_prob: float = 0.5,
        quantizer_l2_normalize: bool = False,
        quantizer_dead_code_steps: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        encoder_strides = [tuple(stride) for stride in encoder_strides]
        decoder_strides = [tuple(stride) for stride in decoder_strides]

        self.encoder = Encoder(
            N_in_factors=encoder_N_in_factors,
            N_out_factors=encoder_N_out_factors,
            strides=encoder_strides,
            base_conv_depth=encoder_base_conv_depth,
            bottleneck_N_in_factor=encoder_bottleneck_N_in_factor,
            bottleneck_N_out_dim=encoder_bottleneck_N_out_dim,
            use_checkpoint=use_checkpoint,
        )

        
        self.quantizer = ResidualVectorQuantizeEMA(
            input_dim=encoder_bottleneck_N_out_dim,
            n_codebooks=n_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            quantizer_bypass_prob=quantizer_bypass_prob,
            l2_normalize=quantizer_l2_normalize,
            drop_unused_after_steps=quantizer_dead_code_steps,
        )

        self.decoder = Decoder(
            N_in_factors=decoder_N_in_factors,
            N_out_factors=decoder_N_out_factors,
            strides=decoder_strides,
            base_conv_depth=decoder_base_conv_depth,
            bottleneck_N_in_dim=decoder_bottleneck_N_in_dim,
            bottleneck_N_out_factor=decoder_bottleneck_N_out_factor,
            use_checkpoint=use_checkpoint,
        )

        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, spectrogram):

        length = spectrogram.shape[-1]

        hop_length = np.prod([stride[1] for stride in self.encoder.strides])

        n_token = math.ceil(length / hop_length)

        time_right_pad = n_token * hop_length - length

        spectrogram = nn.functional.pad(spectrogram, (0, time_right_pad))

        return spectrogram

    def encode(
        self,
        spectrogram: List[torch.Tensor],
        *,
        n_quantizers=None,
        bypass=None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : Tensor[B], optional
            Per-example truncation level to impose instead of letting the
            quantizer draw its own. Used by SpSSplit to give both branches of the
            cascade the same rate. If None, the quantizer samples during training
            and uses all quantizers in eval.
        bypass : Tensor[B], optional
            Per-example bypass mask, shared with `n_quantizers` for the same
            reason.

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
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(spectrogram)

        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers=n_quantizers, bypass=bypass
        )

        return z_q, codes, latents, commitment_loss, codebook_loss

    def decode_from_codes(self, codes: torch.Tensor, length: int = None):
        """Decode a spectrogram straight from integer codes.

        codes : Tensor[B x N x T] -- a prefix of the RVQ stack, so N < n_quantizers
                simply means a lower bitrate.
        Used to turn tokens predicted by the bandwidth-extension transformer back
        into audio, without ever running the encoder.
        """
        z_q = self.quantizer.from_codes(codes)
        x = self.decode(z_q)
        return x if length is None else x[..., :length]

    def decode(self, z: List[torch.Tensor]):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        spectrogram: Union[torch.Tensor,List[torch.Tensor]],
        #input_sample_rates: Union[int, List[int]]
    ):
        """Model forward pass

        Parameters
        ----------
        spectrogram : Tensor[B x 1 x T]
            Spectrogram data to encode
        input_sample_rates : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with  the following keys:
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
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """

        length = spectrogram.shape[-1]
        spectrogram = self.preprocess(spectrogram)

        z, codes, latents, commitment_loss, codebook_loss = self.encode(spectrogram=spectrogram)

        x = self.decode(z)

        return {
            "spectrogram": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_losses": commitment_loss,
            "vq/codebook_losses": codebook_loss,
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = SpS().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
