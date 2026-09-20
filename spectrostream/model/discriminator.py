import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from audiotools import AudioSignal
from audiotools import ml
from audiotools import STFTParams
from einops import rearrange
from torch.nn.utils import weight_norm
from typing import List, Tuple

from .spectrostream import init_weights


class SampleLayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.shape[1:])


class DiscriminatorBlock(nn.Module):
    def __init__(self, N_in:int = 16, N_out:int = 16, stride: Tuple[int] = (1,1)):
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.stride = stride
        self.k_size_2 = (max(3,2*self.stride[0]), max(3,2*self.stride[1]))
        self.padding_2 = ((self.stride[0]+1)//2, (self.stride[1]+1)//2)
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.N_in, out_channels=self.N_in, kernel_size=(3,3), padding=(1,1)),
            SampleLayerNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=self.k_size_2, stride=self.stride, padding=self.padding_2),
            SampleLayerNorm(),
            )
        
        if self.stride == (1,1) :
            self.pooling = nn.Identity()
        else :
            self.pooling = nn.AvgPool2d(kernel_size=self.k_size_2, stride=self.stride, padding=self.padding_2)

        if self.N_in == self.N_out :
            self.projection = nn.Identity()
        else :
            self.projection = nn.Conv2d(in_channels=self.N_in, out_channels=self.N_out, kernel_size=(1,1), bias=False)

    def forward(self, x):
        return self.block(x) + self.projection(self.pooling(x))

class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        N_in_factors: List[int] = [1,2,4,4,8,8],
        N_out_factors: List[int] = [2,4,4,8,8,16],
        strides: List[Tuple[int]] = [(2,1),(2,2),(2,1),(2,2),(2,1),(2,2)],
        base_conv_depth: int = 32,
        frequency_bins: int = 16,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        # Create first convolution
        self.N_in_factors = N_in_factors
        self.N_out_factors = N_out_factors
        self.strides = strides
        self.base_conv_depth = base_conv_depth
        self.frequency_bins = frequency_bins
        self.use_checkpoint = use_checkpoint

        discriminator_blocks = [nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=self.base_conv_depth, kernel_size=(7,7), padding=(3,3)), SampleLayerNorm()])]

        for i_block in range(len(strides)) :

            N_in = self.base_conv_depth * self.N_in_factors[i_block]
            N_out = self.base_conv_depth * self.N_out_factors[i_block]

            discriminator_blocks += [DiscriminatorBlock(N_in, N_out, stride=strides[i_block])]
        
        discriminator_blocks += [nn.Sequential(*[nn.LeakyReLU(0.2), nn.Conv2d(in_channels=self.base_conv_depth * self.N_out_factors[-1], out_channels=1, kernel_size=(frequency_bins//64,1), stride=(frequency_bins//64,1))])]

        self.discriminator_blocks = nn.ModuleList(discriminator_blocks)

    def forward(self, x, return_features: bool = True):

        fmap = [] if return_features else None

        for block in self.discriminator_blocks :
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                # torch.is_grad_enabled() (not x.requires_grad!) is the right guard:
                # GANLoss's D-step passes a *detached* fake in, so input_rep itself
                # has requires_grad=False even though we do want a backward pass
                # through the discriminator's own weights there. is_grad_enabled()
                # correctly stays True in that case and only goes False inside the
                # G-step's `with torch.no_grad():` real-signal pass, where there's
                # no backward to save memory for anyway.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if return_features:
                fmap.append(x)

        return fmap if return_features else x

class Discriminator(nn.Module):
    def __init__(
        self,
        N_in_factors: List[int] = [1,2,4,4,8,8],
        N_out_factors: List[int] = [2,4,4,8,8,16],
        strides: List[Tuple[int]] = [[2,1],[2,2],[2,1],[2,2],[2,1],[2,2]],
        base_conv_depth: int = 32,
        window_lengths: List[int] = [128, 256, 512, 1024, 2048, 4096],
        use_checkpoint: bool = False,
    ):
        super().__init__()

        strides = [tuple(stride) for stride in strides]

        self.window_lengths = window_lengths
        self.discriminators = nn.ModuleList([BaseDiscriminator(N_in_factors, N_out_factors, strides, base_conv_depth, frequency_bins=window_length//2, use_checkpoint=use_checkpoint) for window_length in window_lengths])

        # Previously used PyTorch's default (Kaiming-uniform) init here, while the
        # generator uses a conservative truncated-normal(std=0.02) -- that asymmetry
        # gave the discriminator a more assertive starting point than the generator
        # before either had learned anything, plausibly contributing to the early
        # discriminator collapse observed in training. Match the generator's init.
        self.apply(init_weights)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y
    
    def forward(self, signal, return_features: bool = True):

        outputs = []
        for i_discriminator, discriminator in enumerate(self.discriminators) :
            signal.stft_params = STFTParams(window_length=self.window_lengths[i_discriminator], hop_length=self.window_lengths[i_discriminator]//2, window_type="hann")
            signal.stft()
            spectrogram_modulus = torch.abs(signal.stft_data)
            input_spectrograms = rearrange(torch.view_as_real(signal.stft_data), 'b 1 f t c -> b c f t')
            input_rep = torch.cat([input_spectrograms, spectrogram_modulus], dim=1)
            input_rep = input_rep[:, :, :-1, :]
            output = discriminator(input_rep, return_features=return_features)
            outputs.append(output)
        
        return outputs


if __name__ == "__main__":
    disc = Discriminator()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
