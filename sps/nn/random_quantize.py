from typing import Union

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

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, collapse_mitigant: bool,):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.collapse_mitigant = collapse_mitigant

        if self.collapse_mitigant : 

            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
            self.codebook = nn.Embedding(codebook_size, codebook_dim)

        else :

            self.codebook = nn.Embedding(codebook_size, input_dim)
        
        #self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        #self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        

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
        #z_e = self.in_proj(z)  # z_e : (B x D x T)

        if self.collapse_mitigant :

            # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
            z_e = self.in_proj(z)  # z_e : (B x D x T)
        
        else :

            z_e = z
        
        #print('proj_quant',z_e.size())

        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        #z_q = self.out_proj(z_q)

        if self.collapse_mitigant:
            
            z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):

        encodings = rearrange(latents, "b d t -> (b t) d")
        #print('rearranged_quant',encodings.size())
        codebook = self.codebook.weight  # codebook: (N x D)


        if self.collapse_mitigant :

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
        #print('indices_quant',indices.size())
        z_q = self.decode_code(indices)
        #print('out_quant',z_q.size())
        return z_q, indices
    
class RandomVectorQuantize(nn.Module):
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

    def __init__(self, input_dim: int, randn_codebook_size: int, embedding: nn.Embedding, codebook_dim: int, collapse_mitigant: bool, trained_cb: bool):
        super().__init__()
        self.randn_codebook_size = randn_codebook_size
        self.codebook_dim = codebook_dim
        self.collapse_mitigant = collapse_mitigant
        self.trained_cb = trained_cb

        if self.collapse_mitigant : 

            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

        #self.codebook = nn.Embedding(codebook_size, codebook_dim).requires_grad_(False)
        self.codebook = embedding
        self.codebook_size = self.codebook.num_embeddings
        

    def forward(self, z, sampling):
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

        if self.collapse_mitigant :

            # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
            z_e = self.in_proj(z)  # z_e : (B x D x T)
        
        else :

            z_e = z

        #print('proj_quant',z_e.size())
        
        z_q, indices = self.decode_latents(z_e, sampling)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        
        if self.trained_cb :
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        
        else :
            codebook_loss = torch.zeros(size = commitment_loss.size()).to(commitment_loss.device)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        if self.collapse_mitigant:
            
            z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents, sampling):

        #idx = torch.multinomial(torch.ones(self.codebook_size),self.randn_codebook_size).to(latents.device)
        idx = sampling

        encodings = rearrange(latents, "b d t -> (b t) d")
        #print('rearranged_quant',encodings.size())
        codebook = self.codebook.weight[idx]  # codebook: (N x D)


        if self.collapse_mitigant:
            # L2 normalize encodings and codebook (ViT-VQGAN)
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)
            #self.codebook.weight[idx] = F.normalize(self.codebook.weight[idx])

        # Compute euclidean distance with codebook
        
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        
        """
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.weight[idx].t()
            + self.codebook.weight[idx].pow(2).sum(1, keepdim=True).t()
        )
        """
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        #print('indices_quant',indices.size())
        real_indices = idx[indices]
        
        z_q = self.decode_code(real_indices)
        #print('out_quant',z_q.size())
        
        return z_q, real_indices


class RandomResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        collapse_mitigant: bool = False, #True,
        size_big_codebook: int = 8192,
        number_random_quantizers: int = 4,
        size_sampling: int = 1024,
        random_seed: int = None,
        number_trained_quantizers: int = 4,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.collapse_mitigant = collapse_mitigant
        self.size_big_codebook = size_big_codebook

        assert number_random_quantizers <= n_codebooks, "Too many random quantizers !"
        assert number_trained_quantizers <= number_random_quantizers, "More trained than random quantizers !"

        self.number_random_quantizers = number_random_quantizers
        self.number_trained_quantizers = number_trained_quantizers
        self.size_sampling = size_sampling

        if self.collapse_mitigant:

            if self.number_trained_quantizers == 0:

                self.big_embedding = nn.Embedding(size_big_codebook, codebook_dim[-1]).requires_grad_(False)

            elif self.number_trained_quantizers < self.number_random_quantizers :

                means = [[-0.11520223, -0.12549783, -0.09464508,  0.16919054, -0.0787252 ,  0.11072443,  0.09925366,  0.02332132],
                         [-0.03650802,  0.05348687,  0.10868465, -0.02770074,  0.02295331, -0.00783541,  0.11034289, -0.05973633],
                         [ 0.03956118,  0.02418371,  0.06306202, -0.0390567 ,  0.04374486,  0.009958  , -0.07310977, -0.05468717],
                         [-0.01159949, -0.00974658,  0.01323013, -0.00341866,  0.04071631,  0.0255354 , -0.0304246 , -0.00231216],
                         [-0.04118184, -0.00945165,  0.01324847, -0.05061552, -0.02626047, -0.03228011, -0.00290333, -0.03157319],
                         [-0.00160538,  0.0416545 , -0.00642506, -0.02625478, -0.03411976, -0.02458269, -0.01719413,  0.01888132],
                         [-0.00762004, -0.01692576,  0.03200294,  0.0160811 ,  0.01259751, -0.01412598,  0.00069721, -0.00634967],
                         [-0.00645784, -0.05261763, -0.00584335,  0.02851553, -0.04655663, -0.01716808,  0.04031012, -0.00355523],
                         [ 0.00228948, -0.02568473, -0.00589426,  0.00665507,  0.00271465,  0.02891909,  0.00439095, -0.04728379]]

                std = [[3.8721025, 3.711569 , 3.6155648, 3.8295763, 3.7311566, 3.8846989, 3.7415109, 3.7942195],
                       [3.1496935, 3.1669374, 3.2322445, 3.1972654, 3.2078824, 3.1450446, 3.2423851, 3.2122362],
                       [2.9534364, 2.9380867, 2.8870797, 2.9234796, 2.942341 , 2.906031 , 2.9384878, 2.9791462],
                       [2.7509606, 2.7629414, 2.7770526, 2.7894654, 2.8072937, 2.7585347, 2.7708533, 2.767419 ],
                       [2.6405687, 2.6587307, 2.6143486, 2.6763527, 2.6625085, 2.6728525, 2.644658 , 2.6653118],
                       [2.5570128, 2.5718179, 2.5643344, 2.5449083, 2.5953877, 2.5514872, 2.5617416, 2.5600371],
                       [2.4596157, 2.4822578, 2.4723082, 2.4493248, 2.504611 , 2.4732373, 2.4533474, 2.4709327],
                       [2.3603654, 2.3317597, 2.360175 , 2.3823683, 2.3635175, 2.3733788, 2.3507445, 2.3806777],
                       [2.2336652, 2.2401192, 2.247559 , 2.2346804, 2.257628 , 2.2332304, 2.2471206, 2.2369993]]
                
                """
                big_embedding_init = []

                for i_token in range(size_big_codebook) :

                    for i_dim in range(codebook_dim[-1]) :

                        token = []

                        sample = 10**30

                        while sample < means:
                """

                self.big_embedding = nn.Embedding(size_big_codebook, codebook_dim[-1]).requires_grad_(False)
                self.big_embedding_trained = nn.Embedding(size_big_codebook, codebook_dim[-1])

            else :

                self.big_embedding_trained = nn.Embedding(size_big_codebook, codebook_dim[-1])
        
        else :

            if self.number_trained_quantizers == 0:

                self.big_embedding = nn.Embedding(size_big_codebook, input_dim).requires_grad_(False)

            elif self.number_trained_quantizers < self.number_random_quantizers :
                    
                self.big_embedding = nn.Embedding(size_big_codebook, input_dim).requires_grad_(False)
                self.big_embedding_trained = nn.Embedding(size_big_codebook, input_dim)
            
            else:

                self.big_embedding_trained = nn.Embedding(size_big_codebook, input_dim)

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i], collapse_mitigant)
                for i in range(n_codebooks - number_random_quantizers)
            ]
            +
            [
                RandomVectorQuantize(input_dim, size_sampling, self.big_embedding_trained, codebook_dim[i], collapse_mitigant, trained_cb=True)
                for i in range(n_codebooks-number_random_quantizers, n_codebooks-number_random_quantizers+number_trained_quantizers)
            ]
            +
            [
                RandomVectorQuantize(input_dim, size_sampling, self.big_embedding, codebook_dim[i], collapse_mitigant, trained_cb=False)
                for i in range(n_codebooks-number_random_quantizers+number_trained_quantizers, n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

        if not random_seed is None :
            torch.manual_seed(random_seed)

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
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
        if self.number_random_quantizers > self.number_trained_quantizers :

            permutation = torch.randperm(self.size_big_codebook).to(z.device)

        if self.number_trained_quantizers != 0 :

            permutation_trained = torch.randperm(self.size_big_codebook).to(z.device)

        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
            
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        n_random = 0
        n_random_trained = 0

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            
            if isinstance(quantizer, RandomVectorQuantize) :

                if quantizer.trained_cb :

                    sampling = permutation_trained[n_random_trained*self.size_sampling:(n_random_trained+1)*self.size_sampling]
                    #print(residual.size())
                    z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                        residual,
                        sampling
                    )
                    n_random_trained += 1

                else:

                    sampling = permutation[n_random*self.size_sampling:(n_random+1)*self.size_sampling]
                    #print(residual.size())
                    z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                        residual,
                        sampling
                    )
                    n_random += 1

            else : 
                #print(residual.size())
                z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                    residual
                )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        if self.number_random_quantizers > self.number_trained_quantizers:
            permutation = torch.randperm(self.size_big_codebook).to(latents.device)
        if self.number_trained_quantizers != 0:
            permutation_trained = torch.randperm(self.size_big_codebook).to(latents.device)
        n_random = 0
        n_random_trained = 0
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            if isinstance(self.quantizers[i], RandomVectorQuantize) :
                if self.quantizers[i].trained_cb :
                    sampling = permutation_trained[n_random_trained*self.size_sampling:(n_random_trained+1)*self.size_sampling]
                    z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :], sampling)
                    n_random_trained += 1

                else :
                    sampling = permutation[n_random*self.size_sampling:(n_random+1)*self.size_sampling]
                    z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :], sampling)
                    n_random += 1
            else :
                z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


if __name__ == "__main__":
    
    rvq = RandomResidualVectorQuantize(quantizer_dropout=True)

    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y[1])
    #print(y["latents"].shape)


