from torch.utils.data import DataLoader
from typing import List, Optional, Tuple 

import torch
import torch.nn as nn
import dataclasses
import math
import numpy as np
import regex
import wandb

@dataclasses.dataclass()
class Config:
  learning_rate: float
  epochs: int
  betas: Tuple[float, float]
  weight_decay: float
  device_type: str
  precision: str
  batch_size: int
  num_workers: int


@dataclasses.dataclass()
class HyenaConfig(Config):
  embedding_dim : int
  d_model: int
  n_layers: int
  vocab_size: int
  d_embed: int
  d_filter_mlp: int
  n_filter_layers: int
  context_length: int
  short_conv_size: int
  order: int
  pdrop_hyena: float
  pdrop_embed: float
  omega: Optional[int]

  torch.set_float32_matmul_precision("medium")

class Projection(torch.nn.Module):
  def __init__(self, d_model: int, N: int, conv_len: int):
    super().__init__()
    self.d_model = d_model
    self.N = N
    self.linear = torch.nn.Linear(d_model, d_model * (N + 1))
    self.conv = torch.nn.Conv1d(
      in_channels=d_model * (N + 1),
      out_channels=d_model * (N + 1),
      kernel_size=conv_len,
      groups=d_model * (N + 1),  # Depthwise convolution
      padding=conv_len - 1,
    )
    
  def forward(self, u: torch.Tensor) -> List[torch.Tensor]:
    z = self.linear(u)
    z = z.transpose(1, 2)  # Channels (embedding dim) needs to come first
    
    L = z.shape[2]
    z = self.conv(z)[..., :L]
    
    x = torch.split(z, self.d_model, dim=1)
    return x
  

class FFTConv(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(
      self,
      h: torch.Tensor,
      x: torch.Tensor,
      B: torch.Tensor
    ) -> torch.Tensor:
    L = h.shape[-1]
    h_f = torch.fft.rfft(h, n=2 * L, norm="forward")
    x_f = torch.fft.rfft(x.to(dtype=h.dtype), n=2 * L)
    y = torch.fft.irfft(h_f * x_f, n=2 * L, norm="forward")[..., :L]
    y = y + x * B
    y = y.to(dtype=h.dtype)  # y is ComplexFloat but we need it to be float
    return y
  
class PositionalEmbedding(torch.nn.Module):
  def __init__(self, d_embed: int, max_seq_len: int):
    assert d_embed % 2 == 1, "only odd dimensional positional embeddings are supported"
    assert d_embed > 1, "positional embedding must be at least 3"
    super().__init__()
    
    t = torch.linspace(start=0, end=1, steps=max_seq_len)[:, None]

    tp = torch.linspace(start=0, end=max_seq_len - 1, steps=max_seq_len)[:, None]
    K = (d_embed - 1) // 2   
    k = torch.linspace(start=0, end=K - 1, steps=K)[None, :]
    z = torch.exp(2 * math.pi * 1j * k * tp / max_seq_len)
    self.time_emb = torch.nn.Parameter(t.transpose(0, 1).unsqueeze(0), requires_grad=False)
    self.pos_emb = torch.nn.Parameter(torch.cat([t, z.real, z.imag], dim=-1), requires_grad=True)

  def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]: 
    return self.time_emb[:, :, :L], self.pos_emb[:L]    


class Sin(torch.nn.Module):
  def __init__(self, d_model: int, omega: int = 8, trainable: bool = False):
    super().__init__()
    self.freq = torch.nn.Parameter(omega * torch.ones(1, d_model), requires_grad=trainable)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(self.freq * x)
  
class AuthenticWindow(torch.nn.Module):
  def __init__(
      self,
      d_model: int,
      fast_decay_pct: float = 0.3,  # Defaults from the official implementation
      slow_decay_pct: float = 1.5,
      target: float = 1e-2,
      shift: float = 0.0,
    ):
    super().__init__()
    self.shift = shift
    min_decay = math.log(target) / slow_decay_pct
    max_decay = math.log(target) / fast_decay_pct
    self.alphas = torch.nn.Parameter(
      torch.linspace(
        start=min_decay,
        end=max_decay,
        steps=d_model
      )[None, :, None], requires_grad=True)
        
  def forward(self, t, x):
    L = x.shape[2]
    c = torch.exp(self.alphas * t)[:, :, :L]
    x = x * (c + self.shift)
    return x
  

class AuthenticHyenaFilter(torch.nn.Module):
  def __init__(
      self,
      d_model: int,
      d_mlp: int,
      d_embed: int,
      N: int,
      n_layers: int = 4,
      max_seq_len: int = 128,
      omega: int = 8,
    ):
    assert n_layers >= 2, "n_layers must be at least 2"
    super().__init__()

    self.N = N
    self.d_model = d_model
      
    self.pos_emb = PositionalEmbedding(d_embed, max_seq_len)
    
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(d_embed, d_mlp),
      Sin(d_mlp, omega),
    )
    for _ in range(n_layers - 2):
      self.mlp.append(torch.nn.Linear(d_mlp, d_mlp))
      self.mlp.append(Sin(d_mlp, omega))
    self.mlp.append(torch.nn.Linear(d_mlp, N * d_model, bias=False))

    self.t = torch.nn.Parameter(
      torch.linspace(
        start=0,
        end=1,
        steps=max_seq_len
      )[None, None, :], requires_grad=False)
    self.h = torch.nn.Parameter(torch.randn((N, d_model, max_seq_len)))
      
    self.window = AuthenticWindow(d_model)

  def forward(self, L: int) -> torch.Tensor:
    t, z = self.pos_emb(L)
    h = self.mlp(z)

    h = h.transpose(0, 1)
    h = h.reshape(self.N, self.d_model, L)
    
    h = self.h[:, :, :L]
    h = self.window(self.t, h)
    
    return h
  
class AuthenticHyenaBlock(torch.nn.Module):
  def __init__(self, config: HyenaConfig):
    super().__init__()
    self.proj_input = Projection(config.d_model, config.order, config.short_conv_size)
    self.proj_output = torch.nn.Linear(config.d_model, config.d_model)
    self.filter = AuthenticHyenaFilter(
      config.d_model,
      config.d_filter_mlp,
      config.d_embed,
      config.order,
      config.n_filter_layers,
      config.context_length,
      config.omega,
    )
    self.dropout = torch.nn.Dropout(config.pdrop_hyena)
    self.fft_conv = FFTConv()
    self.B = torch.nn.Parameter(torch.randn((config.order, 1, config.d_model, 1)))  # (order = 2, 1, d_model = 386, 1)

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    L = u.shape[1]
    
    *x, v = self.proj_input(u)
    
    h = self.filter(L)

    # The reference code for the paper does the product with x_i first
    # but we follow the paper eq (6) here in putting it after the convolution
    for i, x_i in enumerate(x):
      h_i = h[i].unsqueeze(0)
      v = x_i * self.fft_conv(h_i, v, self.B[i])
    
    v = v.transpose(1, 2)
    y = self.proj_output(v)

    return y


class FastaModel(nn.Module):
    def __init__(self, config: Config, block_cls: torch.nn.Module):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config
        self.tok_emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = torch.nn.Parameter(
                        torch.randn(1, config.context_length, config.d_model)
                        )
        self.dropout = torch.nn.Dropout(config.pdrop_embed)
        self.layers = torch.nn.Sequential(*[
                        block_cls(config) for _ in range(config.n_layers)
                        ])
        # reduce dimensionality to get embeddings
        self.down = torch.nn.Linear(
            29660,
            config.embedding_dim,
            bias=False
        )
        self.up = torch.nn.Linear(
            config.embedding_dim,
            29660,
            bias=False
        )
        
        self.lnorm = torch.nn.LayerNorm(config.d_model)
        self.head = torch.nn.Linear(  
        config.d_model,
        config.vocab_size,
        bias=False
        )
        # input embedding and logit output weights are tied
        self.head.weight = self.tok_emb.weight

    def forward(self, x, embed=False):
        token_embeddings = self.tok_emb(x)  # torch.Size([1, 29660, 386])
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]  # torch.Size([1, 29660, 386])

        x = self.dropout(token_embeddings + position_embeddings)  # torch.Size([1, 29660, 386])
        x = self.layers(x)  #torch.Size([1, 29660, 386])
        x = torch.transpose(x, 1, 2)  # torch.Size([1, 386, 29660])
        x = self.down(x) # torch.Size([1, 386, 100])
        if embed==True:
            return x  # returns embeddings during inference
        else:
            x = self.up(x)
            x = torch.transpose(x, 1, 2)

            x = self.lnorm(x) # torch.Size([1, 29660, 386])
            logits = self.head(x)  # torch.Size([1, 29660, 13])

            return logits
  

# class FastaModel(nn.Module):
#     def __init__(self, config: Config, block_cls: torch.nn.Module):
#         super().__init__()
#         # each token directly reads off the logits for the next token from a lookup table
#         self.config = config
#         self.tok_emb = torch.nn.Embedding(config.vocab_size, config.d_model)
#         self.pos_emb = torch.nn.Parameter(
#                         torch.randn(1, config.context_length, config.d_model)
#                         )
#         self.dropout = torch.nn.Dropout(config.pdrop_embed)
#         self.layers = torch.nn.Sequential(*[
#                         block_cls(config) for _ in range(config.n_layers)
#                         ])
#         self.lnorm = torch.nn.LayerNorm(config.d_model)
#         self.head = torch.nn.Linear(  
#         config.d_model,
#         config.vocab_size,
#         bias=False
#         )
#         # input embedding and logit output weights are tied
#         self.head.weight = self.tok_emb.weight

#     def forward(self, x, y):
#         token_embeddings = self.tok_emb(x)
#         position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

#         x = self.dropout(token_embeddings + position_embeddings)
#         x = self.layers(x)
#         logits = self.head(self.lnorm(x))

#         return logits, y