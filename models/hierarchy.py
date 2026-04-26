import torch
import torch.nn as nn
from models.transformer import TransformerLayer
class NestHierarchy(nn.Module):
  def __init__(self, config, num_layers, seq_len):
    super().__init__()
    self.pos_embed=nn.Parameter(torch.zeros(1, 1, seq_len, config["embed_dim"]))
    self.recurrent_transformers=nn.ModuleList([TransformerLayer(config) for i in range(num_layers)])
  def forward(self, x):
    x=x+self.pos_embed
    for layer in self.recurrent_transformers:
      x=layer(x)
    return x
