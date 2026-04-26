import torch
import torch.nn as nn
from models.attention import MHA
class TransformerLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.norm1=nn.LayerNorm(config["embed_dim"])
    self.norm2=nn.LayerNorm(config["embed_dim"])
    self.mha=MHA(config)
    self.ffn1=nn.Sequential(nn.Linear(config["embed_dim"], config["embed_dim"]*4), nn.GELU(), nn.Dropout(p=config["dropout"]), nn.Linear(config["embed_dim"]*4, config["embed_dim"]), nn.Dropout(p=config["dropout"]))
    self.dropout=nn.Dropout(p=config["dropout"])
  def forward(self, x):
    y=x+self.mha(self.norm1(x))
    x=y+self.ffn1(self.norm2(y))
    return x