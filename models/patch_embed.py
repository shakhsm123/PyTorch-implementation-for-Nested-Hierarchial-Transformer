import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config=config
    self.conv=nn.Conv2d(in_channels=3, out_channels=self.config["embed_dim"], kernel_size=self.config["patch_size"], stride=self.config["patch_size"])
    self.layernorm=nn.LayerNorm(self.config["embed_dim"])
  def forward(self, x):
    x=self.conv(x)
    x=x.flatten(2)
    x=x.transpose(1,2)
    x=self.layernorm(x)
    return x
