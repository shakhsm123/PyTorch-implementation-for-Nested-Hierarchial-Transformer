import torch
import torch.nn as nn
import einops
from models.utils import block, unblock
class BlockAggregation(nn.Module):
  def __init__(self,  d_in, d_out, num_blocks_per_side):
    super().__init__()
    self.d_in=d_in
    self.d_out=d_out
    self.num_blocks_per_side=num_blocks_per_side
    self.conv=nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)
    self.norm1=nn.LayerNorm(d_out)
    self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  def forward(self, x):
    x=unblock(x, self.num_blocks_per_side)
    x=einops.rearrange(x, 'b h w d -> b d h w')
    x=self.conv(x)
    x=einops.rearrange(x, 'b d h w -> b h w d')
    x=self.norm1(x)
    x=einops.rearrange(x, 'b h w d -> b d h w')
    x=self.maxpool(x)
    x=einops.rearrange(x, 'b d h w -> b h w d')
    x=einops.rearrange(x, 'b h w d -> b (h w) d')
    x=block(x, self.num_blocks_per_side//2)
    return x
