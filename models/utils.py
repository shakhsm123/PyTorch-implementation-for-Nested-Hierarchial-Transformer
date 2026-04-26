import torch
import einops

def block(x, num_blocks_per_side):
  ph = pw = int((x.shape[1] ** 0.5) // num_blocks_per_side)
  x=einops.rearrange(x, "b (nh ph nw pw) c -> b (nh nw) (ph pw) c", nh=num_blocks_per_side, nw=num_blocks_per_side, ph=ph, pw=pw)
  return x
def unblock(x, num_blocks_per_side):
  ph = pw = int((x.shape[2] ** 0.5))
  x=einops.rearrange(x, "b (nh nw) (ph pw) c -> b (nh ph) (nw pw) c", nh=num_blocks_per_side, nw=num_blocks_per_side, ph=ph, pw=pw)
  return x
