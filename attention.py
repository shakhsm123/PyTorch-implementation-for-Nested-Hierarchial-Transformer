import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
class MHA(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.num_heads=config["num_heads"]
    self.head_dim=config["embed_dim"]//self.num_heads
    self.scale=self.head_dim**-0.5
    self.W_q=nn.Linear(config["embed_dim"], config["embed_dim"])
    self.W_k=nn.Linear(config["embed_dim"], config["embed_dim"])
    self.W_v=nn.Linear(config["embed_dim"], config["embed_dim"])
    self.W_o=nn.Linear(config["embed_dim"], config["embed_dim"])
    self.dropout=nn.Dropout(p=config["attn_dropout"])
  def forward(self, x):
    Batch, num_blocks, seq_len, d=x.shape
    x=x.reshape(Batch*num_blocks, seq_len, d)
    Q=self.W_q(x)
    K=self.W_k(x)
    V=self.W_v(x)
    x=x.reshape(Batch*num_blocks, self.num_heads, seq_len, self.head_dim)
    Q=Q.reshape(Batch*num_blocks, self.num_heads, seq_len, self.head_dim)
    K=K.reshape(Batch*num_blocks, self.num_heads, seq_len, self.head_dim)
    V=V.reshape(Batch*num_blocks, self.num_heads, seq_len, self.head_dim)
    scores=Q @ K.transpose(-2,-1) *self.scale
    weights=F.softmax(scores, dim=-1)
    weights=self.dropout(weights)
    attention=weights  @ V
    out=einops.rearrange(attention, "b h n d -> b n (h d)")
    out=self.W_o(out)
    out=out.reshape(Batch, num_blocks, seq_len, d)
    return out