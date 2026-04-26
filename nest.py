import torch
import torch.nn as nn
from models.patch_embed import PatchEmbedding
from models.hierarchy import NestHierarchy
from models.aggregation import BlockAggregation
from models.utils import block
class NeST(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.blocks_per_side = 4  
    num_patches = (config["image_size"] // config["patch_size"]) ** 2
    seq_len     = num_patches // (self.blocks_per_side ** 2)
    
    self.embedding = PatchEmbedding(config)
    self.list_of_hierarchy = nn.ModuleList([
        NestHierarchy(config, num_layers=2, seq_len=seq_len)
        for _ in range(config["num_hierarchy"])
    ])
    self.list_of_blocks_aggregate = nn.ModuleList([
        BlockAggregation(config["embed_dim"], config["embed_dim"],
                         self.blocks_per_side // (2**i))
        for i in range(config["num_hierarchy"] - 1)
    ])
    self.norm       = nn.LayerNorm(config["embed_dim"])
    self.classifier = nn.Linear(config["embed_dim"], config["num_classes"])
  def forward(self, x, return_features=False):
    features = []
    x = self.embedding(x)
    x = block(x, self.blocks_per_side)
    for i, hierarchy in enumerate(self.list_of_hierarchy):
        x = hierarchy(x)
        features.append(x)
        if i < len(self.list_of_blocks_aggregate):
            x = self.list_of_blocks_aggregate[i](x)
    x = self.norm(x.squeeze(1))
    x = x.mean(dim=1)
    x = self.classifier(x)
    if return_features:
        return x, features
    return x