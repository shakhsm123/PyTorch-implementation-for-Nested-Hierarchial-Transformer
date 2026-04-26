import torch

def get_config():
    return {
        "image_size"    : 32,
        "patch_size"    : 2,
        "embed_dim"     : 192,
        "num_heads"     : 3,
        "num_hierarchy" : 3,
        "num_classes"   : 100,
        "mlp_ratio"     : 4,
        "dropout"       : 0.1,
        "attn_dropout"  : 0.0,
        "device"        : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }