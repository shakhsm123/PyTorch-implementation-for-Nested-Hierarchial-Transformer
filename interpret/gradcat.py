import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def gradcat(model, image, class_idx, blocks_per_side, device):
  image=image.to(device).requires_grad_(True)

  logits, features=model(image, return_features=True)

  for f in features:
    f.retain_grad()
  logits[0, class_idx].backward(retain_graph=True)

  path=[]

  for f in range(len(features)-1,-1,-1):
    A=features[f]
    grad=A.grad
    h=A*(-grad)
    h=h.mean(dim=-1)

    seq_len=h.shape[2]
    grid_size=int(seq_len**0.5)

    h = h.reshape(1, h.shape[1], grid_size, grid_size)

    h = h.mean(dim=1, keepdim=True)
    pool_size = grid_size // 2
    scores = F.avg_pool2d(h.squeeze(1).unsqueeze(0), kernel_size=pool_size, stride=pool_size)
    best = scores.argmax().item()
    path.append(best)
  return path

def visualize_gradcat(image, path, patch_size, image_size):
    """
    image     : (1, 3, H, W) tensor, unnormalized or normalized
    path      : list of selected indices from gradcat, top→bottom
    patch_size: S from config
    image_size: H from config
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    # convert tensor to numpy for display
    img = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # normalize to 0-1 for display

    # reconstruct which patch the path landed on
    # path[0] = which 2×2 quadrant at top hierarchy
    # path[1] = which 2×2 quadrant within that
    # path[2] = which 2×2 quadrant within that
    # each step narrows down the spatial region by 2× in each dimension
    num_patches_per_side = image_size // patch_size   # e.g. 32//2 = 16
    region_size = num_patches_per_side               # starts as full grid

    row, col = 0, 0
    for idx in path:
        region_size = region_size // 2
        # idx encodes (r, c) in a 2×2 grid
        r = idx // 2
        c = idx  % 2
        row += r * region_size
        col += c * region_size

    # convert patch coords to pixel coords
    px = col * patch_size
    py = row * patch_size
    pw = region_size * patch_size
    ph = region_size * patch_size

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img)
    rect = patches.Rectangle((px, py), pw, ph,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"GradCAT path: {path}")
    ax.axis('off')
    plt.show()
