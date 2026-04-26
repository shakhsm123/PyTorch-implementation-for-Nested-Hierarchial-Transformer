import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def gradcat(model, image, class_idx, blocks_per_side, device):
    image = image.to(device).requires_grad_(True)
    logits, features = model(image, return_features=True)
    for f in features:
        f.retain_grad()
    logits[0, class_idx].backward(retain_graph=True)

    path       = []
    all_scores = []

    for f in range(len(features) - 1, -1, -1):
        A    = features[f]
        grad = A.grad
        h    = A * (-grad)
        h    = h.mean(dim=-1)

        seq_len   = h.shape[2]
        grid_size = int(seq_len ** 0.5)

        h         = h.reshape(1, h.shape[1], grid_size, grid_size)
        h         = h.mean(dim=1, keepdim=True)
        pool_size = grid_size // 2
        scores    = F.avg_pool2d(h.squeeze(1).unsqueeze(0),
                                 kernel_size=pool_size, stride=pool_size)
        scores    = scores.squeeze().reshape(4)

        all_scores.append(scores.detach().cpu().tolist())
        path.append(scores.argmax().item())

    return path, all_scores


def visualize_gradcat(image, path, all_scores, patch_size, image_size,
                      class_name=None):
    fig = plt.figure(figsize=(14, 5))

    # ── left panel: image with red arrow to selected patch ──────────────────
    ax_img = fig.add_axes([0.0, 0.0, 0.35, 1.0])

    img = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    ax_img.imshow(img)

    num_patches_per_side = image_size // patch_size
    region_size = num_patches_per_side
    row, col = 0, 0
    for idx in path:
        region_size = region_size // 2
        row += (idx // 2) * region_size
        col += (idx  % 2) * region_size

    px = col * patch_size
    py = row * patch_size
    pw = ph = region_size * patch_size
    cx = px + pw / 2
    cy = py + ph / 2

    rect = mpatches.Rectangle((px, py), pw, ph,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax_img.add_patch(rect)
    ax_img.annotate("", xy=(cx, cy),
                    xytext=(cx - image_size * 0.3, cy - image_size * 0.3),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))

    title = class_name if class_name else f"path {path}"
    ax_img.set_title(title, fontsize=13)
    ax_img.axis('off')

    # ── right panel: tree diagram ────────────────────────────────────────────
    ax_tree = fig.add_axes([0.38, 0.0, 0.62, 1.0])
    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.axis('off')

    num_levels = len(all_scores)   # e.g. 3

    # level y positions — top to bottom
    y_positions = [0.85 - i * (0.7 / (num_levels - 1))
                   for i in range(num_levels)]

    # x positions for the 4 children at each level
    x_sets = [
        [0.2, 0.4, 0.6, 0.8],   # level 0 (top)
        [0.2, 0.4, 0.6, 0.8],   # level 1
        [0.2, 0.4, 0.6, 0.8],   # level 2 (bottom)
    ]

    node_positions = []   # (x, y) for each node at each level

    for level_idx, (scores, y) in enumerate(zip(all_scores, y_positions)):
        xs      = x_sets[level_idx]
        best    = path[level_idx]
        pos_row = []

        for child_idx, (score, x) in enumerate(zip(scores, xs)):
            selected = (child_idx == best)
            color    = '#D32F2F' if selected else '#BDBDBD'
            ec       = '#B71C1C' if selected else '#9E9E9E'
            fc_text  = 'white'   if selected else '#333333'

            ellipse = mpatches.Ellipse((x, y), width=0.14, height=0.10,
                                       facecolor=color, edgecolor=ec,
                                       linewidth=1.5, zorder=3)
            ax_tree.add_patch(ellipse)
            ax_tree.text(x, y, f"{score:.2f}",
                         ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         color=fc_text, zorder=4)
            pos_row.append((x, y))

        node_positions.append(pos_row)

    # draw edges between levels
    for level_idx in range(num_levels - 1):
        best_parent = path[level_idx]
        parent_x, parent_y = node_positions[level_idx][best_parent]
        child_y = y_positions[level_idx + 1]

        for child_idx, (child_x, _) in enumerate(node_positions[level_idx + 1]):
            child_selected = (child_idx == path[level_idx + 1])
            color = 'red' if child_selected else '#CCCCCC'
            lw    = 2.0   if child_selected else 0.8

            ax_tree.annotate("",
                xy=(child_x, child_y + 0.05),
                xytext=(parent_x, parent_y - 0.05),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                zorder=2)

    level_labels = ['hierarchy 3 (top)', 'hierarchy 2', 'hierarchy 1 (leaf)']
    for label, y in zip(level_labels, y_positions):
        ax_tree.text(0.02, y, label, ha='left', va='center',
                     fontsize=9, color='#666666')

    plt.savefig("gradcat_output.png", dpi=150, bbox_inches='tight')
    plt.show()
