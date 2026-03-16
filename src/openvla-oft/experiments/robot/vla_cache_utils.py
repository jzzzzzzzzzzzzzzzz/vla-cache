import os
import bisect
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks


@torch.no_grad()
def get_layer_mask_schedule(multihead_attention, apply_weighted_growth=True, growth_factor=0.55):
    """
    Computes per-layer reuse proportions based on normalized attention entropy.

    Args:
        multihead_attention (List[Tensor|None]): Attention maps per layer (shape: [1, heads, tokens, tokens]).
            Entries may be None for skipped layers (layer skip acceleration).
        apply_weighted_growth (bool): Whether to smooth upward deltas.
        growth_factor (float): Weight for smoothing.

    Returns:
        torch.Tensor: Layer-wise reuse proportions, shape (num_layers - 1,).
            Skipped-layer positions are filled with 0.0 (no reuse quota).
    """
    layers_attns = multihead_attention[:-1]  # exclude trailing cache_position entry
    n_layers = len(layers_attns)
    device = next((a.device for a in layers_attns if a is not None), torch.device('cpu'))

    entropies = []
    valid_indices = []
    for i, attn in enumerate(layers_attns):
        if attn is None:          # skipped layer — no attention captured
            continue
        attn = attn.mean(dim=1)[0]
        attn /= attn.sum(dim=-1, keepdim=True) + 1e-10
        attn = torch.nan_to_num(attn, nan=0.0)
        token_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
        entropies.append(token_entropy.mean())
        valid_indices.append(i)

    if not entropies:
        return torch.zeros(n_layers, device=device)

    entropies = torch.stack(entropies)
    norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse_valid = 1.0 - norm_entropy

    if apply_weighted_growth:
        reuse_list = reuse_valid.tolist()
        for i in range(1, len(reuse_list)):
            delta = reuse_list[i] - reuse_list[i - 1]
            if delta > 0:
                reuse_list[i] = reuse_list[i - 1] + delta * growth_factor
        reuse_valid = torch.tensor(reuse_list, dtype=torch.float32, device=device)

    # Build full-length tensor; skipped-layer positions stay 0.0
    full_reuse = torch.zeros(n_layers, device=device)
    for out_idx, layer_idx in enumerate(valid_indices):
        full_reuse[layer_idx] = reuse_valid[out_idx]

    return full_reuse

def patchify(image, patch_size=14):
    """
    Converts an image into non-overlapping patches.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches

def calculate_patch_similarity(patches1, patches2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim

def find_static_patches(img_0, img_1, patch_size=14, top_k=150, sim_threshold=0.996, return_similarity=False):
    """
    Identifies significant patches with high similarity across two images.
    If return_similarity=True, also returns the full (256,) cosine similarity array for all patches.
    """
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    similarity = calculate_patch_similarity(patches1, patches2)
    grid_size = 224 // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    patch_scores = [(i * grid_size + j, similarity_2d[i, j])
                    for i in range(grid_size) for j in range(grid_size)
                    if similarity_2d[i, j] >= sim_threshold]

    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]

    if return_similarity:
        return top_patch_ids, similarity  # similarity shape: (num_patches,)
    return top_patch_ids

@torch.no_grad()
def token_attention_merge(multihead_attention, layer_id=15, primary=True, b_positions=None):
    """
    Computes mean attention from text tokens to vision tokens.

    b_positions: if provided, the attention maps are in compact space (B tokens removed
                 before the LLM). All B positions must be in the vision range [1..512].
                 Returns a full 256-d vector with zeros for pruned patch positions.
    """
    attn_map = multihead_attention[layer_id].to(torch.float32).squeeze(0).mean(dim=0)

    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + 34

    attention_pos = multihead_attention[-1]

    if b_positions is not None and len(b_positions) > 0:
        # Compact mode: B tokens removed → all positions shift down by N_B.
        # All B positions are in [1..512] (vision range), so text tokens shift uniformly.
        n_b = len(b_positions)
        b_set = set(b_positions)
        b_sorted = sorted(b_positions)

        t_compact_start = t_token_start - n_b
        t_compact_end = t_token_end - n_b
        text_mask = (attention_pos >= t_compact_start) & (attention_pos < t_compact_end)

        # Compact indices of non-B vision tokens
        non_b_vision_orig = [p for p in range(v_token_start, v_token_end) if p not in b_set]
        non_b_vision_compact = [p - bisect.bisect_left(b_sorted, p) for p in non_b_vision_orig]

        if not non_b_vision_orig or not text_mask.any():
            return torch.zeros(256, dtype=torch.float32)

        non_b_compact_t = torch.tensor(non_b_vision_compact, dtype=torch.long)
        relation = attn_map[text_mask, :][:, non_b_compact_t]  # [N_text, N_non_B_vision]

        # Map back to full 256-d, zeros for B positions
        result = torch.zeros(256, dtype=attn_map.dtype, device=attn_map.device)
        non_b_local = [p - v_token_start for p in non_b_vision_orig]
        result[non_b_local] = relation.mean(dim=0)
        return result.cpu()

    # Full-sequence mode (original behavior)
    text_mask = (attention_pos >= t_token_start) & (attention_pos < t_token_end)
    relation = attn_map[text_mask, v_token_start:v_token_end]
    return relation.mean(dim=0).cpu()

def get_top_attention_patches(attn_scores, top_k=120):
    """
    Selects top-k patch indices based on attention scores.
    """
    attn_scores = attn_scores.cpu().numpy() if isinstance(attn_scores, torch.Tensor) else attn_scores
    attn = attn_scores.reshape(16, 16)
    attn_resized = cv2.resize(attn, (16, 16))

    flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in flat[:top_k]]

def draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4):
    """
    Draws colored overlays on image for different patch groups.
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            draw.rectangle([top_left, bottom_right], fill=color + (int(255 * alpha),))

    return Image.alpha_composite(image, overlay).convert("RGB")

def visualize_significant_patches_mask(image, patch_ids, patch_size=14, alpha=0.5, color=(255, 255, 255)):
    """
    Highlights specified patches with semi-transparent overlay.
    """
    overlay_group = [(patch_ids, color)]
    return draw_patches_overlay(image, overlay_group, patch_size, alpha)

def compute_preprune_mask(
    img_curr,
    img_prev,
    v_global,
    last_attn_scores,
    K_global=60,
    K_local=80,
    dynamic_threshold=0.99,
):
    """
    SpecPrune-VLA风格的三集合token selection (Cascade v3).

    保留集合 = V_global ∪ V_dynamic ∪ V_local
    剪除集合 = [0..255] - 保留集合

    Args:
        img_curr: 当前帧PIL图像（用于V_dynamic pixel-sim计算）
        img_prev: 上一帧PIL图像
        v_global: set[int] — 上一步保存的top-K patch索引（V_global，跨步时序保护）
                  如果为None则返回空剪除集合（step 0 warmup）
        last_attn_scores: torch.Tensor shape [256] — 当前步text→vision attention分数
                          用于V_local（当前最相关的top-K）
        K_global: V_global保留数量（已预先存在v_global中，此参数仅文档用途）
        K_local:  V_local保留数量
        dynamic_threshold: pixel-sim阈值，低于此值视为"动态"（V_dynamic）
                           注意：比VLA-Cache的0.996更宽松（宁可多留）

    Returns:
        prune_indices: list[int] — 要剪除的patch本地索引 [0..255]，已排序
        keep_indices:  list[int] — 要保留的patch本地索引 [0..255]，已排序
    """
    all_patches = set(range(256))

    # step 0 warmup: 无历史信息，不剪任何token
    if v_global is None or last_attn_scores is None:
        return [], sorted(all_patches)

    # --- V_global: 上一步的历史重要token（跨步时序保护）---
    # 直接使用传入的集合（由调用方每步更新）
    v_global_set = set(v_global)

    # --- V_dynamic: 当前帧中像素变化较大的patch ---
    # find_static_patches返回high-sim（静态）patch；其补集就是动态patch
    static_patches = set(find_static_patches(
        img_curr, img_prev,
        top_k=256,                    # 取尽可能多，只靠threshold筛选
        sim_threshold=dynamic_threshold,
    ))
    v_dynamic = all_patches - static_patches  # 变化的patch（宁可多留）

    # --- V_local: 当前步text-attn top-K（当前最相关区域）---
    scores = last_attn_scores
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    scores = scores.float()
    k = min(K_local, 256)
    v_local = set(scores.topk(k).indices.tolist())

    # --- 三集合并集 ---
    keep = v_global_set | v_dynamic | v_local
    prune = sorted(all_patches - keep)
    keep_sorted = sorted(keep)
    return prune, keep_sorted


def task_relevant_selection(multihead_attention, image, significant_patches, primary=True, top_k=100,
                            return_analysis=False, return_prune=False, attn_scores_override=None):
    """
    Highlights and compares significant patches with top attention patches.

    Token classes (all patch ids are in local [0, 255] space, NOT LLM-sequence space):
      Class A (Cache):    high pixel-sim AND low text-attention  = only_significant
      Class B (Prune):    low  pixel-sim AND low text-attention  = dynamic_irrelevant
      Class C (Recompute): high pixel-sim AND high text-attention = overlap
      Class D (Recompute): low  pixel-sim AND high text-attention = only_top

    If return_analysis=True, returns a third element: dict with full attn_scores and class ids.
    If return_prune=True, returns an extra list of Class B token positions in LLM sequence space.
    attn_scores_override: if provided (256-d tensor), skip token_attention_merge and use directly.
    """
    attn_score = (attn_scores_override if attn_scores_override is not None
                  else token_attention_merge(multihead_attention, primary=primary))
    top_patches = get_top_attention_patches(attn_score, top_k)

    only_significant = set(significant_patches) - set(top_patches)   # Class A
    overlap          = set(significant_patches) & set(top_patches)   # Class C
    only_top         = set(top_patches) - set(significant_patches)   # Class D
    all_patch_ids    = set(range(256))
    dynamic_irrel    = all_patch_ids - set(significant_patches) - set(top_patches)  # Class B

    patch_groups = [
        (significant_patches, (15, 67, 223)),
        (top_patches, (254, 55, 13)),
        (only_significant, (40, 116, 166)),
        (only_top, (241, 196, 15)),
        (overlap, (231, 76, 60)),
    ]

    result_image = draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4)

    v_token_start = 1 if primary else 257
    remaining = sorted([pid + v_token_start for pid in only_significant])
    # Class B positions in LLM sequence space (same offset convention as Class A)
    prune_positions = sorted([pid + v_token_start for pid in dynamic_irrel])

    if return_analysis:
        attn_np = attn_score.cpu().numpy() if hasattr(attn_score, 'cpu') else np.array(attn_score)
        analysis = {
            "attn_scores":  attn_np,                     # (256,) text-to-vision attention per patch
            "class_A_ids":  sorted(only_significant),    # Cache
            "class_B_ids":  sorted(dynamic_irrel),       # Prune candidate
            "class_C_ids":  sorted(overlap),             # Recompute (static+relevant)
            "class_D_ids":  sorted(only_top),            # Recompute (dynamic+relevant)
            "stable_count": len(significant_patches),    # patches with sim >= threshold
        }
        if return_prune:
            return np.array(result_image), remaining, analysis, prune_positions
        return np.array(result_image), remaining, analysis

    if return_prune:
        return np.array(result_image), remaining, prune_positions
    return np.array(result_image), remaining
