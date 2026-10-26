"""
scene_perturbation.py - Deterministic 3D Gaussian Scene Perturbation Engine

Provides reproducible, controlled modifications to a subset of 3D Gaussians
(e.g., rigid translation of 1% of the scene), tracking modified indices
and preserving unmodified scene parameters for baseline comparisons.
"""

from typing import Dict, List, Tuple, Union
import torch


def perturb_scene(
    scene: Dict[str, Union[torch.Tensor, int]],
    fraction: float = 0.01,
    delta: Tuple[float, float, float] = (0.15, -0.10, 0.05),
    seed: int = 42,
    mode: str = "random_subset",
) -> Tuple[Dict[str, Union[torch.Tensor, int]], torch.Tensor, Dict]:
    """
    Applies a deterministic 3D translation to a specified fraction of Gaussians.

    Args:
        scene: Original scene dictionary containing 'means', 'scales', 'quats', etc.
        fraction: Fraction of total Gaussians to modify (e.g. 0.01 = 1%).
        delta: 3D translation vector (dx, dy, dz) to apply to target Gaussians.
        seed: Random seed for deterministic subset selection.
        mode: Selection mode ("random_subset", "spatial_cluster", or "single").

    Returns:
        perturbed_scene: New dictionary with modified parameters (original remains untouched).
        changed_indices: 1D Tensor of integer indices of the modified Gaussians.
        metadata: Detailed record of the perturbation parameters and coordinates.
    """
    # Create an isolated clone of the scene dictionary and tensors
    perturbed_scene = {}
    for k, v in scene.items():
        if isinstance(v, torch.Tensor):
            perturbed_scene[k] = v.clone()
        else:
            perturbed_scene[k] = v

    means = perturbed_scene["means"]
    num_total = means.shape[0]
    device = means.device

    # Deterministic index selection
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if mode == "single":
        num_changed = 1
        changed_indices = torch.tensor([seed % num_total], device=device, dtype=torch.long)
    elif mode == "spatial_cluster":
        # Select Gaussians within a localized 3D radius around a seed point
        num_changed = max(1, int(round(num_total * fraction)))
        center_idx = torch.randint(0, num_total, (1,), generator=gen).item()
        center_pt = means[center_idx:center_idx + 1]
        distances = torch.norm(means - center_pt, dim=-1)
        _, sorted_indices = torch.sort(distances)
        changed_indices = sorted_indices[:num_changed].to(device)
    else:  # default "random_subset"
        num_changed = max(1, int(round(num_total * fraction)))
        perm = torch.randperm(num_total, generator=gen)
        changed_indices = perm[:num_changed].to(device)

    # Ensure changed_indices is sorted for consistent downstream processing
    changed_indices, _ = torch.sort(changed_indices)

    # Apply 3D translation
    delta_tensor = torch.tensor(delta, dtype=means.dtype, device=device)
    orig_coords = means[changed_indices].clone()
    means[changed_indices] += delta_tensor
    new_coords = means[changed_indices].clone()

    metadata = {
        "num_total_gaussians": num_total,
        "num_changed_gaussians": num_changed,
        "fraction_changed": num_changed / num_total,
        "selection_mode": mode,
        "seed": seed,
        "delta": list(delta),
        "changed_indices": changed_indices.cpu().tolist(),
        "original_means_sample": orig_coords[:5].cpu().tolist(),
        "perturbed_means_sample": new_coords[:5].cpu().tolist(),
    }

    return perturbed_scene, changed_indices, metadata


def verify_perturbation_integrity(
    original_scene: Dict[str, Union[torch.Tensor, int]],
    perturbed_scene: Dict[str, Union[torch.Tensor, int]],
    changed_indices: torch.Tensor,
    expected_delta: Tuple[float, float, float],
) -> bool:
    """
    Validates that:
    1. Gaussians NOT in changed_indices are bitwise identical.
    2. Gaussians in changed_indices reflect the exact expected delta.
    3. All other parameter tensors (scales, quats, colors, opacities) are unchanged.
    """
    orig_means = original_scene["means"].to(changed_indices.device)
    pert_means = perturbed_scene["means"].to(changed_indices.device)
    num_total = orig_means.shape[0]

    # Create boolean mask of modified elements
    changed_mask = torch.zeros(num_total, dtype=torch.bool, device=changed_indices.device)
    changed_mask[changed_indices] = True

    # 1. Untouched elements must have 0.0 difference
    untouched_diff = (pert_means[~changed_mask] - orig_means[~changed_mask]).abs().max().item()
    if untouched_diff != 0.0:
        raise AssertionError(f"Untouched Gaussians were modified! Max diff: {untouched_diff}")

    # 2. Touched elements must match expected delta exactly
    expected_d = torch.tensor(expected_delta, dtype=orig_means.dtype, device=changed_indices.device)
    actual_d = pert_means[changed_mask] - orig_means[changed_mask]
    delta_err = (actual_d - expected_d).abs().max().item()
    if delta_err > 1e-6:
        raise AssertionError(f"Perturbation delta mismatch! Max error: {delta_err}")

    # 3. Check other parameter tensors
    for key in ["scales", "quats", "opacities", "colors"]:
        if key in original_scene and key in perturbed_scene:
            diff = (perturbed_scene[key].to(changed_indices.device) - original_scene[key].to(changed_indices.device)).abs().max().item()
            if diff != 0.0:
                raise AssertionError(f"Parameter tensor '{key}' was unintentionally altered! Diff: {diff}")

    return True
