"""Training-free VDS + PPR + STR; no imports from the research repository.

DPP selection follows the research implementation's incremental Cholesky
solver (Chen et al., 2018; used by CDPruner). PPR follows GAP. See NOTICE.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VPRuneConfig:
    budget: float = 0.3
    gamma: float = 0.3
    tau: float = 0.1

    def __post_init__(self):
        if not math.isfinite(self.budget) or not 0 < self.budget <= 1:
            raise ValueError("budget must be finite and in (0, 1]")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.tau) or self.tau < 1e-6:
            raise ValueError("tau must be finite and >= 1e-6 (numerical stability floor)")


@dataclass
class PruningResult:
    features: torch.Tensor
    keep_indices: torch.Tensor
    original_tokens: int
    updated_tokens: int
    pruning_device: str


def _validate_features(features):
    if features.ndim != 3 or features.shape[0] != 1 or min(features.shape[1:]) < 1:
        raise ValueError("visual features must have shape (1, N, D) with N,D > 0")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("visual features must be finite floating-point values")


def _validate_keep(keep, n):
    if keep.ndim != 1 or keep.dtype != torch.long or keep.numel() == 0:
        raise ValueError("keep indices must be a nonempty 1-D torch.long tensor")
    if keep[0] < 0 or keep[-1] >= n or not torch.all(keep[1:] > keep[:-1]):
        raise ValueError("keep indices must be sorted, unique and within [0,N)")


@torch.no_grad()
def greedy_dpp_map(kernel: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy DPP MAP, float64 incremental Cholesky; exact-K even at low rank.

When the residual volume is exhausted (<=1e-10), fill in original index
order. This preserves the research implementation's deterministic behavior.
No global approximation guarantee is asserted for arbitrary log-det kernels.
"""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] == 0:
        raise ValueError("kernel must be a nonempty square matrix")
    if not isinstance(k, int) or not 1 <= k <= kernel.shape[0]:
        raise ValueError("k must be an integer in [1,N]")
    if not torch.isfinite(kernel).all():
        raise ValueError("kernel must be finite")
    device = kernel.device
    # Apple MPS has no float64. Keep this solver's reference arithmetic on CPU.
    kernel = kernel.to(device="cpu" if device.type == "mps" else device, dtype=torch.float64)
    n = kernel.shape[0]
    cis = torch.zeros((k, n), dtype=torch.float64, device=kernel.device)
    diagonal = kernel.diagonal().clone()
    chosen = torch.zeros(n, dtype=torch.bool, device=kernel.device)
    selected = []
    j = int(diagonal.argmax().item())
    for step in range(k):
        selected.append(j)
        chosen[j] = True
        if step == k - 1:
            break
        if diagonal[j] <= 1e-10:
            rest = (~chosen).nonzero(as_tuple=True)[0][:k - len(selected)]
            selected.extend(rest.tolist())
            break
        update = (kernel[j] - cis[:step, j] @ cis[:step]) / diagonal[j].sqrt()
        cis[step] = update
        diagonal = diagonal - update.square()
        diagonal[chosen] = -float("inf")
        j = int(diagonal.argmax().item())
    return torch.tensor(selected, device=device, dtype=torch.long)


@torch.no_grad()
def visual_only_select(features: torch.Tensor, budget: float):
    """Cosine Gram kernel from post-projector features. No text/quality term."""
    _validate_features(features)
    VPRuneConfig(budget=budget)
    n = features.shape[1]
    k = max(1, min(n, round(n * budget)))
    v = F.normalize(features[0].float(), dim=-1, eps=1e-6)
    kernel = v @ v.T
    keep = (torch.arange(n, device=features.device) if k == n
            else greedy_dpp_map(kernel, k).sort().values)
    return keep, kernel


@torch.no_grad()
def recycle_tokens(features, keep, kernel, gamma=0.3, tau=0.1):
    """Nearest retained feature, per-destination softmax, convex fusion.

Features with no assigned discards remain unchanged. Positions never merge.
GPU half-precision scatter accumulation can have small rounding variation.
"""
    _validate_features(features)
    VPRuneConfig(gamma=gamma, tau=tau)
    n = features.shape[1]
    _validate_keep(keep, n)
    if kernel.shape != (n, n) or kernel.device != features.device or keep.device != features.device:
        raise ValueError("features, indices and NxN kernel must share a device")
    if not torch.isfinite(kernel).all():
        raise ValueError("kernel must be finite")
    kept = features[0, keep]
    if gamma == 0 or keep.numel() == n:
        return kept.unsqueeze(0)
    mask = torch.ones(n, dtype=torch.bool, device=features.device)
    mask[keep] = False
    drop = mask.nonzero(as_tuple=True)[0]
    sim = kernel[drop][:, keep]
    owner = sim.argmax(dim=1)
    logits = sim.gather(1, owner[:, None]).squeeze(1) / tau
    k = keep.numel()
    maximum = torch.full((k,), -float("inf"), device=logits.device, dtype=logits.dtype)
    maximum = maximum.scatter_reduce(0, owner, logits, reduce="amax", include_self=True)
    exp = (logits - maximum[owner]).exp()
    den = torch.zeros(k, device=exp.device, dtype=exp.dtype).scatter_add_(0, owner, exp)
    weights = exp / den[owner].clamp_min(1e-9)
    merged = torch.zeros_like(kept)
    merged.index_add_(0, owner, weights[:, None].to(kept.dtype) * features[0, drop])
    count = torch.zeros(k, device=exp.device, dtype=exp.dtype).scatter_add_(0, owner, torch.ones_like(exp))
    g = torch.where(count > 0, torch.as_tensor(gamma, dtype=kept.dtype, device=kept.device),
                    torch.zeros((), dtype=kept.dtype, device=kept.device))[:, None]
    return ((1 - g) * kept + g * merged).unsqueeze(0)


def original_position_ids(num_before, keep, num_visual, num_after):
    """Compact sequence, gapped original IDs; suffix follows N, not K."""
    if min(num_before, num_after) < 0 or num_visual < 1:
        raise ValueError("invalid sequence lengths")
    _validate_keep(keep, num_visual)
    return torch.cat([
        torch.arange(num_before, device=keep.device), num_before + keep,
        torch.arange(num_before + num_visual, num_before + num_visual + num_after, device=keep.device),
    ]).unsqueeze(0)


@torch.no_grad()
def prune_visual_tokens(features, config=None):
    """VDS + STR on post-projector features, batch size 1.

MPS runs the small pruning stage on CPU (float64 DPP support), then returns
features to MPS. CUDA/CPU remain on device. This is not a latency benchmark.
"""
    config = config or VPRuneConfig()
    _validate_features(features)
    device = features.device
    work = features.cpu() if device.type == "mps" else features
    keep, kernel = visual_only_select(work, config.budget)
    fused = recycle_tokens(work, keep, kernel, config.gamma, config.tau)
    changed = int((fused != work[:, keep]).any(dim=-1).sum().item())
    return PruningResult(fused.to(device), keep.to(device), features.shape[1], changed, str(work.device))
