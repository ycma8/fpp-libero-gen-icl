"""Distributed training with a global episode batch and global token mean."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrainingContext:
    rank: int
    world_size: int
    device: torch.device
    local_batch: int


def initialize(device_name: str, global_batch: int) -> TrainingContext:
    """Initialize torchrun, preserving the meaning of the global batch.

    Returns:
        Rank, device, and per-rank episode batch.

    Raises:
        ValueError: The global batch cannot be divided between ranks.
        RuntimeError: CUDA was requested but is unavailable.
    """
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world < 1 or global_batch < 1 or global_batch % world:
        raise ValueError('Global --batch must be positive and divisible by WORLD_SIZE')
    device = torch.device(device_name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        device = torch.device('cuda', local_rank if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return TrainingContext(rank, world, device, global_batch // world)


def global_objective(loss: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight ragged rank losses by their number of real observation targets.

    DDP averages gradients between ranks. Multiplying by world_size * local /
    global target count recovers the single global-batch objective even when
    episode lengths differ between ranks. Horizon/action dimensions are fixed.

    Returns:
        Differentiable local objective and detached global mean for logging.

    Raises:
        ValueError: No valid targets exist on this rank.
    """
    count = valid.sum().to(device=loss.device, dtype=torch.float32)
    if count.item() < 1:
        raise ValueError('No valid action supervision on this rank')
    if not dist.is_initialized():
        return loss, loss.detach()
    totals = torch.stack((count, loss.detach().float() * count))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return loss * (dist.get_world_size() * count / totals[0]), totals[1] / totals[0]
