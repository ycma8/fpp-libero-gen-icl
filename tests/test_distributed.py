"""Unequal episode lengths must not change the DDP global objective."""
import os
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from fpp_libero_mem.distributed import global_objective, initialize
from fpp_libero_mem.prefix_model import PrefixPatchDiT
from test_prefix_model import small_config


def weighted_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(.25)
    model = DistributedDataParallel(model)
    x = torch.tensor([[-2.], [0.], [1.], [2.]])
    target = torch.tensor([[3.], [1.], [0.], [-1.]])
    selected = slice(0, 3) if rank == 0 else slice(3, 4)
    local = (model(x[selected]) - target[selected]).square().mean()
    objective, mean = global_objective(local, torch.ones(len(x[selected]), dtype=torch.bool))
    objective.backward()
    reference = torch.tensor(.25, requires_grad=True)
    expected = (x * reference - target).square().mean()
    expected.backward()
    torch.testing.assert_close(mean, expected.detach())
    torch.testing.assert_close(model.module.weight.grad.flatten()[0], reference.grad)
    dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_ragged_two_rank_gradient_matches_one_global_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(weighted_worker, args=('file://' + str(Path(directory)/'rendezvous'),), nprocs=2)

    def test_global_batch_cannot_be_silently_multiplied(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}):
            with self.assertRaisesRegex(ValueError, 'divisible'):
                initialize('cpu', 3)

    def test_action_microbatch_preserves_loss_and_gradient(self):
        torch.manual_seed(10)
        first=PrefixPatchDiT(small_config())
        torch.nn.init.normal_(first.action.output_projection.weight, std=.02)
        for block in first.action.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
        second=copy.deepcopy(first)
        features=torch.randn(1,3,8,12);text=torch.randn(1,8)
        actions=torch.randn(1,3,4,7);valid=torch.ones(1,3,dtype=torch.bool)
        losses=[]
        for model,microbatch in [(first,1),(second,16)]:
            torch.manual_seed(123)
            loss=model(features,text,actions,valid,action_microbatch=microbatch)
            loss.backward();losses.append(loss.detach())
        torch.testing.assert_close(losses[0],losses[1])
        for left,right in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(left.grad,right.grad,atol=2e-6,rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
