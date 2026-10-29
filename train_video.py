"""Train RGB-only or BPP-demo-prefix FPP on a task-disjoint LIBERO-Gen feature cache."""
import argparse
from functools import partial
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fpp_libero_gen.data import PairedEpisodes, collate
from fpp_libero_gen.model import VideoPrefixFPP
from fpp_libero_gen.sensorimotor import SensorimotorCache
from fpp_libero_gen.ema import ModelEMA
from fpp_libero_gen.execution import ExecutionCache
from fpp_libero_gen.stability import FiniteInstabilityGuard
from fpp_libero_gen.diagnostics import collate_with_ids, module_norms, update_diagnostics
from fpp_libero_mem.distributed import initialize, global_objective


def build_config(manifest, args, demo_manifest=None):
    config = dict(feature_dim=manifest['feature_dim'], grid_size=manifest['grid_size'],
        demo_grid_size=args.demo_grid, demo_frames=args.demo_frames,
        max_steps=manifest['max_steps'], visual_dim=args.width,
        visual_heads=8, visual_layers=args.visual_layers, ffn_ratio=4,
        checkpoint_visual=True, decoder_dim=args.width, decoder_heads=8,
        decoder_layers=args.decoder_layers, horizon=manifest['horizon'], action_dim=7)
    if getattr(args, 'direct_demo', False):
        config['direct_demo'] = True
        config['checkpoint_direct'] = True
    if getattr(args, 'shared_demo_kv', False):
        if not getattr(args, 'direct_demo', False):
            raise ValueError('Shared demo K/V requires --direct-demo')
        config['shared_demo_kv'] = True
    if getattr(args, 'demo_full_patches', False):
        if args.demo_input != 'bpp':
            raise ValueError('Full demo patches require --demo-input bpp')
        config['demo_full_patches'] = True
    if getattr(args, 'demo_pooling', 'average') != 'average':
        config['demo_pooling'] = args.demo_pooling
    if getattr(args, 'demo_input', 'rgb') == 'bpp':
        if demo_manifest is None:
            raise ValueError('BPP input requires a prepared demo cache')
        config.update(demo_input='bpp',demo_frames=demo_manifest['max_chunks'],
                      demo_chunk_size=demo_manifest['chunk_size'])
    if getattr(args, 'query_input', 'rgb') == 'bpp':
        config.update(query_input='bpp', action_dim=10)
    if getattr(args, 'action_qk_norm', False):
        config['action_qk_norm'] = True
    if getattr(args, 'action_fp32_residual', False):
        config['action_fp32_residual'] = True
    if getattr(args, 'query_wrist_pool', 'full') == 'learned':
        if args.demo_pooling != 'learned' or args.query_input != 'bpp':
            raise ValueError('Learned wrist pooling requires learned demo pooling and BPP query inputs')
        config['query_wrist_pool'] = 'learned'
    if getattr(args, 'checkpoint_action', False):
        config['checkpoint_action'] = True
    return config


def main(args):
    if min(args.steps, args.batch, args.target_steps, args.save_every, args.log_every) < 1:
        raise ValueError('Training counts must be positive')
    if args.diagnostics_every < 0:
        raise ValueError('Diagnostics interval cannot be negative')
    if args.ema_decay is not None:
        ModelEMA(args.ema_decay, args.ema_start)
    context = initialize(args.device, args.batch)
    device = context.device
    manifest_bytes = (args.data_root/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if (args.demo_input == 'bpp') != bool(args.demo_cache_root):
        raise ValueError('Use --demo-input bpp together with --demo-cache-root')
    demo_cache = SensorimotorCache(args.demo_cache_root, hashlib.sha256(manifest_bytes).hexdigest()) if args.demo_cache_root else None
    execution_cache = ExecutionCache(args.execution_cache_root, hashlib.sha256(manifest_bytes).hexdigest(), demo_cache.sha256 if demo_cache else None) if args.execution_cache_root else None
    if (args.query_input == 'bpp') != bool(execution_cache):
        raise ValueError('Use --query-input bpp with --execution-cache-root')
    normalization = execution_cache.manifest['normalization']['action'] if execution_cache else manifest['normalization']
    config = build_config(manifest, args, None if demo_cache is None else demo_cache.manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    raw_model = VideoPrefixFPP(config).to(device)
    model = DistributedDataParallel(raw_model, device_ids=[device.index],
        broadcast_buffers=False) if context.world_size > 1 else raw_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    ema = ModelEMA(args.ema_decay, args.ema_start) if args.ema_decay is not None and context.rank == 0 else None
    guard = FiniteInstabilityGuard() if args.stability_guard else None
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['config'] != config or saved['data_manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError('Resume architecture/data changed')
        if saved.get('demo_cache_manifest_sha256') != (demo_cache.sha256 if demo_cache else None):
            raise ValueError('Resume demo cache changed')
        raw_model.load_state_dict(saved['model'])
        if saved.get('execution_cache_manifest_sha256') != (execution_cache.sha256 if execution_cache else None):
            raise ValueError('Resume execution cache changed')
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        if guard is not None:
            if saved.get('finite_stability_guard') is None:
                raise ValueError('Resume cannot add a new finite-instability guard history')
            guard.load_state_dict(saved['finite_stability_guard'])
        if (saved.get('ema') is not None) != (args.ema_decay is not None):
            raise ValueError('Resume cannot add/remove historical EMA')
        if args.ema_decay is not None:
            if saved['ema']['decay'] != args.ema_decay or saved['ema']['start_step'] != args.ema_start:
                raise ValueError('Resume EMA configuration changed')
            if saved['ema']['last_step'] != start:
                raise ValueError('EMA optimizer update count differs')
            if ema is not None:
                ema.load_state_dict(saved['ema'], raw_model)
    random.seed(args.seed + context.rank + start)
    np.random.seed(args.seed + context.rank + start)
    torch.manual_seed(args.seed + context.rank + start)
    dataset = PairedEpisodes(args.data_root, demo_frames=config['demo_frames'], target_steps=args.target_steps,
                             demo_cache_root=args.demo_cache_root, execution_cache_root=args.execution_cache_root)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True,
        num_samples=max(1, args.steps-start)*context.local_batch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=context.local_batch, sampler=sampler,
        collate_fn=partial(collate_with_ids if args.diagnostics_every else collate,
                           max_steps=manifest['max_steps']), num_workers=args.workers, pin_memory=device.type=='cuda',
        persistent_workers=args.workers > 0)
    if context.rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (args.output/'config.json').exists():
            raise FileExistsError('Use a fresh run directory or explicit --resume')
        run = dict(model=config, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            world_size=context.world_size, dataset_tasks=len(dataset.by_task), dataset_episodes=len(dataset),
            data_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            execution_cache_manifest_sha256=execution_cache.sha256 if execution_cache else None,
            conditioning=('support two-view RGB/action/proprio; query dual-view RGB+proprio full history' if execution_cache else
                          'support two-view RGB + action10 chunks + proprio11; query RGB history only' if demo_cache else
                          'support RGB only; query RGB history only'),
            demo_cache_manifest_sha256=demo_cache.sha256 if demo_cache else None,
            resume_semantics='Explicit model/optimizer/EMA restore; sampler/RNG restart is not bitwise replay',
            metric_note='Training loss is not closed-loop success rate')
        (args.output/'config.json').write_text(json.dumps(run, indent=2))
    if context.world_size > 1:
        dist.barrier()
    model.train()
    began = time.monotonic()
    clipped_updates = 0
    for step, batch in enumerate(loader, start+1):
        batch_ids = None
        if args.diagnostics_every:
            batch, batch_ids = batch
        optimizer.zero_grad(set_to_none=True)
        batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
        if device.type == 'cpu':
            batch['features'], batch['demo'] = batch['features'].float(), batch['demo'].float()
        failure_rng = dict(cpu=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None) if args.diagnostics_every else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            loss = model(**batch, action_microbatch=args.action_microbatch)
        objective, reported = global_objective(loss, batch['valid'])
        if not torch.isfinite(objective):
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/f'failure_rank{context.rank}.json').write_text(json.dumps(
                dict(step=step, reason='nonfinite_loss', batch_ids=batch_ids)))
            torch.save(dict(rng_before_forward=failure_rng, valid=batch['valid'].cpu(), batch_ids=batch_ids),
                       args.output/f'failure_rank{context.rank}.pt')
            raise FloatingPointError(f'Nonfinite loss at {step}')
        objective.backward()
        diagnose = context.rank == 0 and args.diagnostics_every and (step == start+1 or step % args.diagnostics_every == 0)
        diagnostic = module_norms(raw_model) if diagnose else None
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/f'failure_rank{context.rank}.json').write_text(json.dumps(
                dict(step=step, reason='nonfinite_gradient', batch_ids=batch_ids)))
            torch.save(dict(rng_before_forward=failure_rng, valid=batch['valid'].cpu(), batch_ids=batch_ids),
                       args.output/f'failure_rank{context.rank}.pt')
            raise FloatingPointError(f'Nonfinite gradient at {step}')
        warmup = min(1., step/max(1, args.warmup))
        decay = .1 + .9*.5*(1+math.cos(math.pi*min(1., max(0.,
            (step-args.warmup)/max(1,args.schedule_steps-args.warmup)))))
        lr = args.lr*warmup*decay
        for group in optimizer.param_groups:
            group['lr'] = lr
        before = [p.detach().clone() for p in raw_model.parameters()] if diagnose else None
        guard_triggered = False
        if guard is not None and step % args.log_every == 0:
            flag = torch.tensor(int(guard.observe(step, float(reported), float(grad))), device=device)
            if context.world_size > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            guard_triggered = bool(flag)
        if guard_triggered:
            args.output.mkdir(parents=True, exist_ok=True)
            reason = dict(step=step, reason='persistent_finite_instability', batch_ids=batch_ids,
                          loss=float(reported), preclip_gradient_norm=float(grad), best_window_loss=guard.best)
            (args.output/f'failure_rank{context.rank}.json').write_text(json.dumps(reason))
            torch.save(dict(step=step-1, model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                            ema=ema.state_dict() if ema else None, config=config, batch={k:v.cpu() for k,v in batch.items()},
                            rng_before_forward=failure_rng, source_revision=args.source_revision),
                       args.output/f'failure_rank{context.rank}.pt')
            raise FloatingPointError(f'Persistent finite instability at{step}; replay captured')
        optimizer.step()
        if ema is not None:
            ema.update(raw_model, step)
        if args.diagnostics_every:
            clipped_updates += int(float(grad) > 1.)
        if diagnose:
            diagnostic.update(update_diagnostics(raw_model, before, ema))
            diagnostic.update(step=step, clip_coefficient=min(1., 1./(float(grad)+1e-6)),
                              local_batch_ids=batch_ids)
            del before
            with (args.output/'diagnostics.jsonl').open('a') as handle:
                handle.write(json.dumps(diagnostic)+'\n')
        log_now = step == start+1 or step % args.log_every == 0 or step == args.steps
        if args.diagnostics_every and log_now:
            global_targets = batch['valid'].sum().to(dtype=torch.int64)
            if context.world_size > 1:
                dist.all_reduce(global_targets, op=dist.ReduceOp.SUM)
        if context.rank == 0 and log_now:
            row = dict(step=step, loss=float(reported), grad_norm=float(grad), lr=lr,
                elapsed_s=time.monotonic()-began, steps_per_s=(step-start)/(time.monotonic()-began),
                peak_vram_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else None)
            if args.diagnostics_every:
                row.update(episode_draws=step*args.batch, global_valid_targets=int(global_targets),
                           local_valid_targets=int(batch['valid'].sum()),
                           clipped_update_fraction=clipped_updates/(step-start),
                           peak_reserved_gb=torch.cuda.max_memory_reserved(device)/1e9 if device.type=='cuda' else None,
                           ema_updates=ema.updates if ema is not None else None)
            with (args.output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if context.rank == 0 and (step % args.save_every == 0 or step == args.steps):
            saved = dict(format='fpp_libero_gen_video_v1', step=step, config=config,
                model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                normalization=normalization, history_stride=manifest['history_stride'],
                dino_model=manifest['dino_model'], dino_checkpoint_sha256=manifest['dino_checkpoint_sha256'],
                data_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                source_revision=args.source_revision)
            if ema is not None:
                saved.update(training_state_version=2, ema=ema.state_dict())
            if guard is not None:
                saved['finite_stability_guard'] = guard.state_dict()
            if execution_cache is not None:
                saved['execution_cache_manifest_sha256'] = execution_cache.sha256
            if demo_cache is not None:
                saved['demo_cache_manifest_sha256'] = demo_cache.sha256
            temporary = args.output/f'step_{step:06d}.partial'
            torch.save(saved, temporary)
            temporary.replace(args.output/f'step_{step:06d}.pt')
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--schedule-steps', type=int, default=20000)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--visual-layers', type=int, default=6)
    p.add_argument('--decoder-layers', type=int, default=8)
    p.add_argument('--demo-frames', type=int, default=32)
    p.add_argument('--demo-grid', type=int, default=7)
    p.add_argument('--demo-input', choices=['rgb','bpp'], default='rgb')
    p.add_argument('--demo-cache-root', type=Path)
    p.add_argument('--execution-cache-root', type=Path)
    p.add_argument('--query-input', choices=['rgb','bpp'], default='rgb')
    p.add_argument('--query-wrist-pool', choices=['full','learned'], default='full')
    p.add_argument('--action-qk-norm', action='store_true')
    p.add_argument('--action-fp32-residual', action='store_true')
    p.add_argument('--checkpoint-action', action='store_true')
    p.add_argument('--stability-guard', action='store_true')
    p.add_argument('--direct-demo', action='store_true')
    p.add_argument('--shared-demo-kv', action='store_true')
    p.add_argument('--demo-full-patches', action='store_true')
    p.add_argument('--demo-pooling', choices=['average', 'learned'], default='average')
    p.add_argument('--target-steps', type=int, default=32)
    p.add_argument('--action-microbatch', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--source-revision')
    p.add_argument('--ema-decay', type=float, help='Optional FP32 parameter EMA; disabled by default')
    p.add_argument('--ema-start', type=int, default=2000)
    p.add_argument('--diagnostics-every', type=int, default=0, help='Optimizer updates between diagnostics; zero disables')
    main(p.parse_args())
