"""FPP RGB-only or sensorimotor-demo rollouts using the same official wrappers as BPP calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import imageio.v2 as imageio
import numpy as np
import timm
import torch

from fpp_libero_gen.data import sample_demo
from fpp_libero_gen.model import VideoPrefixFPP, VideoKVCache
from fpp_libero_gen.official import configure, make_env, protocol, reset
from fpp_libero_gen.prompt_controls import CONDITIONS, transform_demo
from fpp_libero_gen.sensorimotor import SensorimotorCache, transform_prompt
from fpp_libero_mem.data import denormalize
from fpp_libero_gen.ema import select_weights, weight_sha256
from fpp_libero_gen.execution import ExecutionCache, policy_proprio, native_action
from scripts.prepare_cache import _features


def main(args):
    official = configure(args.bpp_root, args.libero_root, args.raw_root, args.output/'libero_config')
    raw = (args.data_root/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if hashlib.sha256(raw).hexdigest() != saved['data_manifest_sha256']:
        raise ValueError('Training data identity differs')
    if manifest['history_stride'] != 12:
        raise ValueError('Official action execution requires stride-12 training')
    use_bpp = saved['config'].get('demo_input') == 'bpp'
    if use_bpp != bool(args.demo_cache_root):
        raise ValueError('BPP demo checkpoint requires its exact --demo-cache-root')
    demo_cache = SensorimotorCache(args.demo_cache_root, saved['data_manifest_sha256']) if use_bpp else None
    if demo_cache is not None and demo_cache.sha256 != saved['demo_cache_manifest_sha256']:
        raise ValueError('Training/evaluation demo cache identity differs')
    use_execution = saved['config'].get('query_input') == 'bpp'
    if use_execution != bool(args.execution_cache_root):
        raise ValueError('Execution checkpoint requires exact --execution-cache-root')
    execution_cache = ExecutionCache(args.execution_cache_root, saved['data_manifest_sha256'], demo_cache.sha256 if demo_cache else None) if use_execution else None
    if execution_cache and execution_cache.sha256 != saved['execution_cache_manifest_sha256']:
        raise ValueError('Training/evaluation execution cache differs')
    device = torch.device(args.device)
    model = VideoPrefixFPP(saved['config']).to(device).eval()
    weights = select_weights(saved, args.weights)
    model.load_state_dict(weights)
    if hashlib.sha256(args.dino_checkpoint.read_bytes()).hexdigest() != saved['dino_checkpoint_sha256']:
        raise ValueError('DINO identity differs')
    dino = timm.create_model(saved['dino_model'], pretrained=False,
        checkpoint_path=str(args.dino_checkpoint), dynamic_img_size=True).to(device).eval()
    support = {(e['task'], e['episode']): e for e in manifest['episodes'] if e['split'] == 'test_support'}
    all_tasks = sorted({r['task'] for r in manifest['evaluation']})
    tasks = all_tasks[:args.max_tasks] if args.max_tasks else all_tasks
    tasks = tasks[args.shard::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.output/f'trials_shard{args.shard}.jsonl'
    if result.exists():
        raise FileExistsError('Use a fresh evaluation directory')
    details = protocol(args.bpp_root)
    details.update(condition=args.condition, checkpoint=str(args.checkpoint), checkpoint_step=saved['step'],
        training_source_revision=saved['source_revision'], evaluator_source_revision=args.source_revision,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=saved['data_manifest_sha256'], trials_per_task=args.trials,
        policy_inputs='Support RGB only and query RGB history only',
        prefix_control='Last frame repeats to full prefix length; shuffled reorders the same frames with a private RNG; blank zeros features (OOD).',
        rng_seed='4242 + 1000*sorted_task_index + official_init_index',
        flow_steps=args.flow_steps, visual_backbone=saved['dino_model'], visual_backbone_sha256=saved['dino_checkpoint_sha256'])
    details.update(weight_source=args.weights, weight_sha256=weight_sha256(weights))
    if args.weights == 'ema':
        details['ema'] = {k: v for k, v in saved['ema'].items() if k != 'model'}
    if use_bpp:
        details.update(policy_inputs='Support two-view RGB, action10 chunks and proprio11; query single-view RGB history only',
            demo_cache_manifest_sha256=demo_cache.sha256,
            prefix_control='All demo modalities change together. Last_frame repeats the final real chunk; shuffled reorders real chunks; blank zeros all modalities; padding mask preserved.')
    if use_execution:
        details.update(policy_inputs='Support dual-view RGB/action10/proprio11; query dual-view RGB/proprio11 full causal history',
                       execution_cache_manifest_sha256=execution_cache.sha256, native_action_dim=7, policy_action_dim=10,
                       history_capacity=saved['config']['max_steps'], query_stride=12)
    (args.output/f'protocol_shard{args.shard}.json').write_text(json.dumps(details, indent=2))
    rows = []
    for task in tasks:
        plan = sorted([x for x in manifest['evaluation'] if x['task'] == task], key=lambda x: x['init_index'])[:args.trials]
        env = make_env(official/plan[0]['bddl'])
        try:
            for trial in plan:
                index = trial['init_index']
                seed = 4242 + 1000*all_tasks.index(task) + index
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                prompt_task = all_tasks[(all_tasks.index(task)+1) % len(all_tasks)] if args.condition == 'wrong' else task
                prompt = support[prompt_task, f'demo_{index % 10}']
                if use_bpp:
                    prompt_tensors, prompt_frame_order = transform_prompt(demo_cache.load(prompt['id']), args.condition, seed)
                    demo = prompt_tensors['demo']
                else:
                    demo = sample_demo(np.load(args.data_root/prompt['cache']/prompt['demo_feature_file'], mmap_mode='r'), saved['config']['demo_frames'])
                    demo, prompt_frame_order = transform_demo(demo, args.condition, seed)
                    prompt_tensors = dict(demo=demo)
                cache = VideoKVCache(model.visual)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    cache.reset(**{k:torch.from_numpy(v).to(device=device,
                        dtype=torch.bool if v.dtype==np.bool_ else torch.float32)[None] for k,v in prompt_tensors.items()})
                initial = np.load(args.data_root/trial['initial_state'])
                assert hashlib.sha256(initial.tobytes()).hexdigest() == trial['initial_state_sha256']
                obs, reward, done, _ = reset(env, initial)
                from OpenGL.GL import glGetString, GL_RENDERER
                renderer = glGetString(GL_RENDERER).decode()
                frames, began = [], time.monotonic()
                with torch.inference_mode():
                    while not done:
                        rgb = np.ascontiguousarray(obs['agentview_image'][-1])
                        if index < args.videos:
                            frames.append(rgb)
                        pixels = np.stack((rgb, np.ascontiguousarray(obs['robot0_eye_in_hand_image'][-1]))) if use_execution else rgb[None]
                        features = _features(dino, pixels, device, 1)
                        query_proprio = None
                        if use_execution:
                            features = features[None]
                            prop = execution_cache.normalize_proprio(policy_proprio(obs))
                            query_proprio = torch.from_numpy(prop).to(device)[None]
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                            memory = cache.encode(torch.from_numpy(features).float().to(device), query_proprio=query_proprio)
                            chunk = model.sample_action(memory, cache, steps=args.flow_steps)[0].float().cpu().numpy()
                        chunk = denormalize(chunk, saved['normalization'])
                        if use_execution:
                            chunk = native_action(chunk)
                        chunk = np.clip(chunk, -1, 1)
                        if not np.isfinite(chunk).all():
                            raise FloatingPointError('Nonfinite policy action')
                        obs, reward, done, _ = env.step(chunk[:12])
                        if reward == 1:
                            done = True
                video = f'task{all_tasks.index(task):03d}_init{index:02d}.mp4' if frames else None
                row = dict(task=task, init_index=index, episode=f'official_init_{index:02d}',
                    initial_state_sha256=trial['initial_state_sha256'], support_id=prompt['id'],
                    seed=seed, environment_seed=7, condition=args.condition, success=bool(max(env.reward) == 1),
                    prompt_frame_order=prompt_frame_order,
                    prompt_features_sha256=hashlib.sha256(demo.tobytes()).hexdigest(),
                    policy_controls=max(0, len(env.reward)-12), total_controls=len(env.reward),
                    elapsed_s=time.monotonic()-began, renderer=renderer, video=video)
                rows.append(row)
                if use_bpp:
                    row['demo_cache_manifest_sha256'] = demo_cache.sha256
                    row['prompt_modalities_sha256'] = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in prompt_tensors.items()}
                with result.open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                if video:
                    imageio.mimsave(args.output/video, frames, fps=20/12)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    (args.output/f'summary_shard{args.shard}.json').write_text(json.dumps(dict(
        trials=len(rows), successes=sum(r['success'] for r in rows)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data-root', 'raw-root', 'bpp-root', 'libero-root', 'checkpoint', 'dino-checkpoint', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--demo-cache-root', type=Path)
    p.add_argument('--execution-cache-root', type=Path)
    p.add_argument('--condition', choices=CONDITIONS, default='correct')
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--flow-steps', type=int, default=10)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--videos', type=int, default=1)
    p.add_argument('--source-revision', required=True)
    p.add_argument('--weights', choices=['raw', 'ema'], default='raw')
    main(p.parse_args())
