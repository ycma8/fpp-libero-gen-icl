"""Use the pinned benchmark's environment and multi-step wrappers verbatim."""
import hashlib
import os
from pathlib import Path
import sys


def configure(bpp_root, libero_root, raw_root, config_dir):
    bpp_root, libero_root = Path(bpp_root).resolve(), Path(libero_root).resolve()
    sys.path[:0] = [str(bpp_root), str(libero_root)]
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    official = bpp_root/'behavior_prompting/train_network/env/libero'
    paths = dict(benchmark_root=libero_root/'libero/libero', assets=libero_root/'libero/libero/assets',
                 bddl_files=official/'bddl_files', init_states=official/'init_files',
                 datasets=raw_root/'demonstration_data')
    temporary = config_dir/f'config.{os.getpid()}.partial'
    temporary.write_text(''.join(f'{k}: {v}\n' for k, v in paths.items()))
    temporary.replace(config_dir/'config.yaml')
    return official


def protocol(bpp_root):
    base = Path(bpp_root)/'behavior_prompting/train_network'
    paths = ['env_runner/libero_image_runner.py', 'env/libero/env_wrapper.py',
             'gym_util/multistep_wrapper.py', 'common/sampler.py']
    return dict(protocol='official Combination, 50 fixed pruned initial states per task',
        environment_seed=7, control_hz=20, open_gripper_steps=12,
        policy_control_budget=550, execution_chunk=12, prediction_horizon=16,
        support_assignment='demo_{initial_state_index % 10}, matching official n_envs=10 pair prompt order',
        reset='seed 7, reset with hard_reset=False, register official init state, reset, open gripper',
        fixture_or_camera_overrides=False, success='max of benchmark reward over the rollout',
        official_source_revision='96294c35705a18823ff2c850d37fbb1e12c21d85',
        libero_source_revision='8f1084e3132a39270c3a13ebe37270a43ece2a01',
        source_hashes={p: hashlib.sha256((base/p).read_bytes()).hexdigest() for p in paths})


def make_env(bddl, image_size=128):
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    from behavior_prompting.train_network.gym_util.multistep_wrapper import MultiStepWrapper
    control = ControlEnvForRunner(bddl_file_name=str(bddl), control_delta=True,
        camera_heights=image_size, camera_widths=image_size, has_offscreen_renderer=True,
        has_renderer=False, use_camera_obs=True)
    control.env.hard_reset = False
    return MultiStepWrapper(control, n_obs_steps=2, n_action_steps=12, max_episode_steps=562)


def reset(env, initial):
    import numpy as np
    control = env.env
    control.register_init_state(None)
    env.seed(7)
    env.reset()
    control.register_init_state(initial)
    env.reset()
    action = np.zeros((12, 7), dtype=np.float32)
    action[:, 6] = -1
    return env.step(action)
