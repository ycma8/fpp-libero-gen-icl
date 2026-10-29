"""Full-horizon LIBERO-Mem rollout and explicit ordered-task metrics."""
import math
import numpy as np

SUBGOALS = [1, 1, 3, 3, 5, 7, 3, 4, 2, 2]


def reset_environment(env, initial_state, settle_steps):
    """Match collection's XML/simulator reset before restoring initial state."""
    env.reset()
    xml = env.sim.model.get_xml()
    current = env.sim.get_state().flatten().copy()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(current)
    env.sim.forward()
    obs = env.set_init_state(initial_state)
    env.env.reset_subgoal_progress()
    env.env._overshot = False
    for _ in range(settle_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs


def validate_trial_plan(manifest, plan):
    """Check explicit test membership separately from optional validation data."""
    expected = {(r['task'], f"{r['source']}:{r['episode']}"): r for r in plan}
    if len(expected) != len(plan):
        raise ValueError('Duplicate evaluation trial')
    declared = manifest.get('evaluation_trials')
    if declared is None:
        declared = [e for e in manifest['episodes'] if e['split'] == 'val']
    heldout = {(e['task'], e['source']): e for e in declared}
    if len(heldout) != len(declared) or set(heldout) != set(expected):
        raise ValueError('Raw holdout membership differs from prepared cache')
    training = {(e['task'], e['source']) for e in manifest['episodes'] if e['split'] == 'train'}
    if training & set(expected):
        raise ValueError('Evaluation IDs overlap training IDs')
    for key, episode in heldout.items():
        if 'initial_state_sha256' in episode and episode['initial_state_sha256'] != expected[key]['init_sha256']:
            raise ValueError('Evaluation initial-state hash differs from manifest')
        if 'horizon' in episode and episode['horizon'] != expected[key]['horizon']:
            raise ValueError('Evaluation horizon differs from manifest')


def rollout(env, policy, encode, text, initial_state, horizon, stride, subgoals,
            image_transform, settle_steps=20, flow_steps=10, observer=None):
    if horizon < 1 or not 1 <= stride <= policy.config['horizon']:
        raise ValueError('Need a positive horizon and 1 <= stride <= action chunk')
    if math.ceil(horizon / stride) > policy.config['max_steps']:
        raise ValueError('Rollout exceeds visual context; increase capacity or change protocol')
    obs = reset_environment(env, initial_state, settle_steps)
    policy.reset()
    final_predicate = False
    for step in range(horizon):
        # Increment exactly once, before the action, as in upstream evaluation.
        env.env._check_success(inc=True)
        if step % stride == 0:
            rgb = obs['agentview_image']
            if image_transform == 'flipud':
                rgb = rgb[::-1]
            elif image_transform == 'rotate180':
                rgb = rgb[::-1, ::-1]
            elif image_transform != 'none':
                raise ValueError('Unknown image transform')
            rgb = np.ascontiguousarray(rgb)
            if observer is not None:
                observer(step, rgb)
            chunk = policy.predict(encode(rgb), text, flow_steps)
        obs, _, done, _ = env.step(chunk[step % stride].tolist())
        # Upstream done may represent a single subgoal. Never stop on it.
        final_predicate = bool(done)
    completed = len(env.env._satisfied_subgoals)
    if not 0 <= completed <= subgoals:
        raise ValueError('Environment subgoal count differs from the evaluation protocol')
    overshot = bool(env.env._overshot)
    return {'success': completed == subgoals and not overshot and final_predicate,
            'subgoals_completed': completed, 'subgoals_total': subgoals,
            'overshot': overshot, 'steps': horizon}


def summarize(records):
    if not records:
        raise ValueError('No evaluation records')
    keys = [(r['task'], r['episode']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate task/episode results')
    tasks = []
    for task in sorted({r['task'] for r in records}):
        rows = [r for r in records if r['task'] == task]
        total = sum(r['subgoals_total'] for r in rows)
        tasks.append({'task': task, 'trials': len(rows),
                      'successes': sum(bool(r['success']) for r in rows),
                      'subgoal_completion': sum(r['subgoals_completed'] for r in rows)/total})
    successes = sum(bool(r['success']) for r in records)
    completed = sum(r['subgoals_completed'] for r in records)
    total = sum(r['subgoals_total'] for r in records)
    return {'trials':len(records), 'successes':successes,
            'whole_task_success':successes/len(records),
            'subgoals_completed':completed, 'subgoals_total':total,
            'pooled_subgoal_completion':completed/total,
            'mean_subgoal_completion':sum(t['subgoal_completion'] for t in tasks)/len(tasks),
            'overshot':sum(bool(r['overshot']) for r in records), 'tasks':tasks}
