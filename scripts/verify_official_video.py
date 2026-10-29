"""Recompute an archived 500-rollout Combination result using only Python's stdlib."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def verify(evaluation: Path, performance: Path) -> dict:
    score = read(performance)
    files = {kind: {p.name.removeprefix(f'{kind}_shard').split('.')[0]: p
                    for p in evaluation.glob(f'{kind}_shard*{suffix}')}
             for kind, suffix in [('protocol', '.json'), ('summary', '.json'), ('trials', '.jsonl')]}
    require(bool(files['protocol']) and set(files['protocol']) == set(files['summary']) == set(files['trials']),
            'Incomplete evaluation shards')
    protocol = read(next(iter(files['protocol'].values())))
    require(protocol == score['protocol'], 'Performance protocol differs from shard protocol')
    for key, expected in dict(environment_seed=7, control_hz=20, open_gripper_steps=12,
                              policy_control_budget=550, execution_chunk=12,
                              prediction_horizon=16, trials_per_task=50,
                              fixture_or_camera_overrides=False).items():
        require(protocol[key] == expected, f'Unexpected official setting: {key}')
    require(protocol['checkpoint_step'] == score['checkpoint_step'], 'Checkpoint step differs')
    require(protocol['condition'] == score['condition'], 'Condition differs')
    records = []
    fingerprints = {}
    for shard in sorted(files['protocol']):
        require(read(files['protocol'][shard]) == protocol, 'Mixed shard protocols/checkpoints')
        path = files['trials'][shard]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summary = read(files['summary'][shard])
        require(summary['trials'] == len(rows), 'Shard trial count differs')
        require(summary['successes'] == sum(r['success'] for r in rows), 'Shard successes differ')
        records.extend(rows)
        for kind in files:
            p = files[kind][shard]
            fingerprints[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    tasks = sorted({r['task'] for r in records})
    require(len(records) == 500 and len(tasks) == 10, 'Expected 10 tasks x 50 trials')
    keys = {(r['task'], r['init_index']) for r in records}
    require(keys == {(task, i) for task in tasks for i in range(50)}, 'Duplicate/missing official trials')
    state_keys = {(r['task'], r['initial_state_sha256'], r['seed']) for r in records}
    require(len(state_keys) == 500, 'Duplicate task/state/seed')
    per_task = {}
    for task in tasks:
        rows = [r for r in records if r['task'] == task]
        successes = sum(r['success'] for r in rows)
        per_task[task] = dict(trials=50, successes=successes, success_rate=successes / 50)
        for row in rows:
            require(type(row['success']) is bool, 'Success must be boolean')
            require(row['seed'] == 4242 + 1000 * tasks.index(task) + row['init_index'], 'Inference seed differs')
            require(row['environment_seed'] == 7 and row['condition'] == score['condition'], 'Trial protocol differs')
            prompt_task = tasks[(tasks.index(task) + 1) % 10] if score['condition'] == 'wrong' else task
            require(row['support_id'].endswith(f"/{prompt_task}:demo_{row['init_index'] % 10}"),
                    'Support demo assignment differs')
            require(0 <= row['policy_controls'] <= 550 and row['total_controls'] == row['policy_controls'] + 12,
                    'Control budget/settling differs')
            require('NVIDIA' in row['renderer'], 'Unexpected renderer')
        require(score['per_task'][task] == per_task[task], 'Per-task score differs')
    successes = sum(row['success'] for row in records)
    require(score['trials'] == 500 and score['successes'] == successes, 'Total score differs')
    require(math.isclose(score['success_rate'], successes / 500, abs_tol=1e-12), 'Success rate differs')
    require(set(score['per_task']) == set(tasks), 'Per-task membership differs')
    return dict(verified=True, checkpoint_step=score['checkpoint_step'], condition=score['condition'],
                trials=500, successes=successes, success_rate=successes / 500,
                checkpoint_sha256=score['checkpoint_sha256'],
                training_source_revision=protocol['training_source_revision'],
                evaluator_source_revision=protocol['evaluator_source_revision'],
                performance_sha256=hashlib.sha256(performance.read_bytes()).hexdigest(),
                artifact_sha256=fingerprints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--performance', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args.evaluation, args.performance), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')
