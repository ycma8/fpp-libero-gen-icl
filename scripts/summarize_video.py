"""Aggregate completed rollout shards and compare prompts on matched trials."""
import argparse
import json
from pathlib import Path


def trial_key(row):
    return row['task'], row['initial_state_sha256'], row['seed']


def summarize(root, prefix='eval5000', official=False, condition_names=('correct', 'wrong', 'blank')):
    if len(set(condition_names)) != len(condition_names) or 'correct' not in condition_names or len(condition_names) < 2:
        raise ValueError('Require correct plus at least one distinct control')
    conditions = {}
    indexed = {}
    checkpoint_identity = None
    for condition in condition_names:
        directory = root / f'{prefix}-{condition}'
        protocols = sorted(directory.glob('protocol_shard*.json'))
        completed = sorted(directory.glob('summary_shard*.json'))
        if not protocols or len(completed) != len(protocols):
            raise ValueError(f'{condition}: evaluation shards are incomplete')
        for path in protocols:
            metadata = json.loads(path.read_text())
            identity = tuple(metadata.get(k) for k in ('checkpoint', 'checkpoint_step',
                             'training_source_revision', 'manifest_sha256'))
            if checkpoint_identity is None:
                checkpoint_identity = identity
            elif identity != checkpoint_identity:
                raise ValueError('Prompt controls use different checkpoint/data identities')
        rows = [json.loads(line)
                for path in sorted(directory.glob('trials_shard*.jsonl'))
                for line in path.read_text().splitlines() if line.strip()]
        index = {trial_key(row): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f'{condition}: duplicate task/state/seed')
        if any(row['condition'] != condition for row in rows):
            raise ValueError(f'{condition}: mislabeled trial')
        task_names = sorted({row['task'] for row in rows})
        expected = {json.loads(p.read_text())['trials_per_task'] for p in protocols}
        if len(expected) != 1:
            raise ValueError(f'{condition}: inconsistent trial counts')
        per_task = {}
        for task in task_names:
            subset = [row for row in rows if row['task'] == task]
            if len(subset) != next(iter(expected)):
                raise ValueError(f'{condition}: incomplete task {task}')
            per_task[task] = dict(trials=len(subset),
                                  successes=sum(row['success'] for row in subset),
                                  success_rate=sum(row['success'] for row in subset) / len(subset))
        if official and (len(rows) != 500 or len(task_names) != 10 or next(iter(expected)) != 50):
            raise ValueError(f'{condition}: official Combination requires10 tasks x50 trials')
        successes = sum(row['success'] for row in rows)
        conditions[condition] = dict(
            trials=len(rows), successes=successes, success_rate=successes / len(rows),
            task_mean_success_rate=sum(x['success_rate'] for x in per_task.values()) / len(per_task),
            per_task=per_task)
        indexed[condition] = index
    control_names = [c for c in condition_names if c != 'correct']
    keys = set(indexed[control_names[0]])
    if any(set(indexed[c]) != keys for c in control_names) or not keys.issubset(indexed['correct']):
        raise ValueError('Prompt controls do not share the same task/state/seed subset')
    comparisons = {}
    for condition in control_names:
        wins = sum(indexed['correct'][k]['success'] and not indexed[condition][k]['success'] for k in keys)
        losses = sum(not indexed['correct'][k]['success'] and indexed[condition][k]['success'] for k in keys)
        comparisons[condition] = dict(correct_only_successes=wins, control_only_successes=losses,
                                      both_successes=sum(indexed['correct'][k]['success'] and indexed[condition][k]['success'] for k in keys))
    return dict(
        conditions=conditions,
        matched_prompt_controls=dict(
            trials_per_condition=len(keys),
            successes={c: sum(indexed[c][k]['success'] for k in keys) for c in indexed},
            success_rates={c: sum(indexed[c][k]['success'] for k in keys) / len(keys) for c in indexed},
            paired_outcomes=comparisons,
            matching='task, initial-state SHA-256 and inference RNG seed'),
        interpretation_limits=[
            'Single training seed; checkpoint identity is recorded in each protocol.',
            'Controls reuse the same checkpoint and are inference ablations; blank zeros DINO features (OOD); last_frame and shuffled keep prefix length unchanged.',
            ('Official environment and multi-step wrappers; all 50 benchmark initial states/task, seed 7, 12 settling steps, 550 policy controls. BPP calibration is reported separately.' if official else
             'Local reset/simulator protocol: 10/10 recorded terminal goals valid, 9/10 checked expert open-loop replays successful.'),
            'LIBERO-Gen Combination tests new combinations of known primitives, not arbitrary new motor skills.',
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='eval5000')
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--conditions', nargs='+', default=['correct', 'wrong', 'blank'])
    args = parser.parse_args()
    result = summarize(args.root, args.prefix, args.official, args.conditions)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
