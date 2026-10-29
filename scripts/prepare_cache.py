"""Prepare a LIBERO-Mem HDF5 corpus for FPP training.

The script stores frozen DINOv3 patch features, simulator actions, a task-text
feature matrix, and a manifest. It never writes the raw demonstrations back to
the repository. The HDF5 layout follows LIBERO-Mem: ``data/<episode>/obs``
contains ``agentview_rgb`` and ``actions`` is a sibling dataset.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as functional


def _task_number(path: Path) -> int:
    match = re.search(r"SCENE1_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"cannot infer task number from {path.name}")
    return int(match.group(1))


def _pad_resize(images: torch.Tensor) -> torch.Tensor:
    height, width = images.shape[-2:]
    side = max(height, width)
    images = functional.pad(
        images,
        (
            (side - width) // 2,
            side - width - (side - width) // 2,
            (side - height) // 2,
            side - height - (side - height) // 2,
        ),
    )
    return (
        functional.interpolate(
            images.float(),
            (224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


def _features(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch: int,
    layout: str = "direct",
) -> np.ndarray:
    if layout not in {"direct", "legacy-paired32"}:
        raise ValueError("Unknown feature batching layout")
    if layout == "legacy-paired32" and batch != 64:
        raise ValueError("The legacy layout requires feature batch 64")
    paired = layout == "legacy-paired32"
    time_batch = 32 if paired else batch
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensor), time_batch):
            if paired:
                # Duplicate a single view to retain the original offline kernel
                # batch geometry; the policy still receives one view per time.
                repeated = np.repeat(images[start : start + time_batch], 2, axis=0)
                pixels = (
                    torch.from_numpy(np.ascontiguousarray(repeated))
                    .permute(0, 3, 1, 2)
                    .to(device)
                )
                pixels = _pad_resize(pixels)
            else:
                pixels = _pad_resize(tensor[start : start + time_batch]).to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model.forward_features((pixels.float() / 255 - mean) / std)
            output = output[:, model.num_prefix_tokens :].float().cpu().numpy()
            if paired:
                output = output[::2]
            parts.append(output)
    return np.concatenate(parts, axis=0).astype(np.float16)


def main(args: argparse.Namespace) -> None:
    """Convert raw HDF5 episodes into the prepared FPP cache."""
    import timm

    files = sorted(
        (
            Path(path).resolve()
            for pattern in args.raw_glob
            for path in glob.glob(pattern)
        ),
        key=_task_number,
    )
    feature_layout = getattr(args, "feature_layout", "direct")
    if feature_layout == "legacy-paired32" and args.batch != 64:
        raise ValueError("Use --batch 64 with --feature-layout legacy-paired32")
    if (
        args.batch < 1
        or args.history_stride < 1
        or args.max_observations < 2
        or args.min_scale <= 0
    ):
        raise ValueError("Batch, stride, context and min-scale must be positive")
    if len(files) != 10 or sorted(_task_number(f) for f in files) != list(range(1, 11)):
        raise ValueError(f"expected ten task HDF5 files, found {len(files)}")
    text = np.load(args.text_features)
    if text.shape != (10, 512) or not np.isfinite(text).all():
        raise ValueError(f"text features must have shape [10, 512], got {text.shape}")
    output = args.output.resolve()
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = timm.create_model(
        args.dino_model,
        pretrained=False,
        checkpoint_path=str(args.dino_checkpoint),
        dynamic_img_size=True,
    )
    model.eval().to(device)

    records = []
    train_actions = []
    reference = (
        json.loads(args.reference_split.read_text()) if args.reference_split else None
    )
    membership = (
        {(e["task"], e["episode"]): e["split"] for e in reference["episodes"]}
        if reference
        else None
    )
    if membership is not None and len(membership) != len(reference["episodes"]):
        raise ValueError("Duplicate reference split IDs")
    for path in files:
        task = _task_number(path)
        with h5py.File(path, "r") as handle:
            episodes = sorted(
                handle["data"].keys(), key=lambda name: int(name.rsplit("_", 1)[-1])
            )
            if len(episodes) <= 20:
                raise ValueError(
                    f"{path} must have training episodes in addition to 20 validation episodes"
                )
            for episode_index, episode_name in enumerate(episodes):
                split = (
                    membership.get((task, episode_name))
                    if membership is not None
                    else "val" if episode_index >= len(episodes) - 20 else "train"
                )
                if split is None:
                    continue  # Explicit test/unused IDs are never cached for training.
                if split not in {"train", "val"}:
                    raise ValueError("Reference episodes must be train or val")
                group = handle["data"][episode_name]
                actions = np.asarray(group["actions"], dtype=np.float32)
                if reference is not None:
                    actions = actions.copy()
                    actions[actions[:, -1] == 0, -1] = -1
                images = np.asarray(group["obs"]["agentview_rgb"], dtype=np.uint8)
                if (
                    actions.ndim != 2
                    or actions.shape[1] != 7
                    or len(actions) != len(images)
                ):
                    raise ValueError(
                        f"invalid action/image lengths in {path}:{episode_name}"
                    )
                if len(actions) == 0 or not np.isfinite(actions).all():
                    raise ValueError("Empty or non-finite actions")
                if (
                    len(actions) + args.history_stride - 1
                ) // args.history_stride > args.max_observations:
                    raise ValueError(
                        "Context too small for the sampled real observations"
                    )
                cache = prepared / f"task_{task:02d}" / episode_name
                cache.mkdir(parents=True, exist_ok=True)
                np.save(
                    cache / "features.npy",
                    _features(model, images, device, args.batch, feature_layout),
                )
                np.save(cache / "actions.npy", actions)
                if split == "train":
                    train_actions.append(actions)
                records.append(
                    {
                        "task": task,
                        "split": split,
                        "cache": str(cache.relative_to(prepared)),
                        "length": len(actions),
                        "source": f"{path.name}:{episode_name}",
                    }
                )

    if membership is not None:
        actual = {(e["task"], e["source"].rsplit(":", 1)[1]) for e in records}
        if actual != set(membership):
            raise ValueError(
                "RGB files do not contain every declared train/validation ID"
            )
    actions = np.concatenate(train_actions, axis=0)
    low, high = actions.min(axis=0), actions.max(axis=0)
    if getattr(args, 'normalization', 'symmetric') == 'symmetric':
        center = (low + high) / 2
        constant = (high - low) < 1e-8
        scale = np.where(constant, 1, (high - low) / 2)
    else:
        center = low
        raw_scale = high - low
        constant = raw_scale <= args.min_scale
        scale = np.maximum(raw_scale, args.min_scale)
    np.save(prepared / "text.npy", text.astype(np.float32))
    manifest = {
        "dino_model": args.dino_model,
        "dino_checkpoint_sha256": hashlib.sha256(
            args.dino_checkpoint.read_bytes()
        ).hexdigest(),
        "text_encoder": args.text_encoder,
        "feature_extraction": {
            "layout": feature_layout,
            "feature_batch": args.batch,
            "resize_device": (
                device.type if feature_layout == "legacy-paired32" else "cpu"
            ),
            "backbone_autocast": "bfloat16" if device.type == "cuda" else "float32",
        },
        "schema": "fpp_libero_mem_prepared_v1",
        "camera_keys": ["agentview_rgb"],
        "history_stride": args.history_stride,
        "max_observations": args.max_observations,
        "episodes": records,
        "normalization": {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "constant": constant.tolist(),
        },
        "text_features_sha256": hashlib.sha256(
            (prepared / "text.npy").read_bytes()
        ).hexdigest(),
    }
    if reference is not None:
        manifest["reference_split_sha256"] = hashlib.sha256(
            args.reference_split.read_bytes()
        ).hexdigest()
        manifest["zero_gripper_to_open"] = True
        manifest["evaluation_trials"] = [
            {
                "task": task["task"],
                "source": f'{task["file"]}:{episode["id"]}',
                "initial_state_sha256": episode["init_sha"],
                "horizon": episode["horizon"],
            }
            for task in reference["tasks"]
            for episode in task["raw_episodes"]
            if episode["id"] in task["test_ids"]
        ]
    (prepared / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    np.save(prepared / "text.npy", text.astype(np.float32))
    print(
        json.dumps(
            {"output": str(output), "episodes": len(records), "tasks": 10}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-glob", action="append", required=True)
    parser.add_argument(
        "--text-encoder",
        required=True,
        help="Encoder name, version, pooling and prompt convention",
    )
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--feature-layout",
        choices=["direct", "legacy-paired32"],
        default="direct",
        help="Legacy mode duplicates each agent image in 32-time blocks for cache reproduction; FPP still receives one view",
    )
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument("--normalization", choices=['symmetric','minrange'], default='symmetric',
                        help="Reference train-only symmetric min/max, or the earlier simplified [0,1] recipe")
    parser.add_argument("--max-observations", type=int, default=144)
    parser.add_argument("--min-scale", type=float, default=1e-4)
    parser.add_argument(
        "--reference-split",
        type=Path,
        help="Use recovered train/val/test IDs and map zero gripper commands to open",
    )
    main(parser.parse_args())
