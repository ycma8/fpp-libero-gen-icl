# Reproduce FPP on LIBERO-Gen Combination

## Scope and dependencies

The released recipe trains a fresh policy for **10K updates while retaining a 50K cosine schedule**. Do not replace that schedule with a 10K cosine. Generic DINOv3 features are pretrained/frozen. No robot-policy checkpoint initializes training. Exact bits are not promised across hardware/runtime versions.

Install Python dependencies from the root README, then separately obtain BPP at `96294c35705a18823ff2c850d37fbb1e12c21d85`, its LIBERO submodule at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, and the official simulator dependencies. Follow upstream setup for assets and compatible gym/robosuite/MuJoCo. The package install alone does not install a working simulator.

```bash
git clone --recursive https://github.com/real-stanford/behavior_prompting /repos/behavior_prompting
cd /repos/behavior_prompting
git checkout 96294c35705a18823ff2c850d37fbb1e12c21d85
git submodule update --init --recursive
```

From this FPP repository, set paths for your machine:

```bash
fpp_raw=/datasets/libero-gen
fpp_base=/datasets/fpp/base
fpp_demo=/datasets/fpp/demo
fpp_execution=/datasets/fpp/execution
fpp_dino=/models/dinov3/model.safetensors
fpp_bpp=/repos/behavior_prompting
fpp_libero=/repos/LIBERO
fpp_run=outputs/fpp-seed42
fpp_revision=$(git rev-parse HEAD)
export PYTHONPATH="$PWD:$fpp_bpp:$fpp_libero"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl
```

`fpp_libero` must point to the actual pinned LIBERO import root, not an unrelated installation. Check its revision explicitly. Configure NVIDIA EGL for your local driver; `configs/nvidia_egl.json` is an example and is not portable to every system.

## Data preparation

Obtain `austinpatel/libero_gen_spatial_combination_hdf5` at revision `34503f3b51883acf175247b94dd55b2bdc2e1d96`, preserving repository-relative paths under `fpp_raw`. `configs/data-sources.json` lists every expected HDF5 size/hash. The visual encoder is `vit_base_patch16_dinov3.lvd1689m`; the checkpoint SHA256 must be `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`.

The following example uses eight available GPUs. Do not start it on devices occupied by other jobs. Each stage must finish successfully before the next stage. The RGB prefix extraction in stage2 is required by the reference base-manifest schema; the policy's actual support is the sensorimotor sidecar from stage4.

```bash
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  python prepare_full_video.py --source-manifest configs/data-sources.json \
    --raw-root "$fpp_raw" --output "$fpp_base" --dino-checkpoint "$fpp_dino" \
    --official-env-root "$fpp_bpp/behavior_prompting/train_network/env/libero" \
    --source-revision "$fpp_revision" --max-steps 46 --shards 8 --shard "$shard" \
    --device "cuda:$shard" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  python prepare_demo_prefix.py --source-manifest configs/data-sources.json \
    --raw-root "$fpp_raw" --output "$fpp_base" --dino-checkpoint "$fpp_dino" \
    --shards 8 --shard "$shard" --device "cuda:$shard" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done

python prepare_full_video.py --source-manifest configs/data-sources.json \
  --raw-root "$fpp_raw" --output "$fpp_base" --dino-checkpoint "$fpp_dino" \
  --official-env-root "$fpp_bpp/behavior_prompting/train_network/env/libero" \
  --source-revision "$fpp_revision" --max-steps 46 --shards 8 --merge

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  python prepare_sensorimotor.py --data-root "$fpp_base" --raw-root "$fpp_raw" \
    --output "$fpp_demo" --dino-checkpoint "$fpp_dino" \
    --bpp-config "$fpp_bpp/behavior_prompting/train_network/config/libero_policy_dunetp.yaml" \
    --source-revision "$fpp_revision" --shards 8 --shard "$shard" --device "cuda:$shard" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
python prepare_sensorimotor.py --data-root "$fpp_base" --raw-root "$fpp_raw" \
  --output "$fpp_demo" --dino-checkpoint "$fpp_dino" \
  --bpp-config "$fpp_bpp/behavior_prompting/train_network/config/libero_policy_dunetp.yaml" \
  --source-revision "$fpp_revision" --shards 8 --merge

python prepare_execution.py --data-root "$fpp_base" --demo-cache-root "$fpp_demo" \
  --raw-root "$fpp_raw" --output "$fpp_execution" --source-revision "$fpp_revision" --stage-raw
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  python prepare_execution.py --data-root "$fpp_base" --demo-cache-root "$fpp_demo" \
    --output "$fpp_execution" --dino-checkpoint "$fpp_dino" \
    --source-revision "$fpp_revision" --shards 8 --shard "$shard" --device "cuda:$shard" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
python prepare_execution.py --data-root "$fpp_base" --demo-cache-root "$fpp_demo" \
  --output "$fpp_execution" --source-revision "$fpp_revision" --shards 8 --merge
```

Expected membership: 164 train tasks/8200 train trajectories, 10 held-out tasks/500 support trajectories, and 500 official rollout initial states. Query observations are sampled at raw rows `0,12,24,...`; next-action targets begin at `j+1`. Sensorimotor support chunks follow upstream same-row alignment, `j:j+20`. Image orientation is flipud; images are padded/resized to224×224 for frozen patch extraction. Normalization uses training data only. Test support demonstrations condition inference; their targets do not enter optimizer updates.

Cache manifest hashes include preparation/source/configuration metadata. Rebuilding under this release produces its own identities; do not force them to equal the supplied reference manifest or bypass identity checks to load an incompatible checkpoint. Keep the three caches together throughout any new training/evaluation. Original reference metadata is supplied for record verification, not as a replacement for generated cache files.

## Training

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root "$fpp_base" --demo-cache-root "$fpp_demo" \
  --execution-cache-root "$fpp_execution" --output "$fpp_run" \
  --source-revision "$fpp_revision"
```

The defaults are the reported method: all46 history slots, complete support patches, learned wrist pooling, direct demo attention, global64, seed42, steps10000, schedule50000, warmup2000, LR2e-4 and EMA0.999 from2000. Width512, six visual/eight flow layers, target cap32. The policy starts randomly; DINO stays pretrained/frozen. The reported model used eight A100-SXM4-80GB GPUs. I/O and simulator throughput are environment dependent.

## Official 500-trial evaluation

```bash
fpp_eval=outputs/eval-correct
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  MUJOCO_EGL_DEVICE_ID="$shard" python evaluate.py \
    --checkpoint "$fpp_run/step_010000.pt" --data-root "$fpp_base" \
    --demo-cache-root "$fpp_demo" --execution-cache-root "$fpp_execution" \
    --raw-root "$fpp_raw" --bpp-root "$fpp_bpp" --libero-root "$fpp_libero" \
    --dino-checkpoint "$fpp_dino" --shards 8 --shard "$shard" \
    --device "cuda:$shard" --source-revision "$fpp_revision" --output "$fpp_eval" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
python scripts/summarize_evaluation.py --evaluation "$fpp_eval" --output "$fpp_eval/performance.json"
python scripts/verify_official_video.py --evaluation "$fpp_eval" --performance "$fpp_eval/performance.json"
```

Use a fresh directory. EMA selection is explicit/default; no silent raw fallback. The protocol is 10 tasks×50 canonical pruned states, environment seed7, 12 settling controls,550 policy controls,20Hz, horizon16, execution chunk12, flow10, support `demo_{init_index%10}` and inference seed `4242+1000*sorted_task_index+init_index`.

For a matched wrong-prompt control, repeat with `--condition wrong` and another output directory. Preserve the same checkpoint and state/seed protocol. The offline verifier validates recorded identities/outcomes; it does not reproduce physical policy behavior or prove statistical superiority.
