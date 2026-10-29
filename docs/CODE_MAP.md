# Code map

`train.py` fixes the published defaults and calls `train_video.py`. The latter contains the distributed optimizer loop and configuration builder. The complete model is `fpp_libero_gen/model.py`; direct demonstration attention is in `direct_head.py`, sensorimotor tokenization in `sensorimotor.py`, and native action/proprio conversion in `execution.py`.

`fpp_libero_gen/data.py` randomly pairs distinct same-task support/query episodes. Query observation histories are causal and sampled at stride 12. `fpp_libero_mem/distributed.py` weights rank losses by valid target counts. `ema.py` maintains constant FP32 EMA without changing raw parameter updates.

`evaluate.py` calls `evaluate_official_video.py` with EMA as the default. `official.py` uses the pinned upstream environment/wrappers; `simulation.py` contains XML reset utilities. The evaluator seeds each state independently, records all input fingerprints and refuses an existing trial output.

`fpp_libero_mem` is a shared numerical library, not an additional method or experiment. Internal configurable branches support compact tests and checkpoint compatibility. The release recipe is unambiguously `configs/reference.json` and the defaults in `train.py`.
