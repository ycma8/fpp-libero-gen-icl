# Reference result

The fresh seed-42 FPP policy at 10,000 updates scores **391/500 (78.2%)** with its EMA weights. The matched wrong-task demonstration condition scores **51/500 (10.2%)**. Both conditions use the same checkpoint and official states/seeds. The released BPP seed-2 reference is391/500; the FPP point estimate matches it.

`correct/` and `wrong/` each contain eight protocol/summary/trial shards and a performance summary. Trial and summary files are unchanged exports. Protocol metadata replaces a machine-specific checkpoint path with `checkpoints/fpp-libero-gen-10k.pt`; original source hashes and that export transformation are declared in `provenance.json`. The original source revision is a reproducibility identifier, not a required branch checkout. Core implementation bytes are fingerprinted and checked by the verifier.

`canonical-manifest.json.gz` is the exact base-cache manifest: 8,200 training episodes, 500 support episodes, and 500 official initial-state identities. It contains metadata only, no image/action arrays. Policy checkpoint and EMA hashes are published, but weight binaries are not currently hosted. Offline record verification cannot replace loading those tensors or rerunning the simulation.

This is one training seed. Benchmark evaluations informed model selection. No significance, matched-compute superiority or optimal training budget is claimed. The wrong-prompt control supports dependence on the demonstration as a whole, not a claim about each modality or arbitrary unseen motor skills.

```bash
python3 scripts/verify_results.py
```
