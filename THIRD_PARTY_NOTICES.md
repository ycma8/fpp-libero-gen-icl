# Attribution and external assets

- FPP attention and history primitives originate from [ycma8/fpp-libero-mem](https://github.com/ycma8/fpp-libero-mem), by Yuchi Ma. The shared library is retained under `fpp_libero_mem/`.
- Benchmark task definitions, official wrappers and sensorimotor conventions follow [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting), revision `96294c35705a18823ff2c850d37fbb1e12c21d85` (MIT). The external repository is required for simulation; its code/assets are not vendored here.
- LIBERO is pinned to revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Obtain its simulator dependencies and assets from the official distribution under their respective terms.
- LIBERO-Gen Combination data: `austinpatel/libero_gen_spatial_combination_hdf5`, revision `34503f3b51883acf175247b94dd55b2bdc2e1d96`. Original HDF5 data and simulator assets are not redistributed.
- DINOv3 ViT-B/16 LVD-1689M visual features are computed using timm. The external checkpoint SHA256 is `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`. Its source/model license and access conditions apply separately from this code's MIT license.
- The article layout is adapted from Thibaud Frere's [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template), revision `c75fc29c361755f66c46ae6e83d958b10c4d5feb`, under CC BY 4.0. Its retained notice is `article/LICENSE.template`.

The MIT license covers first-party implementation code. It does not relicense external datasets, pretrained weights, simulation packages, or the article template.
