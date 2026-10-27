# Language-Gated GR-ConvNet

**Steering a lightweight grasp detector with CLIP patch–text similarity**
Minh-Ngoc Luong. Short paper for the FPT Frontier Innovators 2026 (QACI) assignment on language-driven grasp detection, Grasp-Anything++.

![architecture](results/architecture.png)

Given an RGB image and a prompt such as *"grasp the mug by its handle"*, the model predicts one grasp rectangle `(x, y, w, h, θ)` on the part of the object that the prompt names. The base network is [GR-ConvNet](https://arxiv.org/abs/1909.04810), which predicts four dense maps (quality `Q`, `cos 2Θ`, `sin 2Θ`, opening `W`) and reads the grasp at `argmax Q`. The prompt enters the network in two places:

* **FiLM** after each of the 5 residual blocks: `F ← (1 + Δγ_k(t)) ⊙ F + β_k(t)`. The FiLM layers are zero-initialised, so training starts from the vision-only network.
* **Language gate (proposed)**: `S = cos(W_p P, W_t t)`, `A = Up(σ(τ S))`, `F' = F ⊙ (1 + A)`. Here `P` are the 14×14 frozen CLIP ViT-B/16 patch tokens and `t` is the frozen CLIP text embedding.

There is also a control variant, **C0**. It replaces `W_t t` with a learned, prompt-independent vector `q`. C0 therefore sees the same CLIP features but cannot read the prompt.

## Results (official LGD test split of Grasp-Anything++)

All runs train for 6 epochs on 35,964 scenes (183,559 prompt–grasp pairs). Each run uses one Kaggle T4 and takes about 2.7 h.

Success is the top-1 grasp with IoU > 0.25 and angle error < 30°. `H` is the harmonic mean of seen and unseen success. Latency is end-to-end, batch 1, fp16 on a T4, and includes the frozen CLIP encoders.

| run | model | seen | unseen | H | latency |
|---|---|---|---|---|---|
| A | GR-ConvNet, vision only | 27.3 | 27.8 | 27.5 | 4.3 ms |
| B | + FiLM(t) | 52.0 | 40.1 | 45.3 | 12.3 ms |
| C0 (`E`) | B + text-free CLIP gate | 62.8 | 47.9 | 54.4 | 18.2 ms |
| **C** | **B + language gate (ours)**, mean of seeds 0/1 | **68.6 ± 0.5** | **57.8 ± 1.9** | **62.8 ± 1.3** | 18.9 ms |
| D | C + gate supervision | 70.9 | 57.7 | 63.6 | 18.8 ms |

* **B → C0 (+9.1 H):** the gain from adding frozen CLIP visual features.
* **C0 → C (+8.4 H, +9.9 unseen):** the gain from letting the gate read the prompt. This gap is larger than both the seed spread and the 95% binomial intervals (about ±2.5 points seen, ±3.9 points unseen).
* **D:** gate supervision adds 2.2 seen points, which is within binomial noise, and leaves unseen and H unchanged. We therefore keep the simpler C.

The numbers are **not directly comparable** to the LGD paper (seen 0.48 / unseen 0.42 / H 0.45). We train on about 4% of the data, and we evaluate only on the test pairs whose annotations are released.

Per-run metrics are in [`results/summary.csv`](results/summary.csv). Validation curves are in [`results/fixed_val_curves.csv`](results/fixed_val_curves.csv) and latency in [`results/latency_e2e.csv`](results/latency_e2e.csv).

## Evaluation protocol

* **Test set.** We use the official LGD split files (`split/grasp-anything++/test/{seen,unseen}.obj`): 573 seen and 230 unseen object keys. The files are parsed with a regex and never unpickled. This gives 1,376 seen and 601 unseen prompt–grasp pairs that have both a prompt and a label file.
* **Leakage control.**
  * All 620 test scenes are removed from training.
  * Every training pair whose object name appears among the unseen test objects is also removed. This is done after name normalisation.
  * Validation uses 2% of the training scenes (seed 0). Checkpoints are selected on a fixed 1,000-pair subset of it.
* **Metric.** The metric code is [`src/eval_utils.py`](src/eval_utils.py) (rotated-rectangle IoU with Shapely, angle difference taken mod π). It passed sanity checks: ground truth used as the prediction scores 1.0; a +31° rotation fails; a +180° rotation succeeds.
* **Upper bound of the map representation.** Decoding the ground-truth target maps scores 97.5% on validation.

## Repository layout

```
notebooks/01_build_eval_data.ipynb   Kaggle CPU: official test split + training subset -> packed dataset
notebooks/02_train_evaluate.ipynb    Kaggle GPU T4 x2: writes src/, prepares caches, trains two runs in parallel,
                                     evaluates, measures latency (self-contained copy of src/)
src/
  prepare.py      copy packed data, decode 224x224 image cache, CLIP text embeddings, CLIP patch tokens (C/D/E)
  data.py         dataset + GR-ConvNet-style target maps
  model.py        GraspNet: variants A (vision), B (+FiLM), C (+language gate), D (+gate loss), E (= C0, text-free gate)
  train.py        AdamW + cosine, fp16, best checkpoint on fixed validation subset, final val/test evaluation
  evaluate.py     top-1 decoding + success metric, latency, qualitative figure
  infer.py        Gaussian smoothing and argmax decoding
  tests.py        shape / FiLM-identity checks and target round-trip (must reach >= 80%)
  launch.py       two-GPU launcher with hard watchdog; run specs like  A  E  C_s1 (variant C, seed 1)
  latency_e2e.py  batch-1 end-to-end latency incl. CLIP text / vision encoders
  eval_utils.py   success metric (shared verbatim with the data notebook)
  packed_io.py    reader for the packed image/label files
results/          summary tables, validation curves, latency, figures
paper/            LaTeX source (CVPR template) and PDF of the 2-page paper
```

## Reproducing

1. **Build the data.** Run `notebooks/01_build_eval_data.ipynb` on Kaggle (CPU, Internet on).
   * It downloads the official test-split keys from the LGD repository.
   * It streams a ~3 GB image prefix plus the annotation archives from the public Hugging Face dataset `airvlab/Grasp-Anything-pp`.
   * It writes packed files (`images.bin`, `labels.npy`, parquet indexes) and the train/val/test manifests.
   * Save the output as a Kaggle dataset; ours is called `ga-pp-packed`.
2. **Train and evaluate.** Run `notebooks/02_train_evaluate.ipynb` with **GPU T4 x2** and the packed dataset attached.
   * Choose the runs in cell 3, e.g. `VARIANTS=['A','B']`, `['C','D']` or `['E','C_s1']`.
   * Each pair of runs trains in parallel, one per GPU, in about 3 h.
   * Results are written to `/kaggle/working/results/`.
3. **Run the scripts directly** on any machine with two CUDA GPUs (instead of the notebook):

```bash
cd src
python prepare.py --root /path/to/ga_eval --variants C E
python tests.py --device cuda:0
python launch.py --variants C E --max-hours 3.5
python summarize.py --variants C E
python latency_e2e.py --gpu 0
```

## Acknowledgements

* Grasp maps and decoding follow GR-ConvNet (Kumra et al., IROS 2020).
* Dataset, split and baselines come from Grasp-Anything++ / LGD (Vuong et al., CVPR 2024).
* Text and patch features come from OpenAI CLIP ViT-B/16 via Hugging Face `transformers`.
