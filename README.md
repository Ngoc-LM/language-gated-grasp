# Language-Gated GR-ConvNet

**Steering a compact grasp CNN with frozen CLIP patch–text similarity**
Minh-Ngoc Luong. Short paper for the FPT Frontier Innovators 2026 (QACI) assignment on language-driven grasp detection, Grasp-Anything++.

![architecture](results/architecture.png)

Given an RGB image and a prompt such as *"grasp the mug by its handle"*, the model predicts one grasp rectangle `(x, y, w, h, θ)` on the part of the object that the prompt names. The base network is [GR-ConvNet](https://arxiv.org/abs/1909.04810), which predicts four dense maps (quality `Q`, `cos 2Θ`, `sin 2Θ`, opening `W`) and reads the grasp at the peak of the smoothed `Q` map (most interior pixel of the maximum plateau; see *Decoding* below). The height is fixed to `h = w/2`, the GR-ConvNet convention. The prompt enters the network in two places:

* **FiLM** after each of the 5 residual blocks: `F ← (1 + Δγ_k(t)) ⊙ F + β_k(t)`. The FiLM layers are zero-initialised, so training starts from the vision-only network.
* **Language gate (proposed)**: `S = cos(W_p P, W_t t)`, `A = Up(σ(τ S))`, `F' = F ⊙ (1 + A)`. Here `P` are the 14×14 frozen CLIP ViT-B/16 patch tokens and `t` is the frozen CLIP text embedding.

There is also a control variant, **C0**. It replaces `W_t t` with a learned, prompt-independent vector `q`. C0 therefore sees the same CLIP features but cannot read the prompt.

## Results (official LGD test split of Grasp-Anything++)

All runs train for 6 epochs on 35,927 scenes (183,559 prompt–grasp pairs), one Kaggle T4 per run (2.5–2.7 h).

Success is the top-1 grasp with IoU > 0.25 and angle error < 30° against any ground-truth grasp of the prompted part. `H` is the harmonic mean of seen and unseen success. Latency is end-to-end, batch 1, fp16 on a T4: it includes the CLIP text encoder (every frame) and the CLIP image encoder, but not JPEG decoding. B, C0 and C are the mean ± half-range of seeds 0 and 1; A and D use seed 0 only.

| run | model | seen | unseen | H | latency |
|---|---|---|---|---|---|
| A | GR-ConvNet, RGB only, no text | 27.3 | 27.8 | 27.5 | 4.7 ms |
| B | + FiLM(t) | 54.8 ± 2.7 | 40.7 ± 0.6 | 46.7 ± 1.4 | 14.1 ms |
| C0 (`E`) | B + text-free CLIP gate | 63.3 ± 0.5 | 46.0 ± 1.9 | 53.3 ± 1.1 | 19.6 ms |
| **C** | **B + language gate (proposed)** | **68.6 ± 0.4** | **57.7 ± 1.8** | **62.7 ± 1.3** | 19.7 ms |
| D | C + gate supervision | 70.8 | 57.7 | 63.6 | 21.5 ms |

With the prompt embedding cached (the prompt rarely changes between frames), C takes 13.0 ms per frame (6.4 ms network + 6.6 ms CLIP image encoder).

* **B → C0 (+6.6 H):** gain from frozen CLIP patch features behind a gate that cannot read the prompt. Its unseen part varies between seeds: +7.8 points in seed 0 (McNemar p < 1e-3) and +2.8 in seed 1 (p = 0.21).
* **C0 → C (+9.4 H):** gain from letting the gate read the prompt. Seed for seed, C exceeds C0 by 11.6 / 11.8 unseen points and by 6.3 / 4.4 seen points. Paired McNemar tests on the same test pairs give p < 1e-3 in all four comparisons (2 seeds × seen/unseen), and 95% bootstrap intervals that resample whole scenes exclude zero.
* **Share of the B → C gain due to prompt-dependent gating:** 69% unseen, 59% H, 38% seen.
* **D:** gate supervision does not differ significantly from C of the same seed (p = 0.06 seen, 0.28 unseen), so the simpler C is kept.
* **Gate maps.** In the 118 test scenes with more than one annotated object, C (seed 0) is correct for two prompts on different objects in 54 scenes and C0 in 40. The C0 gate is identical for every prompt (mean absolute change 0.0); the C gate changes by 0.09 on average. The paper's Fig. 2 shows one such scene.

The numbers are **not comparable** to the LGD paper (seen 48 / unseen 42; 18 unseen for GR-ConvNet + CLIP). Training here uses 3.6% of the images and testing covers only the annotated test pairs; the prompt-blind model A already exceeds the reported GR-ConvNet + CLIP unseen score, which indicates a different protocol.

### Decoding

The ground-truth `Q` target is a flat block, so its smoothed maximum is a plateau. A plain `argmax` returns the first (top-left) pixel of that plateau, which on synthetic rectangles lies a median 24 px from the centre. The decoder therefore picks the most interior pixel of the near-maximum set (`infer.select_peak`); for a unique maximum this equals `argmax`. All checkpoints were re-evaluated with [`src/reeval.py`](src/reeval.py) under both decoders ([`results/reeval_summary.csv`](results/reeval_summary.csv)): `decode='argmax'` reproduces the original numbers of all 8 runs exactly, and the plateau decoder changes each model's success by at most 0.2 points. It raises the ground-truth-map decoding ceiling from 97.5% to 98.0% (200 validation pairs). The table above uses the plateau decoder.

### Result files ([`results/`](results))

| file | content | produced by |
|---|---|---|
| `reeval_summary.csv` | test success of every checkpoint with both decoders | `src/reeval.py` |
| `mcnemar.csv` | paired McNemar tests and scene-clustered bootstrap intervals | `src/extras.py` |
| `coverage.json`, `coverage_success.csv` | training coverage of test object names, independent leakage search, success on single- vs multi-object scenes | `src/extras.py` |
| `fixed_val_curves.csv` | per-epoch success on the fixed 1,000-pair validation subset | `src/extras.py` (from `train_log.csv`) |
| `latency_e2e.csv` | latency per stage | `src/latency_e2e.py` |
| `summary.csv` | one row per run; `val` is full-validation success at the selected checkpoint, `kaggle_version` the Kaggle notebook version that trained the run, `wall_hours` is empty where the run log was not kept | collected by hand from the run outputs |

## Evaluation protocol

* **Test set.** The official LGD split files (`split/grasp-anything++/test/{seen,unseen}.obj`) are used: 573 seen and 230 unseen object keys. The files are parsed with a regex and never unpickled. This gives 1,376 seen and 601 unseen prompt–grasp pairs that have both a prompt and a label file.
* **Leakage control.**
  * All 620 test scenes are removed from training.
  * Every training pair whose object name appears among the unseen test objects is also removed. This is done after name normalisation.
  * Validation uses 2% of the training scenes (seed 0). Checkpoints are selected on a fixed 1,000-pair subset of it.
  * After filtering, training keeps 35,927 scenes and 183,559 pairs. 40 of the 41 seen test object names (1,373 of 1,376 seen pairs) occur in training, with a median of 3,723 training pairs per name.
  * The filter parses the object name from each prompt (`prepare.normalize_name`). As an independent check, `extras.py` also searches the raw training prompts for every unseen name (word boundaries, plural forms). It finds 3 names in 5,332 prompts, all as parts of other objects ("spoon at its bowl-shaped", "wine on its bottle", "candlestick at its candle holder"); none occurs in a parsed object name.
* **Metric.** [`src/eval_utils.py`](src/eval_utils.py) is this project's implementation of the LGD criterion (rotated-rectangle IoU with Shapely, angle difference taken mod π), written in notebook 01. Sanity checks (also in `tests/test_offline.py`): ground truth as prediction scores 1.0; +29° passes, +31° fails; +180° passes.
* **Upper bound of the map representation.** Decoding the ground-truth target maps of 200 validation pairs scores 98.0% (97.5% with plain argmax).

## Repository layout

```
notebooks/01_build_eval_data.ipynb   Kaggle CPU: official test split + training subset -> packed dataset
notebooks/02_train_evaluate.ipynb    Kaggle GPU T4 x2: writes the code, prepares caches, trains two runs in parallel,
                                     evaluates, measures latency (as-run snapshot; see the note in its first cell)
src/
  prepare.py      copy packed data, decode 224x224 image cache, CLIP text embeddings, CLIP patch tokens (C/D/E)
  data.py         dataset + GR-ConvNet-style target maps
  model.py        GraspNet: variants A (vision), B (+FiLM), C (+language gate), D (+gate loss), E (= C0, text-free gate)
  train.py        AdamW + cosine, fp16, best checkpoint on fixed validation subset, final val/test evaluation
  evaluate.py     top-1 decoding + success metric, latency, qualitative figure
  infer.py        Gaussian smoothing and top-1 decoding (plateau-aware; decode='argmax' reproduces the original runs)
  tests.py        shape / FiLM-identity checks and target round-trip (must reach >= 80%)
  launch.py       two-GPU launcher with hard watchdog; run specs like  A  E  C_s1 (variant C, seed 1)
  latency_e2e.py  batch-1 end-to-end latency incl. CLIP text / vision encoders
  extras.py       post-hoc analyses: McNemar tests, scene bootstrap, name coverage, leakage search, gate maps
  reeval.py       re-evaluates saved checkpoints on the test split (both decoders) and the GT-map ceiling
  collect_artifacts.py  copies the small per-run files of attached Kaggle outputs
  make_gate_figure.py  gate-map figure from extras.py output
  eval_utils.py   success metric (written in notebook 01, copied unchanged)
  packed_io.py    reader for the packed image/label files
results/          result tables, per-run outputs, analyses (see the table above)
tests/            CPU-only checks: python -m pytest tests -q
paper/            LaTeX source (CVPR 2026 author kit: main.tex, preamble.tex, sec/) and PDF of the 2-page paper
```

## Reproducing

1. **Build the data.** Run `notebooks/01_build_eval_data.ipynb` on Kaggle (CPU, Internet on).
   * It downloads the official test-split keys from the LGD repository.
   * It streams a ~3 GB image prefix plus the annotation archives from the public Hugging Face dataset `airvlab/Grasp-Anything-pp`.
   * It writes packed files (`images.bin`, `labels.npy`, parquet indexes) and the train/val/test manifests.
   * Save the output as a Kaggle dataset (here `ga-pp-packed`).
2. **Train and evaluate.** Run `notebooks/02_train_evaluate.ipynb` with **GPU T4 x2** and the packed dataset attached.
   * Choose the runs in cell 3, e.g. `VARIANTS=['A','B']`, `['C','D']`, `['E','C_s1']` or `['B_s1','E_s1']`.
   * Each pair of runs trains in parallel, one per GPU, in about 3 h.
   * Results are written to `/kaggle/working/results/`.
3. **Run the scripts directly** instead of the notebook.
   * Requirements: two CUDA GPUs (the CLIP patch-token cache for C/D/E is built on both), at least 25 GB free under `/tmp` (the patch tokens take about 14 GB), and Internet access for the CLIP weights. Defaults write runs to `/kaggle/working/runs`; pass `--base` and `--destination` elsewhere.
   * Training is not bit-reproducible: cuDNN autotuning is enabled for speed, so reruns differ slightly.

```bash
cd src
python prepare.py --root /path/to/ga_eval --variants C E
python tests.py --device cuda:0
python launch.py --variants C E --max-hours 3.5 --base runs
python summarize.py --variants C E --base runs --destination results
python latency_e2e.py --gpu 0 --out results
python reeval.py --data-root /path/to/ga_eval --runs-dir .. --out reeval     # checkpoints under ../results/<run>/best.pt
python extras.py --data-root /path/to/ga_eval --runs-dir reeval/plateau --out reeval/extras
```

4. **Offline checks** (CPU, no data): `python -m pytest tests -q`.

## Acknowledgements

* Grasp maps and decoding follow GR-ConvNet (Kumra et al., IROS 2020).
* Dataset, split and baselines come from Grasp-Anything++ / LGD (Vuong et al., CVPR 2024).
* Text and patch features come from OpenAI CLIP ViT-B/16 via Hugging Face `transformers`.
