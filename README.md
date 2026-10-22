# Marine Debris Segmentation on Sentinel-2: ViT Encoder + MAE-style SSL + U-Net++

A PyTorch pipeline for pixel-level segmentation of marine debris and co-occurring sea-surface features in 11-band Sentinel-2 imagery. It combines a ViT-S/16 encoder, optional masked-autoencoder (MAE) self-supervised pre-training on Indian coastal-line imagery, a simple feature pyramid neck, and a U-Net++ decoder.

> **Scope of the reported results.** All numbers below come from supervised training and testing on the **MARIDA dataset only**. The Indian coastal-line SSL stage is part of the pipeline and is used for my separate marine debris detection project; it is not a source of labelled data here.
>
> **Disclosure.** The pipeline code was written with the help of an AI assistant. I reviewed it, ran it, and am responsible for the results reported here.

---

## Table of contents
1. [Pipeline overview](#pipeline-overview)
2. [Why this architecture](#why-this-architecture)
3. [Experiments](#experiments)
4. [Results](#results)
5. [Comparison with published work](#comparison-with-published-work)
6. [Usage](#usage)
7. [Repository layout](#repository-layout)
8. [Limitations](#limitations)
9. [References](#references)

---

## Pipeline overview

```
Sentinel-2 (11 bands)
        │
        ▼
┌───────────────────────────┐
│ ViT-S/16 encoder          │  patch 16, dim 384, depth 12, heads 6 (code defaults),
│ (sin-cos positional emb.) │  optional BandAdapter for external checkpoints
└─────────────┬─────────────┘
              │ token grid (H/16 × W/16)
              ▼
┌───────────────────────────┐
│ Simple Feature Pyramid    │  builds 4 scales from ONE feature map:
│ (deconv / 1×1 / strided)  │  strides 4, 8, 16, 32
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ U-Net++ decoder           │  nested dense skip connections, optional deep supervision
└─────────────┬─────────────┘
              ▼
   bilinear upsample → per-pixel class logits
```

Optional first stage (for `ssl_adapted` / `ssl_from_scratch`): MAE-style pre-training. 75% of patches are masked, the encoder sees only the visible patches, a light 4-layer decoder reconstructs the missing pixels, and the loss is computed on masked patches only, with per-patch pixel normalisation.

---

## Why this architecture

**ViT-S/16 encoder.** Vision Transformers model long-range context with global self-attention [3]. Floating debris is spectrally ambiguous (ships, foam, sargassum, wakes and debris can look alike), and the original MARIDA study stresses that spectral discrimination of debris from other sea-surface features is not straightforward [1]. Global context across a scene helps separate such look-alikes. The small variant (ViT-S) keeps the model trainable on the modest amount of labelled data MARIDA provides.

**MAE-style self-supervised pre-training.** Masked autoencoders learn by reconstructing heavily masked inputs, which is simple, scalable and effective for ViTs [2]. The method has already been extended to multispectral satellite data (SatMAE) [13]. The reasons for using it here:
- MARIDA is small and weakly labelled, while unlabelled Sentinel-2 imagery is plentiful.
- Pre-training on Indian coastal-line patches adapts the encoder to the spectral statistics, water types and coastal clutter of the region I care about for my downstream debris project.
- Masking 75% of tokens makes the encoder pass cheap (only visible tokens are processed).

**Simple feature pyramid neck.** A plain ViT produces a single-scale, stride-16 feature map, while segmentation decoders expect a multi-scale pyramid. ViTDet showed that a *simple* pyramid built from the last feature map (using deconvolutions and strided convolutions) is enough, without needing a hierarchical backbone [5]. This keeps the encoder a plain ViT, so MAE pre-training transfers directly.

**U-Net++ decoder.** U-Net++ replaces plain skip connections with nested, dense ones to reduce the semantic gap between encoder and decoder features [4], an extension of the original U-Net [6]. Debris objects are tiny (often a few pixels at 10 m resolution), so recovering fine detail from the high-resolution pyramid levels matters. The original MARIDA U-Net baseline already shows that convolutional encoder–decoders are a reasonable fit for this task [1].

**Loss and labels.** MARIDA is extremely imbalanced (marine debris is a very small fraction of annotated pixels), so training uses class-weighted losses (`losses/segmentation_losses.py`) and an `IGNORE_INDEX` for unlabelled pixels, since MARIDA labels are sparse and weakly supervised [1].

---

## Experiments

`main.py --experiment` selects the encoder initialisation:

| Experiment | Encoder init | SSL stage |
|---|---|---|
| `baseline` | random | none |
| `pretrained` | external ViT checkpoint (BandAdapter if band count differs) | none |
| `ssl_adapted` | external checkpoint, then MAE on Indian coastal imagery | yes |
| `ssl_from_scratch` | random, then MAE on Indian coastal imagery | yes |

<!-- TODO: state which experiment produced the numbers below, e.g. "Results are from the `ssl_adapted` run, seed 42." -->

---

## Results

MARIDA **test split**, from `logs/test_metrics.csv`:

| Metric | Value |
|---|---|
| Pixel accuracy | 0.9568 |
| Mean IoU (mIoU) | 0.6662 |
| Macro F1 | 0.7643 |
| Mean Dice | 0.7643 |
| Mean precision | 0.7673 |
| Mean recall | 0.7713 |

**Marine Debris class only**

| Metric | Value |
|---|---|
| IoU | 0.3441 |
| F1 | 0.5120 |
| Precision | 0.4740 |
| Recall | 0.5567 |

Notes:
- Mean Dice equals macro F1 by definition (for a binary mask, Dice and F1 are the same quantity), so the two columns are identical.
- Per-class IoU/F1/precision/recall are in `logs/per_class_metrics.csv`; the confusion matrix is in `logs/confusion_matrix.csv`.

---

## Comparison with published work

Reference numbers are taken from the cited papers as reported by them. **They are not strictly like-for-like**: papers differ in class grouping (11 vs 15 classes, whether Marine Debris is merged with suspected plastics), averaging (macro vs micro), and exact splits. Treat this as context, not a leaderboard.

### Marine Debris class (the hard, rare class)

| Model | Source | MD IoU | MD F1 |
|---|---|---|---|
| U-Net (MARIDA baseline) | Kikaki et al. 2022 [1], as tabulated in [11] | 0.33 | 0.49 |
| Random Forest (spectra + indices + GLCM texture) | Kikaki et al. 2022 [1], as tabulated in [11] | 0.67 | 0.83 |
| **This work (ViT-S + SimpleFPN + U-Net++)** | MARIDA test | **0.34** | **0.51** |

### Overall (all classes)

| Model | Source | mIoU | Macro F1 | Pixel acc. |
|---|---|---|---|---|
| U-Net (MARIDA baseline) | [11] (reporting [1]) | 0.66 | 0.74 | n/a |
| RF SS+SI+GLCM | cited in [10] (reporting [1]) | 0.69 | 0.79 | n/a |
| ResAttUNet | [8], cited in [10] | 0.67 | n/a | n/a |
| ResUNext (UNet + ResNeXt50) | Rao 2025 [12] (own protocol) | 0.75 | 0.84 | 0.85 |
| **This work** | MARIDA test | **0.67** | **0.76** | **0.96** |

### Reading the comparison honestly

- **Overall mIoU/F1 are competitive with the deep-learning baselines** (about 0.67 mIoU, in line with the MARIDA U-Net and ResAttUNet figures) but below the best Random Forest baseline and the ResUNext result.
- **Marine Debris IoU/F1 (0.34 / 0.51) are close to the original MARIDA U-Net (0.33 / 0.49) and well below the Random Forest baseline (0.67 / 0.83).** Published work also notes the RF baselines outperform the U-Net on this dataset [1][10]. So this model does not yet beat the strongest classical baseline on the target class.
- **Precision (0.47) is lower than recall (0.56)** for debris: the model over-predicts debris on look-alike classes, a failure mode also reported for the MARIDA baselines, which often misclassify water, foam, ships and clouds as debris [11].
- **Pixel accuracy (0.96) is not a useful headline number** here because water and clouds dominate the pixel count.
- A larger or newer benchmark is MADOS with MariNeXt [9], but it is a different dataset and is not comparable to these numbers.

---

## Usage

```bash
# 1. Sanity check (shapes, label ids, forward/backward pass)
python main.py --Config config/Config.yaml --mode sanity

# 2. SSL pre-training on Indian coastal-line patches
python main.py --Config config/Config.yaml --mode ssl --experiment ssl_adapted

# 3. Supervised segmentation on MARIDA
python main.py --Config config/Config.yaml --mode segmentation --experiment ssl_adapted

# 4. Test split evaluation + visualisations
python main.py --Config config/Config.yaml --mode test --experiment ssl_adapted

# Everything end-to-end
python main.py --Config config/Config.yaml --mode full --experiment ssl_adapted

# Resume an interrupted run
python main.py --Config config/Config.yaml --mode full --experiment ssl_adapted --resume
```

Outputs are written to `<runs_dir>/<experiment>/` (`checkpoints/`, `logs/`, `predictions/`, `visualizations/`, `summary.csv`, `run_metadata.json`).

---

## Repository layout

```
main.py                       entry point (sanity / ssl / segmentation / test / full)
models/encoder.py             ViT encoder, BandAdapter, simple feature pyramid, checkpoint loader
models/ssl_model.py           MAE-style masked autoencoder
models/unetpp.py              U-Net++ decoder
models/segmentation_model.py  encoder + neck + decoder
data/                         MARIDA and Indian coastal SSL datasets, transforms
losses/                       class-weighted segmentation losses
training/                     SSL and segmentation trainers, checkpointing
evaluation/                   metrics and confusion matrix
utils/                        seeding, logging, visualisation, system info
```

---

## Limitations

- Trained and evaluated on MARIDA only; generalisation to other regions (including the Indian coast) is not measured by these numbers.
- MARIDA labels are sparse and weakly supervised, so metrics reflect annotated pixels only.
- The marine debris class remains the weak point (IoU 0.34).
- Single reported run; no multi-seed variance estimate yet.
- Comparisons to the literature are indicative only (different class sets and protocols).
- Code was AI-assisted; independent review and reproduction are encouraged.

---

## References

1. Kikaki K., Kakogeorgiou I., Mikeli P., Raitsos D. E., Karantzalos K. **MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.** *PLOS ONE* 17(1): e0262247, 2022. https://doi.org/10.1371/journal.pone.0262247
2. He K., Chen X., Xie S., Li Y., Dollár P., Girshick R. **Masked Autoencoders Are Scalable Vision Learners.** CVPR 2022. https://arxiv.org/abs/2111.06377
3. Dosovitskiy A. et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR 2021. https://arxiv.org/abs/2010.11929
4. Zhou Z., Siddiquee M. M. R., Tajbakhsh N., Liang J. **UNet++: A Nested U-Net Architecture for Medical Image Segmentation.** DLMIA 2018. https://arxiv.org/abs/1807.10165
5. Li Y., Mao H., Girshick R., He K. **Exploring Plain Vision Transformer Backbones for Object Detection (ViTDet).** ECCV 2022. https://arxiv.org/abs/2203.16527
6. Ronneberger O., Fischer P., Brox T. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI 2015. https://arxiv.org/abs/1505.04597
7. Kikaki K., Kakogeorgiou I., Mikeli P., Raitsos D. E., Karantzalos K. **MARIDA quick-start repository.** https://github.com/marine-debris/marine-debris.github.io
8. **ResAttUNet: Detecting Marine Debris using an Attention activated Residual UNet.** arXiv:2210.08506. https://arxiv.org/abs/2210.08506
9. Kikaki K., Kakogeorgiou I., Hoteit I., Karantzalos K. **Detecting marine pollutants and sea surface features with deep learning in Sentinel-2 imagery (MADOS / MariNeXt).** *ISPRS J. Photogramm. Remote Sens.* 210: 39-54, 2024. https://doi.org/10.1016/j.isprsjprs.2024.02.017
10. **Binary reformulation for marine debris detection in Sentinel-2 imagery: an empirical study on extreme class imbalance using the first benchmarks on combined MARIDA and MADOS datasets.** *Frontiers in Marine Science*, 2026. https://doi.org/10.3389/fmars.2026.1765021
11. **High-precision Density Mapping of Marine Debris and Floating Plastics via Satellite Imagery.** arXiv:2210.05468. https://arxiv.org/abs/2210.05468
12. Rao (2025). **Hybrid Deep Learning Approach for Marine Debris Detection in Satellite Imagery Using UNet with ResNext50 Backbone.** https://www.researchgate.net/publication/392991161
13. Cong Y. et al. **SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.** NeurIPS 2022. https://arxiv.org/abs/2207.08051