# VCH-SegFormer

Code of **VCH-SegFormer**, a street-view semantic segmentation network that separates **trees** from **low vegetation**. It accompanies the manuscript

> X. Fei, W. Yan, Q. Zhu, X. Yan, and Z. Wu, "VCH-SegFormer for Street-View Vegetation Mapping and Assessing the Associations Between Urban Vegetation and PM2.5" (under review).

This repository contains the segmentation code only: the network, the preparation of the training data, training and evaluation, and the scripts that apply the trained networks to street-level photographs. It contains **no data, no trained weights, and no results**. The analyses of the association between vegetation and PM2.5 are not part of this repository.

## The network

VCH-SegFormer builds on SegFormer (Xie et al., NeurIPS 2021): a Mix Transformer (MiT-B2 or MiT-B5) encoder and an all-MLP decoder that fuses the four encoder stages at 1/4 of the input resolution into a 768-channel feature. Two modules are added (`code/seg/vch_segformer.py`); each can be switched off, which gives the ablation variants.

* **Vertical Context Gate (VCG).** The fused decoder feature is averaged over the image width, reduced to 1/8 of its channels by a 1-D convolution, and added to a learned table of 32 vertical positions that is linearly interpolated to the feature height. GELU activations, a dilated 1-D convolution, and a projection back to the full channel number give a gate *g*; the output is `x * 2 * sigmoid(g)`. The last projection is initialized to zero, so the gate starts as the identity. The gate re-weights each image row and channel and does not mix image columns.
* **Hierarchical Vegetation Head (HVH).** A coarse head predicts five super-classes (vegetation, sky, ground, built structure, other). A fine head (3×3 convolution from 768 to 192 channels, batch normalization, ReLU, 1×1 convolution to two classes) predicts tree versus low vegetation. The six class probabilities are composed as p(tree) = p(vegetation) · p(tree | vegetation) and p(low vegetation) = p(vegetation) · p(low vegetation | vegetation); the other four come from the coarse head. The loss is the negative log-likelihood of the composed six-class distribution plus 0.4 × the cross-entropy of the coarse head plus 0.4 × the cross-entropy of the fine head.

Output classes: `0` tree, `1` low vegetation, `2` sky, `3` ground, `4` built structure, `5` other; `255` is ignored.

| Variant name (`--variant`) | Network |
|---|---|
| `base` | SegFormer-B2 (flat six-class classifier) |
| `vcg` | SegFormer-B2 + VCG |
| `hvh` | SegFormer-B2 + HVH |
| `vch` | VCH-SegFormer-B2 (VCG + HVH) |
| `base_b5`, `vch_b5` | SegFormer-B5 and VCH-SegFormer-B5 (MiT-B5 encoder) |
| `unet`, `pspnet`, `deeplab` | comparison networks: U-Net, PSPNet, DeepLabV3+ with an ImageNet ResNet-50 encoder (segmentation-models-pytorch) |
| `upcnxt`, `upswint` | comparison networks: UperNet with a ConvNeXt-T or Swin-T encoder (Hugging Face Transformers) |

The encoders start from ImageNet-1k weights downloaded from the Hugging Face Hub (`nvidia/mit-b2`, `nvidia/mit-b5`, `facebook/convnext-tiny-224`, `microsoft/swin-tiny-patch4-window7-224`; `timm` ResNet-50 for the CNNs). On a machine without access to the Hub, set `NDVI_GVI_MODELS` to a folder that contains local copies named after the last part of the model id (for example `<NDVI_GVI_MODELS>/mit-b5`).

## Repository layout

```
code/
  config.py                 paths from environment variables (used by collect_results.py and prepare_kv_inputs.py)
  seg/
    vch_segformer.py        VCH-SegFormer: VCG, HVH, loss, build(variant)
    prepare_ade.py          ADE20K SceneParsing -> six-class labels, outdoor flag, train / select / test splits
    train_seg.py            training and evaluation of every variant under one protocol
    run_queue.sh            runs a list of variants and seeds on one GPU (resumes interrupted runs)
    eval_perimage.py        per-image confusion matrices on the test split (argmax and vegetation-first decoding)
    complexity.py           parameters, FLOPs, and inference time for a 512 x 512 image
    eval_cityscapes.py      zero-shot check on the 500 Cityscapes validation images (five merged classes)
    collect_results.py      result tables, seed summaries, and paired bootstrap intervals over test images
    prepare_kv_inputs.py    resizes the KartaView photographs used in the study to a short side of 512 px
    apply_kv.py             applies a trained network to photographs and writes the per-image class shares (green view index)
    predict_examples.py     label maps of example images for the figures
    export_weights.py       exports a checkpoint as .safetensors with a JSON description
    post_runs.sh            runs the evaluation and application scripts after training
```

Every script describes its inputs, outputs, and command line in its module docstring.

## Installation

The experiments of the paper used Python with PyTorch 2.4.1, Transformers 4.44.2, and segmentation-models-pytorch 0.3.4 for U-Net, PSPNet, DeepLabV3+, and the SegFormer-B2 variants, and PyTorch 2.8.0 (CUDA 12.8) with the same package versions for the UperNet and MiT-B5 networks, the Cityscapes check, and the inference on the Guangzhou street views. The training and evaluation scripts require a CUDA GPU.

```bash
git clone https://github.com/FEI-2006/NDVI-GVI-PM2.5-Study.git
cd NDVI-GVI-PM2.5-Study
pip install -r requirements-gpu.txt      # pins torch==2.4.1; install the PyTorch build that matches your CUDA version first if needed
```

## Data

The data are not redistributed here; obtain them from their providers under their own terms.

* **ADE20K SceneParsing** (`ADEChallengeData2016`, 150 classes): training data and test images.
* **Cityscapes** (fine annotations, validation split): zero-shot check only. `eval_cityscapes.py` expects `<root>/images/val/<name>.jpg` and `<root>/labels/val/<name>.png` with Cityscapes trainIds.

## Usage

**1. Prepare the six-class data set from ADE20K.**

```bash
python code/seg/prepare_ade.py --root /path/ADEChallengeData2016 --out /path/vegstruct
```

ADE20K classes are mapped to the six classes (tree: tree, palm; low vegetation: grass, plant, flower, field; sky; ground: road, sidewalk, earth, path, sand, runway, dirt track, land; built: wall, building, house, skyscraper, fence, bridge, tower, railing, column, awning, booth, hovel; other: every other class). An image is flagged as outdoor when sky covers at least 1% of its labeled pixels and it contains no ceiling. Splits: ADE20K training images with labeled pixels → `train`, except 500 randomly drawn outdoor training images (seed 20260929) → `select` (checkpoint choice); outdoor ADE20K validation images → `test`; the other validation images → `val_other`. The script writes `splits.csv`, `label_map.csv`, `summary.json`, and the label images.

**2. Train and evaluate.**

```bash
python code/seg/train_seg.py --variant vch_b5 --data /path/vegstruct --out runs/vch_b5 --gpu 0
```

Protocol, identical for every variant: 16,000 iterations, batch size 8, random scaling (0.5–2.0 of a 512-px short side), 512 × 512 crops, horizontal flips, and mild photometric jitter (gain 0.9–1.1, offset ±10); AdamW with a learning rate of 6 × 10⁻⁵ for the pretrained encoder and 6 × 10⁻⁴ for the new layers, weight decay 0.01, 1,000 warm-up iterations and linear decay; bf16 autocast. The `select` split is evaluated every 2,000 iterations, and the checkpoint with the highest vegetation IoU (tree and low vegetation merged) is kept as `best.pt`. The `test` split is evaluated once with that checkpoint (`test_metrics.json`) and with the weights of the last iteration (`test_metrics_final.json`). The default seed is 20260929; further runs use `--seed`.

To run several variants and seeds in a row (environment variables `WORK` for the working directory with `data/vegstruct`, `runs/`, and `logs/`, `REPO` for this clone, and optionally `PY` for the Python interpreter):

```bash
WORK=/path/work REPO=$PWD bash code/seg/run_queue.sh 0 base vcg hvh vch vch:20260930 base_b5 vch_b5
```

**3. Evaluate, measure, and summarize.**

```bash
python code/seg/eval_perimage.py --data /path/vegstruct --runs runs --gpu 0
python code/seg/complexity.py --out runs/complexity.csv --gpu 0
python code/seg/eval_cityscapes.py --root /path/cityscapes --runs runs --models base_b5,vch_b5 --out cityscapes_zeroshot.csv --gpu 0
```

`collect_results.py` expects the run folders under `results/seg/<variant>/` of this repository and writes its tables to `results/tables/`.

**4. Apply a trained network to street-level photographs.**

```bash
python code/seg/apply_kv.py --ckpt runs/vch_b5/best.pt --variant vch_b5 --img /path/photos --out vch_b5_shares.csv --gpu 0
```

For every `.jpg` image the script writes the share of pixels predicted as each class and `gvi_vs` = (tree + low vegetation) / all pixels. Images should have a short side of about 512 px, the inference size of the networks (`prepare_kv_inputs.py` does this for the KartaView photographs of the study; it reads the list of retained photographs produced by the study's data pipeline, which is not part of this repository).

## License

MIT License (see `LICENSE`). Third-party data sets and pretrained models keep their own licenses and terms; in particular, the SegFormer (MiT) encoder weights are distributed by NVIDIA under the NVIDIA Source Code License for SegFormer, and ADE20K and Cityscapes have their own terms of use.

## Citation

Please cite the manuscript above (the reference will be completed on publication) and this code repository (`CITATION.cff`).
