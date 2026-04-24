# CFCT: ConvNeXt-FPN-CBAM Topology-Aware Skin Lesion Segmentation

This repository contains the open-source PyTorch implementation for a skin lesion segmentation pipeline based on a ConvNeXt encoder, an FPN-style decoder, CBAM attention, deep supervision, an edge branch, and a topology-aware loss component.

## 📄 Abstract

In this paper, we present a novel hybrid deep learning framework, **CFCT**, for skin lesion segmentation. Our method utilizes a **ConvNeXt-based backbone** for robust feature extraction. The framework incorporates a **Feature Pyramid Network (FPN)** decoder, which is further enhanced by **Convolutional Block Attention Modules (CBAM)** and an auxiliary **edge detection branch**.

We integrate **multi-scale deep supervision** and leverage a composite loss function that combines **Dice-BCE**, **edge-aware**, and **topological (Betti number difference)** penalties. This design improves both **boundary precision** and **topological correctness** of segmentation outputs.

We conducted extensive experiments on widely used benchmark datasets, including **HAM10000** and **ISIC 2018**. The results demonstrate that our approach **outperforms state-of-the-art methods**, highlighting the effectiveness and reliability of the proposed framework.

## Folder Structure

```text
cfct-open-source/
├── cfct/
│   ├── __init__.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── configs/
│   └── default.yaml
├── scripts/
│   └── train_slurm.sh
├── train.py
├── test.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Components

| File | Purpose |
|---|---|
| `cfct/model.py` | ConvNeXt backbone, FPN decoder, CBAM modules, deep supervision heads, and edge head |
| `cfct/dataset.py` | Training/validation segmentation dataset and test dataset loaders |
| `cfct/losses.py` | Dice+BCE loss, edge loss, edge-target generation, and topology loss |
| `cfct/metrics.py` | IoU, Dice, and pixel-accuracy evaluation |
| `cfct/utils.py` | Average meter, seed setup, and utility functions |
| `train.py` | Full training loop with deep supervision, edge loss, and topology loss |
| `test.py` | Prediction generation and test-set metric computation |
| `configs/default.yaml` | Dataset, model, training, and loss hyperparameters |
| `scripts/train_slurm.sh` | Example SLURM script for GPU training |

## Expected Dataset Layout

Set the dataset name and root in `configs/default.yaml`. By default, the code expects this structure:

```text
../data/HAM_p2/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

The default configuration uses:

```yaml
dataset: HAM_p2
root: data
input_size: 352
batch_size: 20
num_epochs: 50
learning_rate: 0.0001
backbone_name: convnext_base
model_name: ConvNeXt_v3
edge_weight: 0.2
topology_weight: 0.2
```

## Installation

```bash
git clone https://github.com/shakilur19/CFCT-Topology-Aware-Segmentation.git
cd CFCT-Topology-Aware-Segmentation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled training, install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

## Training

Edit `configs/default.yaml` first, then run:

```bash
python train.py
```

The best validation checkpoint is saved as:

```text
Snapshots/ConvNeXt_v3/<DATASET>/ConvNeXt_v3.pth
```

The final epoch checkpoint is saved as:

```text
Snapshots/ConvNeXt_v3/<DATASET>/ConvNeXt_v3.pth
```

## Testing and Evaluation

After training, run:

```bash
python test.py
```

Predicted masks are saved to:

```text
../data/<DATASET>/outputs/<DATASET>/ConvNeXt_v3/
```

The script prints:

```text
Average IoU
Average Pixel Accuracy
Average Dice
```

## Training Objective

The training loss keeps the original strategy:

```text
Total Loss = mean(Dice+BCE over out1, out2, out3, out4)
           + edge_weight × Edge Loss
           + topology_weight × Topology Loss
```

The four segmentation outputs provide deep supervision. The edge branch is trained using morphological edge targets generated from the ground-truth mask. The topology component compares Betti-1 persistence counts from binarized prediction and ground-truth masks using Gudhi cubical complexes.

## Important Notes

- The pipeline intentionally keeps the original model strategy: ConvNeXt encoder, FPN-style decoder, CBAM attention, deep supervision, edge branch, and topology-aware training.
- `test.py` preserves the original inference behavior by using `outputs[3]` as the prediction map.
- Large files such as datasets, checkpoints, outputs, and `.pth` files are ignored by `.gitignore`.
- If you release pretrained weights, upload them separately through GitHub Releases, Google Drive, Zenodo, or Hugging Face, then link them here.
