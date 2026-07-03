# Dataset Specifications: Brain Tumor MRI Dataset

This document records the details, provenance, and structure of the dataset selected for the Brain Tumor Detection task.

## Dataset Provenance

- **Original Compiler**: Masoud Nickparvar
- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **License**: **CC0: Public Domain** (Freely available for commercial, academic, and private use without copyright restrictions).
- **Format**: High-contrast, single-channel or three-channel (RGB) MRI scan slices in `.jpg` format.

## Directory Structure

The dataset contains a pre-split structure representing training and testing partitions:

```
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Dataset Characteristics

- **Total Image Count**: 7,023 scans
- **Training Set Count**: 5,712 scans
- **Testing Set Count**: 1,311 scans
- **Target Categories**:
  - `glioma`: Glial cell tumors starting in the brain or spine (1,621 train + 300 test = 1,921 total)
  - `meningioma`: Tumors arising from meninges (membranes covering brain/spine) (1,645 train + 306 test = 1,951 total)
  - `notumor`: Control group MRI scans showing normal brain structures (1,595 train + 405 test = 2,000 total)
  - `pituitary`: Tumors growing in the pituitary gland (851 train + 150 test = 1,001 total)

## Expected Preprocessing Challenges

1. **Resolution Variance**: Scans are not all uniform in size (ranging from 150x150 to over 500x500 pixels). We must resize them to $224 \times 224$ pixels.
2. **Class Imbalance**: Pituitary scans are roughly half the volume of glioma, meningioma, and control scans. Data augmentation and potentially class weight adjustment will be applied during modeling.
3. **Contrast Variations**: Scans are collected from different machines. Min-max normalization (0 to 1 scaling) will ensure consistent input distribution.
