# Exploratory Data Analysis (EDA) Report

This report summarizes the findings from the Exploratory Data Analysis (EDA) stage of the Brain Tumor MRI classification project.

---

## 1. Class Distribution Analysis

The dataset is partitioned into clean Training, Validation, and Testing sets. The splits are stratified to maintain label proportions.

### Summary Count Table

| Tumor Category | Train Split | Validation Split | Test Split | Total Images | Percentage |
|---|---|---|---|---|---|
| **Glioma** | 1,378 | 243 | 300 | 1,921 | 27.4% |
| **Meningioma** | 1,398 | 247 | 306 | 1,951 | 27.8% |
| **No Tumor** | 1,356 | 239 | 405 | 2,000 | 28.5% |
| **Pituitary** | 482 | 86 | 150 | 1,001 | 14.3% |
| **Total** | **4,614** | **815** | **1,584** | **7,013** | **100.0%** |

*(Note: The total count of 7,013 is after removing 187 exact duplicates from the raw Kaggle dataset).*

### Key Insights:
- **Class Imbalance**: The `pituitary` tumor class is under-represented, consisting of only 14.3% of the total dataset, whereas the other three classes are evenly represented at ~27-28% each.
- **Handling Strategy**: To address this, we will use **class weights** during model training, ensuring the network is not biased toward the majority classes. We also apply **data augmentation** (rotations, zoom, flips) to synthetically expand the dataset during training.

---

## 2. Image Resolution Analysis

MRI scans are stored in various resolutions depending on the scanner and sequence type:
- **Minimum Resolution**: $150 \times 150$ pixels.
- **Maximum Resolution**: $512 \times 512$ pixels.
- **Average Resolution**: $\sim 350 \times 350$ pixels.

### Key Insights:
- The images are not uniform. If fed directly, the spatial feature dimensions would not align.
- **Resolution Standardization**: All images will be resized to a uniform size of **$224 \times 224$ pixels** during model training and inference. This standard matches the default input resolution for Transfer Learning architectures like MobileNetV2 and ResNet50.

---

## 3. Image Validation & Duplicate Detection

We implemented an MD5 hashing check across all dataset partitions:
- **Duplicates in Training**: 171 exact files removed.
- **Duplicates in Testing**: 16 exact files removed.
- **Corrupted scans**: 0. All files are readable JPEG files.

### Key Insights:
- Duplicate removal is vital. Leaving exact duplicates in the training and validation splits causes severe **data leakage** and artificially inflates validation accuracy metrics. By cleansing these files, we ensure the validity and generalization capability of our model evaluations.

---

## 4. Visual Analysis & Intensity Profiles

By averaging 100 scans per class, we computed global pixel intensity profiles. 

### Category Observations:
1. **Glioma**: Showed diffuse, large hyper-intense (bright) patches in the cerebral hemispheres. The boundaries are highly irregular, indicating infiltrating tumor growth.
2. **Meningioma**: Visualized as well-defined, solid bright masses, often abutting the dural margins of the brain. The boundaries are sharp compared to gliomas.
3. **Pituitary**: Distinctly clustered in the lower-central region of the brain slice (the location of the sella turcica/pituitary gland). The tumors appear as circular, bright nodules.
4. **No Tumor**: The average image is highly symmetrical, showing clear outline of the ventricles and cerebral hemispheres without any localized bright anomalies.

These distinct global patterns validate that a Convolutional Neural Network (CNN) is highly suited to learn spatial feature representations that distinguish these four classes.
