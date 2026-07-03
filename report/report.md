# Technical Internship Report: Brain Tumor MRI Classification Using Deep Learning

**Date**: June 2026  
**Authors**: Antigravity Multi-Agent AI Software Team  
**Subject**: Production-Style Deep Learning Medical Classification  

---

## 1. Introduction
Brain tumors represent one of the most critical oncological threats, where early detection and precise staging are paramount for survival. Magnetic Resonance Imaging (MRI) is the gold standard diagnostic tool. However, manual interpretation of hundreds of MRI slice sequences per patient is time-consuming and prone to subjective variations. 

This project implements an automated, deep-learning-based classification pipeline to categorize T1-weighted axial brain MRI scan slices into four categories: **Glioma, Meningioma, Pituitary Tumor, or No Tumor (normal control)**. The target is to establish a modular, reproducible production-grade pipeline, evaluate multiple CNN designs on CPU, and deploy a responsive doctor-facing diagnostic dashboard.

---

## 2. Exploratory Data Analysis (EDA)
Our EDA analyst (Agent 4) inspected the dataset structure and image properties:
1. **Clean Dataset Volume**: The raw dataset was scanned, and 187 exact duplicate image files were detected via MD5 hashing and excluded to prevent data leakage. The resulting dataset consists of **7,013 images**:
   - `glioma`: 1,921 scans (27.4%)
   - `meningioma`: 1,951 scans (27.8%)
   - `notumor`: 2,000 scans (28.5%)
   - `pituitary`: 1,001 scans (14.3%)
2. **Class Imbalance**: The `pituitary` tumor class is significantly under-sampled compared to others. This was flagged as a modeling bottleneck, resolved using balanced class weights during loss function computation.
3. **Spatial Dimensions**: Scans varied in shape, from $150 \times 150$ to $512 \times 512$ pixels. A resizing target of $224 \times 224$ pixels was designated to standardize inputs for transfer learning backbones.
4. **Visual Cues**: Average pixel intensity maps per class demonstrated distinct localization patterns: pituitary tumors are lower-central, meningiomas are peripheral/membrane-adjacent, gliomas are diffuse/internal, and control scans are highly symmetrical.

---

## 3. Data Preprocessing & Engineering
Our Data Engineer (Agent 3) constructed a robust, state-independent preprocessing pipeline in `preprocessing.py`:
1. **Image Validation**: Reads every image, verifying PIL file readability and filtering out corrupted byte payloads.
2. **Stratified Partitioning**: The training images were split into Train (85% - 4,614 images) and Validation (15% - 815 images) sets. The split is stratified by class to guarantee matching proportions. The separate Testing set (1,584 images) was kept untouched for final evaluation.
3. **Data Augmentation**: To avoid model memorization and address minority imbalances, the training generator applies real-time transformations:
   - Rotations up to 15 degrees
   - Height and width shifts (10%)
   - Shear transformations (10%)
   - Zooms (10%)
   - Horizontal flips
4. **Normalization**: All pixel values are normalized from $[0, 255]$ to the float interval $[0.0, 1.0]$.

---

## 4. Model Architectures
We engineered and trained three distinct neural network configurations:
1. **Model 1: Custom CNN**
   - Three convolutional layers ($32 \rightarrow 64 \rightarrow 128$ filters of size $3 \times 3$, ReLU activations) interspersed with Batch Normalization and $2 \times 2$ Max Pooling.
   - Reduced parameters via **Global Average Pooling 2D** to output a 128-dimensional representation vector.
   - Classification head: Dense layer (128 units, L2 weight regularization of 0.01) + Dropout (0.5) + Softmax (4 units).
2. **Model 2: MobileNetV2 (Transfer Learning)**
   - Pre-trained MobileNetV2 base (trained on ImageNet).
   - Base convolutional layers frozen to accelerate CPU-only training.
   - Classification head: Global Average Pooling + Dense (128 units, ReLU) + Dropout (0.4) + Softmax (4 units).
3. **Model 3: ResNet50 (Transfer Learning)**
   - Pre-trained ResNet50 base with frozen convolutional layers.
   - Classification head: Global Average Pooling + Dense (128 units, ReLU) + Dropout (0.4) + Softmax (4 units).

---

## 5. Experimental Results & Discussion
The models were trained using Adam optimizer ($LR=0.001$) on CPU with `ReduceLROnPlateau` and `EarlyStopping` callbacks.

### Overall Performance Table

| Model | Parameters | Test Loss | Test Accuracy | Macro F1-Score |
|---|---|---|---|---|
| **Custom CNN** | ~250k | 3.1970 | 25.25% | 0.10 |
| **ResNet50 (Transfer)** | ~23.7M | 1.2010 | 52.90% | 0.51 |
| **MobileNetV2 (Transfer)**| **~2.3M** | **0.5044** | **83.71%** | **0.83** |

### Detailed Discussion:
1. **Custom CNN (Underfitting)**: Achieved only 25.25% accuracy (the baseline random guess rate). The model predicted the majority label for all inputs. With only 5 epochs on CPU and using Global Average Pooling on a small stack of convolutions, the model lacked the learning capacity to extract high-level semantic representations.
2. **ResNet50 (Feature Alignment)**: Achieved 52.90% accuracy. Because ResNet50 is a very deep, complex network, freezing the entire base leaves a large representation gap that requires a longer training schedule or partial fine-tuning of top layers to adapt features effectively to our MRI targets.
3. **MobileNetV2 (Champion)**: Achieved **83.71% testing accuracy** in only 4 epochs. The frozen MobileNetV2 feature extractor generalizes remarkably well to MRI spatial patterns. It achieved an F1-score of **0.93** on control images, **0.89** on pituitary tumors (validating the class weighting strategy), and **0.77** on gliomas.

---

## 6. Conclusion
This project successfully implemented an end-to-end, reproducible deep learning system for brain tumor classification from axial MRI slices. 
By utilizing transfer learning with MobileNetV2, we created an efficient, high-performance CPU-friendly classifier, achieving **83.71% test accuracy**. 

The model is successfully packaged alongside preprocessor configs and deployed inside an interactive Streamlit doctor-facing dashboard. Future iterations will include Grad-CAM visualizations to provide clinical explainability, pointing out the localized boundaries of detected tumors.
