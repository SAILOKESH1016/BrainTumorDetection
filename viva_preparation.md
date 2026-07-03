# Viva & Technical Interview Preparation Guide

This document contains **20 key questions and answers** prepared by our QA Agent (Agent 12) to help succeed in the project presentation and viva examination.

---

### 🧠 Section 1: Deep Learning & CNN Foundations

#### Q1: Why do we use Convolutional Neural Networks (CNNs) for image classification instead of Standard Multi-Layer Perceptrons (MLPs)?
**A**: MLPs flatten images into 1D vectors, losing all spatial/hierarchical relationships between pixels. They also suffer from parameter explosion. CNNs use shared weights (convolutional kernels) to extract localized spatial features (edges, textures, shapes) while maintaining **translation invariance** and requiring far fewer parameters.

#### Q2: What is the purpose of Max Pooling layers in a CNN?
**A**: Max Pooling reduces the spatial dimensions (width and height) of feature maps. This:
1. Reduces computational complexity and memory usage.
2. Controls overfitting by limiting parameters.
3. Provides translation invariance, allowing the network to recognize features even if they are slightly shifted.

#### Q3: What is the difference between Flattening and Global Average Pooling (GAP)?
**A**: Flattening unrolls all spatial grids of a feature map into a long 1D vector, creating a massive number of connections to the next dense layer. Global Average Pooling computes the average value of each feature map channel, reducing the spatial grid ($H \times W$) to $1 \times 1$. GAP drastically reduces model parameters (e.g. from 12.8M to 16.3k in our Custom CNN), acts as a strong regularizer, and prevents overfitting.

#### Q4: Why did we use Batch Normalization in the Custom CNN?
**A**: Batch Normalization scales the activations of conv layers to have zero mean and unit variance for each training batch. This reduces **internal covariate shift**, stabilizes gradient flow, allows for higher learning rates, accelerates training convergence, and has a slight regularization effect.

---

### 🔄 Section 2: Transfer Learning

#### Q5: What is Transfer Learning, and why is it highly useful in this project?
**A**: Transfer Learning utilizes feature representations learned by a model pre-trained on a massive dataset (like ImageNet) and applies them to a new, smaller dataset (our MRI scans). It is highly useful because:
1. Training deep models from scratch requires tens of thousands of images. We only have ~5k training images.
2. It dramatically reduces training time and compute resources, enabling rapid training on CPU.

#### Q6: Why did we freeze the base of MobileNetV2 and ResNet50?
**A**: We froze the base layers (`base_model.trainable = False`) to prevent their pre-trained weights from being destroyed during backpropagation. This limits parameter updating solely to our new classification head (dense layers), reducing trainable parameters from millions to thousands and enabling fast training on CPU.

#### Q7: Why did MobileNetV2 perform significantly better than ResNet50 in our experiments?
**A**: MobileNetV2 is designed for mobile/edge systems, using lightweight **depthwise separable convolutions**. With frozen backbones and a limited training schedule (3-4 epochs on CPU), MobileNetV2's compact representation aligned quickly with the new dense head. ResNet50 has a much deeper architecture, and its complex feature space requires a longer training schedule or partial fine-tuning of top residual blocks to adapt properly.

---

### 🧹 Section 3: Data Engineering & Preprocessing

#### Q8: Why is checking for duplicate images using MD5 hashing important?
**A**: Leaving exact duplicate images in the dataset can cause them to be split into both the training and testing sets. This causes **data leakage**, where the model is evaluated on images it memorized during training, leading to artificially inflated test metrics and poor real-world generalization.

#### Q9: What is stratified splitting, and why did we use it?
**A**: Stratified splitting ensures that each subset (Train, Validation, Test) contains the exact same proportion of class labels as the original dataset. We used it to prevent split bias, ensuring that our validation metrics accurately represent minority classes (like pituitary).

#### Q10: Why did we normalize pixel values to the $[0, 1]$ range?
**A**: Raw pixels range from 0 to 255. Inputting large integers causes vanishing or exploding gradients during training. Scaling pixels to $[0.0, 1.0]$ stabilizes the input distribution, ensuring smooth weight updates and faster model convergence.

#### Q11: What is the benefit of using real-time Data Augmentation?
**A**: Data augmentation synthetically expands the training dataset by applying random modifications (flips, shifts, rotations, zooms) to the images. This prevents the model from memorizing specific coordinates, encouraging the extraction of robust geometric features and improving generalization.

#### Q12: Why did we NOT apply data augmentation to the validation and test datasets?
**A**: Validation and testing datasets must represent clean, unperturbed real-world clinical scans to ensure unbiased performance evaluation. Augmenting them would corrupt the validation metrics.

---

### ⚖️ Section 4: Regularization & Overfitting

#### Q13: How do you identify overfitting from training curves?
**A**: Overfitting is identified when training loss continuously decreases and training accuracy increases, while the validation loss begins to increase and validation accuracy plateaus or decreases.

#### Q14: What is the purpose of Dropout?
**A**: Dropout randomly deactivates a fraction of neurons (e.g., 40-50%) during each training forward pass. This prevents neurons from co-adapting (memorizing features together) and forces the network to learn robust, redundant pathways.

#### Q15: Explain how L2 Regularization works.
**A**: L2 regularization (Ridge regression penalty) adds a penalty term proportional to the square of the magnitude of the model weights to the loss function. This discourages weights from growing excessively large, smoothing the model's decision boundaries and reducing overfitting.

---

### 📉 Section 5: Optimization & Tuning

#### Q16: How does the `ReduceLROnPlateau` callback improve training?
**A**: It monitors validation loss. If the loss plateaus for a specific number of epochs, it scales down the learning rate. This allows the optimizer to take smaller steps near the loss valley, helping the model converge to a flatter, more stable local minimum.

#### Q17: What is the difference between early stopping and model checkpointing?
**A**: Early stopping terminates training when validation loss stops improving to prevent overfitting. Model checkpointing automatically saves the model weights at the end of each epoch only if validation performance improves, ensuring we do not lose our best model weights if training degrades.

---

### 📊 Section 6: Evaluation & Deployment

#### Q18: Why is Accuracy alone not a reliable metric for this project?
**A**: The dataset has a class imbalance (pituitary has half the images of other classes). An imbalanced model could predict majority classes and still achieve high accuracy while failing to identify pituitary tumors. We track **Precision, Recall, and F1-score** to evaluate performance per class.

#### Q19: What is a Confusion Matrix, and what clinical insight does it provide?
**A**: A confusion matrix plots true classes against predicted classes. In a clinical context, it highlights specific diagnostic risks, such as how often a malignant glioma is misclassified as meningioma or normal brain tissue (False Negatives), which would delay treatment.

#### Q20: How does caching models via `@st.cache_resource` optimize Streamlit apps?
**A**: Streamlit reruns the entire script from top to bottom on every user interaction. Without caching, the app would reload the massive 11MB model file from disk on every click, creating multi-second lag. Caching loads the model once into memory, enabling instantaneous diagnostic predictions (&lt;100ms).
