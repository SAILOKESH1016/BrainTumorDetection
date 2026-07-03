# Model Optimization Strategies

This document details the ethical and architectural optimization techniques implemented to achieve maximum generalization and performance on the Brain Tumor MRI Classification task.

---

## 1. Class Imbalance Optimization (Class Weighting)

- **Problem**: The `pituitary` class accounts for only 14.3% of the dataset, which is half of the size of the other classes. This imbalances gradient updates, biasing predictions toward `glioma`, `meningioma`, or `notumor`.
- **Solution**: We computed balanced class weights from the training split:
  $$\text{weight}_c = \frac{N_{\text{total}}}{C \times N_c}$$
- **Impact**: Applied directly in Keras `model.fit()`, the loss penalty is scaled inversely to class size. This forces the model to pay equal attention to the minority pituitary class, preventing a drop in pituitary recall.

---

## 2. Regularization & Preventing Overfitting

Deep models are prone to overfitting on medical images due to high visual correlation and lack of diversity. We applied a multi-tiered regularization strategy:
- **Global Average Pooling**: We replaced `layers.Flatten()` in our Custom CNN with `layers.GlobalAveragePooling2D()`. This reduced the dense layer parameters from 12.8M to 16.3k, dramatically lowering model capacity to focus strictly on spatial features instead of pixel memorization.
- **Dropout**: A dropout rate of `0.4` to `0.5` is placed before the final classification head in all models. This forces the model to learn redundant feature representations.
- **L2 Weight Regularization**: Implemented in the Custom CNN dense layers (`kernel_regularizer=regularizers.l2(0.01)`) to penalize excessively large weights, smoothing the decision boundaries.
- **Batch Normalization**: Placed after every convolution block in the Custom CNN. This stabilizes the activation distribution, accelerating convergence and serving as a slight regularizer.

---

## 3. Training & Learning Rate Optimization

Stochastic Gradient Descent (SGD) can get trapped in local minima, and static learning rates often cause loss divergence.
- **Adaptive Optimizer**: We use the **Adam** optimizer, which maintains per-parameter adaptive learning rates based on first and second moments of gradients.
- **Learning Rate Decayer (`ReduceLROnPlateau`)**: If validation loss does not improve for 2 epochs, the learning rate is scaled down by a factor of 5 (from $10^{-3}$ to $2 \times 10^{-4}$). This allows the model to fine-tune weights as it approaches convergence.
- **Early Stopping & Checkpoints**: Validation loss is monitored. If it fails to improve for 3 consecutive epochs, training terminates. The model weights are automatically rolled back to the epoch with the lowest validation loss (`restore_best_weights=True`), preventing final-epoch overfitting.
