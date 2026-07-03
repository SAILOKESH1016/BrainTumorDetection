import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, ResNet50

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import prepare_splits, get_image_generators

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurations
EPOCHS_CUSTOM = 5
EPOCHS_MOBILENET = 4
EPOCHS_RESNET = 3
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4):
    """Builds a lightweight Custom CNN model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=4):
    """Builds a MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers for CPU speed
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_resnet50(input_shape=(224, 224, 3), num_classes=4):
    """Builds a ResNet50 transfer learning model."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers for CPU speed
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate_model(model_name, build_fn, train_gen, val_gen, test_gen, epochs, class_weights_dict):
    """Trains a model, saves it, and evaluates it on the test set."""
    print(f"\n==========================================")
    print(f" TRAINING MODEL: {model_name}")
    print(f"==========================================")
    
    model = build_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_split' if hasattr(tf.keras.losses, 'categorical_split') else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = f'models/best_{model_name.lower()}.h5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model as the default model
    final_path = f'models/{model_name.lower()}.h5'
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    
    # Evaluate on Test Set
    print(f"Evaluating {model_name} on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Metrics
    class_labels = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_labels)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save metrics and training history
    results = {
        'history': history.history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'report': report,
        'report_text': report_text,
        'confusion_matrix': cm.tolist()
    }
    
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{model_name.lower()}_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved metrics results to: {results_path}")
    
    # Plot training loss & accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = f'results/{model_name.lower()}_training_curves.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved training curve plot to: {plot_path}")
    
    return results

def main():
    # Load dataset split frames
    train_df, val_df, test_df, le = prepare_splits('archive')
    
    # Build Image Generators
    train_gen, val_gen, test_gen = get_image_generators(train_df, val_df, test_df, batch_size=BATCH_SIZE)
    
    # Compute Class Weights to handle Pituitary imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label_encoded']),
        y=train_df['label_encoded']
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\nCalculated Class Weights: {class_weights_dict}")
    
    # Store training metadata
    os.makedirs('models', exist_ok=True)
    
    # Train and evaluate Model 1: Custom CNN
    train_and_evaluate_model(
        'Custom_CNN',
        build_custom_cnn,
        train_gen,
        val_gen,
        test_gen,
        epochs=EPOCHS_CUSTOM,
        class_weights_dict=class_weights_dict
    )
    
    # Train and evaluate Model 2: MobileNetV2
    train_and_evaluate_model(
        'MobileNetV2',
        build_mobilenet_v2,
        train_gen,
        val_gen,
        test_gen,
        epochs=EPOCHS_MOBILENET,
        class_weights_dict=class_weights_dict
    )
    
    # Train and evaluate Model 3: ResNet50
    train_and_evaluate_model(
        'ResNet50',
        build_resnet50,
        train_gen,
        val_gen,
        test_gen,
        epochs=EPOCHS_RESNET,
        class_weights_dict=class_weights_dict
    )
    
    print("\n==========================================")
    print(" ALL MODELS TRAINED AND EVALUATED!")
    print("==========================================")

if __name__ == '__main__':
    main()
