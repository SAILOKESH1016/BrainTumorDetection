import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from preprocessing import prepare_splits, get_image_generators

def fine_tune():
    print("==========================================")
    # 1. Load Dataset Splits and Generators
    print("Preparing splits and generators from 'archive'...")
    train_df, val_df, test_df, le = prepare_splits('archive')
    train_gen, val_gen, test_gen = get_image_generators(train_df, val_df, test_df, batch_size=32)
    
    # 2. Compute Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label_encoded']),
        y=train_df['label_encoded']
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")
    
    # 3. Load Champion Model
    model_path = 'models/model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Cannot fine-tune.")
        return
        
    print("Loading pre-trained champion model...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Unfreeze last 20 layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all except the last 20 layers of MobileNetV2 base
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    print("Model compilation...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # very low learning rate for fine-tuning
        loss='categorical_split' if hasattr(tf.keras.losses, 'categorical_split') else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Fit the model for 2 epochs on the real-world dataset
    print("Starting fine-tuning training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Save the fine-tuned model
    model.save('models/model.h5')
    print("Fine-tuned champion model saved successfully to: models/model.h5")
    
    # 6. Evaluate on Test Set
    print("Evaluating fine-tuned model on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Fine-tuned Test Accuracy: {test_acc:.4f} (previously 0.8371)")
    
if __name__ == '__main__':
    fine_tune()
