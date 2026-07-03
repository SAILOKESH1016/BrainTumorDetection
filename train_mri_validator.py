import os
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image
import random

# Direct, public Unsplash photo IDs for diverse everyday objects
UNSPLASH_IDS = [
    # Food/drink
    "photo-1546069901-ba9599a7e63c", "photo-1567620905732-2d1ec7ab7445", "photo-1565299624946-b28f40a0ae38", 
    "photo-1484723091739-30a097e8f929", "photo-1476224203421-9ac39bcb3327", "photo-1498837167922-ddd27525d352", 
    "photo-1482049016688-2d3e1b311543", "photo-1473093295043-cdd812d0e601", "photo-1504674900247-0877df9cc836", 
    "photo-1493770348161-369560ae357d",
    # People/portraits/selfies
    "photo-1507003211169-0a1dd7228f2d", "photo-1500648767791-00dcc994a43e", "photo-1438761681033-6461ffad8d80", 
    "photo-1534528741775-53994a69daeb", "photo-1544005313-94ddf0286df2", "photo-1506794778202-cad84cf45f1d", 
    "photo-1522075469751-3a6694fb2f61", "photo-1517841905240-472988babdf9", "photo-1501196354995-cbb51c65aaea", 
    "photo-1539571696357-5a69c17a67c6",
    # Animals (dogs, cats, birds)
    "photo-1543466835-00a7907e9de1", "photo-1514888286974-6c03e2ca1dba", "photo-1472491235688-bdc81a63246e", 
    "photo-1537151608828-ea2b117b62e4", "photo-1552053831-71594a27632d", "photo-1444569433885-41582c3dc166", 
    "photo-1583511655857-d19b40a7a54e", "photo-1504595403659-908b507b39ee", "photo-1518791841217-8f162f1e1131", 
    "photo-1533738363-b7f9aef128ce",
    # Nature/landscapes/trees/flowers
    "photo-1470071459604-3b5ec3a7fe05", "photo-1441974231531-c6227db76b6e", "photo-1447752875215-b2761acb3c5d", 
    "photo-1472214222555-d404758b1c42", "photo-1469474968028-56623f02e42e", "photo-1501854140801-50d01698950b", 
    "photo-1465146633011-14f8e0781093", "photo-1506744038136-46273834b3fb", "photo-1513836279014-a89f7a76ae86", 
    "photo-1500530855697-b586d89ba3ee",
    # Cars/vehicles
    "photo-1503376780353-7e6692767b70", "photo-1525609004556-c46c7d6cf0a3", "photo-1494976388531-d1058494cdd8", 
    "photo-1580273916550-e323be2ae537", "photo-1553440569-bcc63803a83d", "photo-1502877338535-766e1452684a", 
    "photo-1492144534655-ae79c964c9d7", "photo-1489824904134-891ab64532f1", "photo-1511919884226-fd3cad34687c", 
    "photo-1542282088-fe8426682b8f",
    # Tech/devices (laptops, phones)
    "photo-1496181130204-7552241d63e7", "photo-1585776245991-cf89dd7fc73a", "photo-1511707171634-5f897ff02aa9", 
    "photo-1588872657578-7efd1f1555ed", "photo-1505740420928-5e560c06d30e", "photo-1527443224154-c4a3942d3acf", 
    "photo-1542751371-adc38448a05e", "photo-1468495244123-6c6c332eeece", "photo-1563986768609-322da13575f3", 
    "photo-1587829741301-dc798b83add3",
    # Books/documents/paper
    "photo-1544716278-ca5e3f4abd8c", "photo-1497633762265-9d179a990aa6", "photo-1512820790803-83ca734da794", 
    "photo-1495446815901-a7297e633e8d", "photo-1481627834876-b7833e8f5570", "photo-1506880018603-83d5b814b5a6", 
    "photo-1532012197267-da84d127e765", "photo-1516979187457-637abb4f9353", "photo-1524995997946-a1c2e315a42f", 
    "photo-1513001900722-370f803f498d",
    # Everyday objects (bottles, cups, toys, buildings, items)
    "photo-1526170375885-4d8ecf77b99f", "photo-1513542789411-b6a5d4f31634", "photo-1523275335684-37898b6baf30", 
    "photo-1485955900006-10f4d324d411", "photo-1542291026-7eec264c27ff", "photo-1509048191080-d2984bad6ae5", 
    "photo-1481349518771-20055b2a7b24", "photo-1505330622279-bf7d7fc918f4", "photo-1486406146926-c627a92ad1ab", 
    "photo-1485827404703-89b55fcc595e"
]

def download_non_mri_images(output_dir="data/non_mri"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {len(UNSPLASH_IDS)} diverse everyday images from Unsplash...")
    
    count = 0
    for photo_id in UNSPLASH_IDS:
        save_path = os.path.join(output_dir, f"{photo_id}.jpg")
        if os.path.exists(save_path):
            count += 1
            continue
        
        url = f"https://images.unsplash.com/{photo_id}?w=224&h=224&fit=crop"
        try:
            urllib.request.urlretrieve(url, save_path)
            count += 1
            if count % 10 == 0:
                print(f"Downloaded {count}/{len(UNSPLASH_IDS)} images...")
        except Exception as e:
            print(f"Failed to download {photo_id}: {e}")
            
    print(f"Completed downloading non-MRI images. Total files: {count}")

def load_data(non_mri_dir="data/non_mri", mri_dir="archive/Training"):
    X = []
    y = []
    
    # 1. Load Non-MRI images (Class 1)
    non_mri_files = [os.path.join(non_mri_dir, f) for f in os.listdir(non_mri_dir) if f.endswith('.jpg')]
    for path in non_mri_files:
        try:
            img = Image.open(path).convert('RGB').resize((224, 224))
            X.append(np.array(img, dtype=np.float32))
            y.append(1)  # Class 1: Non-Brain MRI
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    num_non_mri = len(X)
    print(f"Loaded {num_non_mri} Non-MRI images.")
    
    # 2. Load MRI images (Class 0)
    mri_classes = ["glioma", "meningioma", "pituitary", "notumor"]
    mri_paths = []
    for c in mri_classes:
        c_dir = os.path.join(mri_dir, c)
        if os.path.exists(c_dir):
            files = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            mri_paths.extend(files)
            
    # Sample matching number of images
    sampled_mri_paths = random.sample(mri_paths, min(len(mri_paths), num_non_mri))
    for path in sampled_mri_paths:
        try:
            img = Image.open(path).convert('RGB').resize((224, 224))
            X.append(np.array(img, dtype=np.float32))
            y.append(0)  # Class 0: Brain MRI
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    print(f"Loaded {len(X) - num_non_mri} MRI images.")
    return np.array(X), np.array(y)

def train_model():
    download_non_mri_images()
    X, y = load_data()
    
    # Normalize inputs for MobileNetV2
    X_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(X)
    
    # Train/Val split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X_preprocessed[indices]
    y_shuffled = y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_val = X_shuffled[:split], X_shuffled[split:]
    y_train, y_val = y_shuffled[:split], y_shuffled[split:]
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Load pretrained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting binary classification model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16
    )
    
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/mri_validator.h5"
    model.save(model_save_path)
    print(f"Validation model saved successfully to: {model_save_path}")

if __name__ == "__main__":
    train_model()
