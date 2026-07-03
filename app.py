import streamlit as st
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:wght@700&display=swap');
    
    /* Core Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FF6B6B, #4D96FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #A5C9CA;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4D96FF;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4D96FF;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #E2D786;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info Box Styling */
    .info-box {
        background: rgba(77, 150, 255, 0.1);
        border-left: 5px solid #4D96FF;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1.5rem;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6C63FF, #3F3D56);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: background 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #3F3D56, #6C63FF);
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load resources
@st.cache_resource
def load_model_and_encoder():
    model_path = 'models/model.h5'
    encoder_path = 'models/label_encoder.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    validator_path = 'models/mri_validator.h5'
    
    model = None
    label_encoder = None
    preprocessor_config = None
    mri_validator = None
    imagenet_model = None
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor_config = pickle.load(f)
    if os.path.exists(validator_path):
        mri_validator = tf.keras.models.load_model(validator_path)
        
    # Always load pre-trained ImageNet model for object recognition
    imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
            
    return model, label_encoder, preprocessor_config, mri_validator, imagenet_model

model, label_encoder, preprocessor, mri_validator, imagenet_model = load_model_and_encoder()

def validate_mri_image(img):
    """
    Performs heuristic validation checks to verify if the uploaded image is a brain MRI scan.
    Returns: (is_valid, error_message)
    """
    try:
        # 1. Color Saturation Check
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32)
        
        # Standard deviation across R, G, B channels for each pixel
        channel_std = np.std(arr, axis=2)
        avg_channel_std = np.mean(channel_std)
        if avg_channel_std > 10.0:
            return False, f"Image contains too much color (average saturation: {avg_channel_std:.1f}). True brain MRI scans are grayscale."
            
        # 2. Border Darkness Check (Typical MRI scans are centered with a black background)
        img_gray = img.convert('L')
        arr_gray = np.array(img_gray, dtype=np.float32)
        h, w = arr_gray.shape
        
        # Border region is the outer 10%
        border_h = int(h * 0.1)
        border_w = int(w * 0.1)
        
        top_border = arr_gray[:border_h, :]
        bottom_border = arr_gray[-border_h:, :]
        left_border = arr_gray[:, :border_w]
        right_border = arr_gray[:, -border_w:]
        
        border_pixels = np.concatenate([
            top_border.flatten(),
            bottom_border.flatten(),
            left_border.flatten(),
            right_border.flatten()
        ])
        
        avg_border_intensity = np.mean(border_pixels)
        if avg_border_intensity > 45.0:
            return False, f"Image border is too bright (average: {avg_border_intensity:.1f}/255). True brain MRI scans must have a dark background border."
            
        # 3. Center Structure Check (Typical brain MRIs have the brain tissue in the center)
        center_h_start = int(h * 0.25)
        center_h_end = int(h * 0.75)
        center_w_start = int(w * 0.25)
        center_w_end = int(w * 0.75)
        
        center_pixels = arr_gray[center_h_start:center_h_end, center_w_start:center_w_end]
        avg_center_intensity = np.mean(center_pixels)
        if avg_center_intensity < 30.0:
            return False, f"Image center is too dark (average: {avg_center_intensity:.1f}/255). True brain MRI scans must contain bright anatomical tissue in the center."
            
        # 4. Variance Check (To filter out blank/uniform images or extreme high-contrast line art/text)
        global_std = np.std(arr_gray)
        if global_std < 15.0:
            return False, f"Image details are too uniform (contrast std: {global_std:.1f}). True brain MRI scans contain complex anatomical details."
            
        return True, "Image validated successfully as a candidate brain MRI scan."
    except Exception as e:
        return False, f"Failed to analyze image structures: {str(e)}"

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/plasticine/200/brain.png", use_container_width=True)
st.sidebar.title("NeuroScan AI")
st.sidebar.subheader("Navigation Menu")
page = st.sidebar.radio(
    "Select a Page:",
    ["🏠 Home", "📝 Project Overview", "📊 Dataset & EDA", "📈 Model Performance", "🧠 Diagnostic Prediction"]
)

# Header Section
st.markdown("<div class='main-title'>NeuroScan AI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-Style Brain Tumor MRI Classification Dashboard</div>", unsafe_allow_html=True)

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.header("Welcome to NeuroScan AI")
    st.write(
        "NeuroScan AI is a deep learning web system designed to analyze magnetic resonance imaging (MRI) scans "
        "of the human brain to detect and classify tumors into four categories: **Glioma, Meningioma, Pituitary, "
        "or No Tumor**."
    )
    
    # Showcase Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>83.71%</div>
            <div class='metric-label'>Champion Accuracy (MobileNetV2)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>7,013</div>
            <div class='metric-label'>MRI Scans Cleansed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>&lt; 100ms</div>
            <div class='metric-label'>Diagnostic Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.subheader("System Architecture")
    st.write(
        "The system pipeline is designed with modular software engineering practices. Below is the workflow diagram:"
    )
    st.code("""
[MRI Input Image] 
       │
       ▼
[Ingestion & Verification] ──► (Invalid format/corruption filter & MD5 duplicate check)
       │
       ▼
[Preprocessing Pipeline] ───► (224x224 Bilinear Resizing & [0,1] Min-Max Scaling)
       │
       ▼
[Model Inference Engine] ───► (MobileNetV2 Frozen Back-bone + Dense Head Classifier)
       │
       ▼
[Diagnostic Output] ────────► (Classification Prediction + Numerical Confidence Scores)
    """, language="text")

# ----------------- PROJECT OVERVIEW -----------------
elif page == "📝 Project Overview":
    st.header("Project Overview & Tumor Categories")
    st.write(
        "This project is a Computer Vision classification system that identifies tumors based on T1-weighted axial "
        "brain MRI slices. The prediction engine distinguishes four main classes:"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧬 1. Glioma")
        st.write(
            "Gliomas are tumors arising from glial cells (astrocytes, oligodendrocytes, ependymal cells) in the brain. "
            "They show diffuse, infiltrating growth patterns, making boundaries difficult to distinguish on MRI scans."
        )
        st.markdown("### 🧫 2. Meningioma")
        st.write(
            "Meningiomas arise from the meninges (membranes covering the brain and spinal cord). They are typically benign, "
            "well-circumscribed, and appear as bright, solid contrast-enhancing masses along meningeal boundaries."
        )
    with col2:
        st.markdown("### 🧠 3. Pituitary Tumor")
        st.write(
            "Pituitary tumors develop in the pituitary gland at the base of the brain. They are characterized by "
            "well-defined lesions in the sella turcica region, easily localizable on MRI slices."
        )
        st.markdown("### 🟢 4. No Tumor (Control Group)")
        st.write(
            "Control group scans representing normal brain anatomy. They exhibit high symmetry, clear ventricular "
            "pathways, and absence of localized contrast-enhancing masses."
        )

# ----------------- DATASET & EDA PAGE -----------------
elif page == "📊 Dataset & EDA":
    st.header("Dataset Statistics & Exploratory Data Analysis")
    
    st.write(
        "We performed Exploratory Data Analysis (EDA) on the clean partitions. The raw dataset from Kaggle "
        "contained 7,200 images, which was reduced to **7,013 images** after checking MD5 file hashes to remove "
        "187 exact duplicate scans."
    )
    
    tab1, tab2, tab3 = st.tabs(["📊 Class Distribution", "🖼️ Sample MRI Grid", "🔥 Intensity Heatmaps"])
    
    with tab1:
        st.subheader("Class Representation Across Splits")
        st.write(
            "The dataset exhibits a minor class imbalance, where the `pituitary` category consists of half the "
            "volume of the other categories. This was resolved mathematically using class weighting during training."
        )
        if os.path.exists("results/class_distribution.png"):
            st.image("results/class_distribution.png", caption="Class counts across Train, Validation, and Test sets", use_container_width=True)
            
    with tab2:
        st.subheader("MRI Scans Samples")
        st.write(
            "Below is a grid representing random samples of glioma, meningioma, pituitary, and control scans "
            "showing their original resolution boundaries before preprocessing."
        )
        if os.path.exists("results/sample_images.png"):
            st.image("results/sample_images.png", caption="Sample MRI scans per class", use_container_width=True)
            
    with tab3:
        st.subheader("Average Class Intensity Profiles")
        st.write(
            "By averaging 100 images per class, we can visualize the global localization cues: pituitary scans "
            "show a clear lower-central concentration, while meningiomas show boundary accents, and gliomas exhibit "
            "diffuse central patterns."
        )
        if os.path.exists("results/average_intensity_maps.png"):
            st.image("results/average_intensity_maps.png", caption="Mean pixel intensity maps", use_container_width=True)

# ----------------- MODEL PERFORMANCE -----------------
elif page == "📈 Model Performance":
    st.header("Model Training & Performance Comparison")
    st.write(
        "We trained and evaluated three neural network architectures on CPU. MobileNetV2 was selected "
        "as the champion model based on testing set results."
    )
    
    # Results metrics table
    metrics_data = {
        "Model Name": ["Custom CNN", "ResNet50 (Transfer)", "MobileNetV2 (Transfer)"],
        "Parameters": ["~250k (Optimized)", "~23.7M (Frozen Base)", "~2.3M (Frozen Base)"],
        "Test Accuracy": ["25.25%", "52.90%", "83.71%"],
        "F1-Score (Pituitary)": ["0.40", "0.41", "0.89"],
        "Recommendation": ["Underfitted", "Requires more epochs", "🏆 Champion (Selected)"]
    }
    st.table(pd.DataFrame(metrics_data))
    
    tab1, tab2, tab3 = st.tabs(["🏆 Accuracy Chart", "🧩 Confusion Matrices", "📈 Training Curves"])
    
    with tab1:
        if os.path.exists("results/model_accuracy_comparison.png"):
            st.image("results/model_accuracy_comparison.png", caption="Accuracy Comparison Bar Chart", use_container_width=True)
    with tab2:
        if os.path.exists("results/confusion_matrices_comparison.png"):
            st.image("results/confusion_matrices_comparison.png", caption="Side-by-Side Confusion Matrices Comparison", use_container_width=True)
    with tab3:
        st.subheader("Training History Curves")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/mobilenetv2_training_curves.png"):
                st.image("results/mobilenetv2_training_curves.png", caption="MobileNetV2 Accuracy & Loss Curves", use_container_width=True)
        with col2:
            if os.path.exists("results/resnet50_training_curves.png"):
                st.image("results/resnet50_training_curves.png", caption="ResNet50 Accuracy & Loss Curves", use_container_width=True)

# ----------------- PREDICTION PAGE -----------------
elif page == "🧠 Diagnostic Prediction":
    st.header("Automated Diagnostic Prediction Engine")
    st.write(
        "Upload a brain MRI scan slice in JPG, JPEG, or PNG format to trigger model prediction."
    )
    
    st.info(
        "ℹ️ **System Scope Disclaimer:** This diagnostic engine is specifically trained to analyze axial brain MRI scans "
        "to detect Glioma, Meningioma, Pituitary, or No Tumor. Any non-MRI images will be detected and rejected "
        "by our automated image structure validation algorithms."
    )
    
    if model is None or label_encoder is None:
        st.error(
            "Error: Pre-trained model weights (`models/model.h5`) or configurations are missing. "
            "Please ensure model execution has completed."
        )
    else:
        uploaded_file = st.file_uploader("Choose an MRI image slice...", type=["jpg", "jpeg", "png"])
        test_path = st.text_input("Or enter local image path for testing (optional):")
        
        img_source = None
        if uploaded_file is not None:
            img_source = uploaded_file
        elif test_path and os.path.exists(test_path):
            img_source = test_path
            
        if img_source is not None:
            try:
                # Load and check image validity first to prevent st.image crash on invalid file types
                img = Image.open(img_source)
                img_format = img.format if (hasattr(img, 'format') and img.format) else 'Unknown'
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_source, caption='Uploaded Image', width=350)
                    
                with col2:
                    st.success(f"Image loaded successfully! Format: {img_format}, Size: {img.size}")
                    
                    # Convert to RGB and resize for validation model and ImageNet model
                    img_eval_base = img.convert('RGB').resize((224, 224))
                    img_array_base = np.array(img_eval_base, dtype=np.float32)
                    
                    # 1. Run Binary MRI Validator Model
                    is_brain_mri = True
                    validation_msg = ""
                    
                    st.markdown("### 🔍 MRI Validation Stage")
                    
                    with st.spinner("Validating image structure..."):
                        if mri_validator is not None:
                            # Preprocess input for validation model (MobileNetV2 preprocess_input)
                            val_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_base.copy())
                            val_input = np.expand_dims(val_input, axis=0)
                            val_pred = mri_validator.predict(val_input)[0][0]
                            
                            # Class 0: Brain MRI, Class 1: Non-Brain MRI
                            if val_pred >= 0.5:
                                is_brain_mri = False
                                validation_msg = f"Structure validation model score: {val_pred:.4f} (expected < 0.5 for Brain MRI)"
                        
                        # Double-check using Heuristics
                        if is_brain_mri:
                            is_valid_heur, heur_msg = validate_mri_image(img)
                            if not is_valid_heur:
                                is_brain_mri = False
                                validation_msg = heur_msg
                                
                    if not is_brain_mri:
                        # Identify what the image is using ImageNet model
                        detected_object = "Unknown everyday object"
                        with st.spinner("Analyzing invalid object class..."):
                            if imagenet_model is not None:
                                imagenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_base.copy())
                                imagenet_input = np.expand_dims(imagenet_input, axis=0)
                                imagenet_preds = imagenet_model.predict(imagenet_input)
                                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(imagenet_preds, top=1)[0][0]
                                detected_object = decoded[1].replace('_', ' ').title()
                                
                        # Display Warning Message in professional medical style
                        st.markdown(f"""
                        <div style='background-color: rgba(231, 76, 60, 0.1); border-left: 6px solid #C0392B; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>
                            <h3 style='color: #C0392B; margin-top: 0; font-family: "Outfit", sans-serif; font-weight: 700;'>⚠️ Invalid Input</h3>
                            <p style='font-size: 1.15rem; color: #2C3E50; font-family: "Outfit", sans-serif; margin-bottom: 0.8rem;'>
                                <strong>Detected Image:</strong> <span style='color: #D35400; font-weight: bold;'>{detected_object}</span>
                            </p>
                            <p style='color: #2C3E50; font-family: "Outfit", sans-serif; margin-bottom: 0.5rem; font-size: 1.05rem;'>
                                This application is designed <strong>only for Brain MRI scan images</strong>.
                            </p>
                            <p style='color: #2C3E50; font-family: "Outfit", sans-serif; margin-bottom: 0.5rem; font-size: 1.05rem;'>
                                The uploaded image does not belong to a Brain MRI.
                            </p>
                            <p style='color: #C0392B; font-family: "Outfit", sans-serif; font-weight: bold; margin-bottom: 0.5rem; font-size: 1.05rem;'>
                                Please upload a valid Brain MRI image for tumor detection.
                            </p>
                            <p style='font-size: 0.85rem; color: #7F8C8D; font-family: "Outfit", sans-serif; margin-top: 1.2rem; margin-bottom: 0;'>
                                <em>Diagnostics Details: {validation_msg}</em>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("No prediction generated. Tumor classification was bypassed to prevent diagnostic errors.")
                    else:
                        st.success("✅ **Brain MRI scan structure validated successfully!** proceeding to tumor classification...")
                        
                        st.markdown("### 🧠 Tumor Prediction Stage")
                        with st.spinner("Detecting tumor class and computing confidence..."):
                            # Preprocess for classification model (min-max scaling / 255.0)
                            img_batch = np.expand_dims(img_array_base / 255.0, axis=0)
                            
                            # Inference
                            preds = model.predict(img_batch)[0]
                            pred_idx = np.argmax(preds)
                            
                            # Map labels
                            pred_label = label_encoder.inverse_transform([pred_idx])[0]
                            confidence = preds[pred_idx] * 100
                            
                            # Style output
                            color = "#2980B9" # Dark blue default
                            if "no" in pred_label.lower():
                                color = "#27AE60" # Dark green
                            elif "glioma" in pred_label.lower():
                                color = "#C0392B" # Dark red
                                
                            # Success Card in professional style
                            st.markdown(f"""
                            <div style='background-color: rgba(46, 204, 113, 0.1); border-left: 6px solid #27AE60; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>
                                <h3 style='color: #27AE60; margin-top: 0; font-family: "Outfit", sans-serif; font-weight: 700;'>✅ Brain MRI Detected</h3>
                                <p style='font-size: 1.25rem; color: #2C3E50; font-family: "Outfit", sans-serif; margin-bottom: 0.5rem;'>
                                    <strong>Prediction:</strong> <span style='color: {color}; font-weight: bold;'>{pred_label.upper()}</span>
                                </p>
                                <p style='font-size: 1.25rem; color: #2C3E50; font-family: "Outfit", sans-serif; margin-bottom: 0;'>
                                    <strong>Confidence:</strong> <span style='color: #8E44AD; font-weight: bold;'>{confidence:.2f}%</span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Plot prediction probabilities
                            prob_df = pd.DataFrame({
                                'Category': [label.capitalize() for label in label_encoder.classes_],
                                'Probability': preds
                            })
                            
                            fig, ax = plt.subplots(figsize=(6, 3))
                            sns.barplot(data=prob_df, x='Probability', y='Category', palette='viridis', ax=ax)
                            ax.set_xlim(0, 1.0)
                            ax.set_xlabel('Probability Score')
                            ax.set_ylabel('')
                            plt.title('Category Probabilities')
                            st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ **Error reading or processing the image file:** {e}")
                st.warning("Please ensure the uploaded file is a valid image format (JPEG, PNG, JPG).")
