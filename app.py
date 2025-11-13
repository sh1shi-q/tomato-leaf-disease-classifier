import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        width: 100%;
        background-color: #FF6347;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🍅 Tomato Leaf Disease Classifier")
st.markdown("**Powered by EfficientNetB0 | 97.99% Accuracy**")
st.write("Upload an image of a tomato leaf to detect diseases instantly")

# Load model and classes
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('tomato_disease_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_classes():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading classes: {e}")
        return []

model = load_model()
class_names = load_classes()

# Disease information dictionary
disease_info = {
    'Bacterial_spot': '🦠 Small brown spots with yellow halos. Use copper-based sprays.',
    'Early_blight': '🍂 Dark spots with concentric rings. Remove infected leaves.',
    'Late_blight': '💧 Water-soaked lesions on leaves. Apply fungicides immediately.',
    'Leaf_Mold': '🟢 Yellow spots on upper leaf surface. Improve air circulation.',
    'Septoria_leaf_spot': '⚫ Small circular spots with dark borders. Remove lower leaves.',
    'Spider_mites Two-spotted_spider_mite': '🕷️ Tiny yellow spots, webbing present. Use miticides.',
    'Target_Spot': '🎯 Concentric rings forming target pattern. Apply fungicides.',
    'Tomato_Yellow_Leaf_Curl_Virus': '🟡 Yellowing and curling leaves. Control whiteflies.',
    'Tomato_mosaic_virus': '🦠 Mottled light and dark green on leaves. Remove infected plants.',
    'healthy': '✅ No disease detected! Keep up good plant care.',
    'powdery_mildew': '⚪ White powdery coating on leaves. Apply sulfur-based fungicides.'
}

# File uploader
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a tomato leaf for best results"
)

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        with st.spinner('🔍 Analyzing leaf...'):
            predictions = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
        
        # Display results with styling
        st.markdown("### 📊 Diagnosis Results")
        
        if confidence > 90:
            st.success(f"**{predicted_class}**")
        elif confidence > 70:
            st.warning(f"**{predicted_class}**")
        else:
            st.info(f"**{predicted_class}**")
        
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show disease information
        st.info(disease_info.get(predicted_class, "No information available"))
    
    # Show detailed predictions
    st.markdown("### 📈 Detailed Analysis")
    
    # Create dataframe for all predictions
    pred_data = []
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        pred_data.append({
            'Disease': class_name,
            'Probability': f"{prob*100:.2f}%",
            'Score': prob
        })
    
    # Sort by probability
    pred_data.sort(key=lambda x: x['Score'], reverse=True)
    
    # Display top 5 predictions
    for i, pred in enumerate(pred_data[:5]):
        with st.expander(f"#{i+1} - {pred['Disease']} ({pred['Probability']})"):
            st.write(disease_info.get(pred['Disease'], "No information available"))

# Sidebar information
st.sidebar.header("ℹ️ About This App")
st.sidebar.info("""
This AI-powered classifier uses **EfficientNetB0** deep learning model 
to identify 11 different tomato leaf conditions with **97.99% accuracy**.

**Model Performance:**
- Test Accuracy: 97.99%
- Test Precision: 98.01%
- Test Recall: 97.99%
- Test F1 Score: 97.99%
""")

st.sidebar.header("📋 Disease Classes")
st.sidebar.write("""
1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Yellow Leaf Curl Virus
9. Mosaic Virus
10. Healthy
11. Powdery Mildew
""")

st.sidebar.header("💡 Tips for Best Results")
st.sidebar.write("""
- Use clear, well-lit images
- Focus on the affected leaf area
- Avoid blurry or dark photos
- Include the entire leaf if possible
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with ❤️ using TensorFlow & Streamlit**")
