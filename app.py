import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained CNN model 
@st.cache_resource
model_url = r'https://github.com/Light-9630/Dermsage/blob/main/trained_model.h5'

def load_model():
    return tf.keras.models.load_model(model_url)

# Load the model at app startup
model = load_model()

# Title and Description
st.markdown("<h1 style='text-align: center; color:#0CAFFF;'>Dermsage - Skin Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #333; text-align: center;'>Welcome to Dermsage! Upload an image to check for skin diseases.</p>", unsafe_allow_html=True)

# Get user's name and age
user_name = st.text_input("Enter your name:")
user_age = st.number_input("Enter your age:", min_value=1)  # Set min_value to 1 to remove the placeholder

# Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_image is not None:
    
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Create a spinner to display during model prediction
    with st.spinner("Predicting..."):
      
        image = Image.open(uploaded_image)
        image_for_prediction = image.resize((224, 224))  
        image_for_prediction = np.asarray(image_for_prediction)
        image_for_prediction = image_for_prediction / 255.0  # Normalize the image data
        image_for_prediction = np.expand_dims(image_for_prediction, axis=0) 

        # Make predictions
        prediction = model.predict(image_for_prediction)
    
    # Display the prediction
    st.success("Prediction Complete!")
    class_names = [
        "Acne and Rosacea",
        "Eczema",
        "Normal",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Vitiligo"
    ]

    # Display the prediction
    st.markdown("<h2 style='text-align: center; color: #0CAFFF;'>Result:</h2>", unsafe_allow_html=True)
    predicted_class = class_names[np.argmax(prediction)]
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'><strong>Hi {user_name},</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'>The image you uploaded is classified as:</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'><strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

# About Dermsage
st.sidebar.markdown("<h2>About Dermsage:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Dermsage is an AI-powered skin disease detection service. We aim to provide quick and accurate skin disease diagnosis to our users.</p>", unsafe_allow_html=True)

# Benefits
st.sidebar.markdown("<h2>Benefits:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<ul><li>Fast and reliable skin disease detection.</li><li>Accessible from anywhere.</li><li>Support for multiple skin conditions.</li></ul>", unsafe_allow_html=True)

# Contact Us
st.sidebar.markdown("<h2>Contact Us:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>If you have any questions or feedback, please email us at info@dermsage.com.</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        background-color: #E6E6FA;
    }
    .stApp {
        background-color: transparent !important;
    }
    .sidebar .markdown-text-container {
        background-color: #FFF8DC;
    }
    .sidebar h2 {
        color: #FF69B4;
    }
    .sidebar p {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
