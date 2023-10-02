import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the CNN model
model = tf.keras.models.load_model("trained_model.h5")

# Hiding GitHub logo
hidden = """ <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hidden, unsafe_allow_html=True)

# Dermsage logo and Title

# Title and Description
st.markdown("<h1 style='text-align: center; color:#0CAFFF;'>Dermsage</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color:#0CAFFF;'>Transforming Skin Diagnosis</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #333; text-align: center;'>Welcome to Dermsage! Upload an image to check for skin diseases.</p>", unsafe_allow_html=True)

# Get user's name
user_name = st.text_input("Enter your name:")

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
    st.markdown("<h3 style='text-align: center; color: #0CAFFF;'>Result</h3>", unsafe_allow_html=True)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'><strong>Hi {user_name},</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'>The image you uploaded is classified as:</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: black; text-align: center;'><strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

    # Brief explanation about the predicted class
    class_descriptions = [
        " Acne is a common skin condition that causes pimples and other blemishes. Rosacea is a chronic skin condition that causes redness and visible blood vessels on the face.",
        " Eczema is a condition that causes the skin to become red, itchy, and inflamed. It often appears as dry, scaly patches on the skin.",
        " This image appears to be of normal skin with no signs of any specific skin condition.",
        " Psoriasis is a chronic skin condition that causes cells to build up rapidly on the surface of the skin. Lichen Planus is an inflammatory skin condition. Both can cause rashes and skin lesions.",
        " These are fungal skin infections that can cause itching, redness, and rashes on the skin.",
        " Vitiligo is a long-term skin condition characterized by patches of the skin losing their pigment. This results in the appearance of white patches on the skin."
    ]

    # Show description of the predicted class
    st.markdown(f"<p style='font-size: 18px; color: #333; text-align: left;'><b>Description:</b>{class_descriptions[predicted_class_index]}</p>", unsafe_allow_html=True)

# Note
st.markdown("<div style='border: 2px solid red; padding: 7px; margin-top: 20px;'><p><strong>Note:</strong> The AI model used in this application is under development. Please use the results as general information and consult a healthcare professional for accurate diagnosis and treatment.</p></div>", unsafe_allow_html=True)

# About Dermsage
st.sidebar.markdown("<h2>About Dermsage:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Dermsage is an AI-powered skin disease detection service. We aim to provide quick and accurate skin disease diagnosis to our users.</p>", unsafe_allow_html=True)

# Benefits
st.sidebar.markdown("<h2>Benefits:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<ul><li>Fast and reliable skin disease detection.</li><li>Accessible from anywhere.</li><li>Support for multiple skin conditions.</li></ul>", unsafe_allow_html=True)

# Contact Us
st.sidebar.markdown("<h2>Contact Us:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>If you have any questions or feedback, please email us at dermsage@gmail.com.</p>", unsafe_allow_html=True)

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
