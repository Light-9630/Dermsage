import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the CNN model
model = tf.keras.models.load_model("trained_model.h5")

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

    # Display the prediction and description
    predicted_class = class_names[np.argmax(prediction)]

    class_descriptions = {
        "Acne and Rosacea": "This category includes skin conditions characterized by pimples, redness, and inflammation, such as acne and rosacea.",
        "Eczema": "Eczema encompasses various forms of dermatitis that cause itchy and inflamed skin.",
        "Normal": "The image you uploaded shows normal, healthy skin without any specific skin conditions.",
        "Psoriasis pictures Lichen Planus and related diseases": "This category covers skin conditions like psoriasis, lichen planus, and related diseases known for their distinctive rashes and scales.",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "This category includes fungal skin infections like ringworm and candidiasis, known for their itchy and contagious nature.",
        "Vitiligo": "Vitiligo is a condition where the skin loses its pigmentation, resulting in white patches, which can affect any part of the body."
    }

    st.markdown("<h2 style='text-align: center; color: #0CAFFF;'>Result:</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'><strong>Hi {user_name},</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'>The image you uploaded is classified as:</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #0CAFFF; text-align: center;'><strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

    # Display description for the predicted class
    if predicted_class in class_descriptions:
        st.markdown(f"<p style='font-size: 16px; color: #333; text-align: center;'>{class_descriptions[predicted_class]}</p>", unsafe_allow_html=True)

# Note about the model
st.sidebar.markdown("<h2>Note:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>The AI model used in this application is under development, and its accuracy is around 70%. Please use the results as general information and consult a healthcare professional for accurate diagnosis and treatment.</p>", unsafe_allow_html=True)

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
