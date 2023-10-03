import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the CNN model
model = tf.keras.models.load_model("trained_model.h5")

#Showing the logo

with open("logo_1.png", "rb") as icon_image_file:
    icon_image_data = icon_image_file.read()

st.image(icon_image_data, use_column_width=True)

# Title and Description
st.markdown("<p style='font-size: 22px; color: #5045F2; text-align: center;'>Welcome to Dermsage!</p>", unsafe_allow_html=True)

# Get user's name
st.markdown("<br>",unsafe_allow_html=True)
user_name = st.text_input("Enter your name:")

# Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_image is not None:
    # Convert RGBA image to RGB
    image = Image.open(uploaded_image).convert("RGB")
    # Resize the image to the desired dimensions
    image_for_prediction = image.resize((224, 224))
    # Convert to NumPy array
    image_for_prediction = np.asarray(image_for_prediction)
    image_for_prediction = image_for_prediction / 255.0  # Normalize the image data
    image_for_prediction = np.expand_dims(image_for_prediction, axis=0)

    # Make predictions
    prediction = model.predict(image_for_prediction)
            
    # Display the prediction
    # image_1= image.resize((150, 150))
    st.image(image, caption="Uploaded Image",use_column_width=True)
    st.success("Prediction Complete!")
    class_names = [
    "Acne / Rosacea",
    "Eczema",
    "Normal Skin",
    "Psoriasis/Lichen Planus",
    "Fungal Infections",
    "Vitiligo"
]

    # Display the prediction
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    st.markdown(f"<p style='font-size: 24px; color: #5045F2; text-align: center;'><strong>Hi {user_name}, Thanks for visiting us!</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 24px; color: #5045F2; text-align: center;'><b>Condition: </b><strong style='color:black;'>{predicted_class}</strong></p></p>", unsafe_allow_html=True)
    
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
    st.markdown("<div style='font-size: 6px; border: 2px solid black; padding: 5px; margin-top: 15px;'><p><strong>Note:</strong> The AI model used in this application is under development. Please use the results as general information and consult a healthcare professional for accurate diagnosis and treatment.</p></div>", unsafe_allow_html=True)

    
# About Dermsage
st.sidebar.markdown("<h2 style='color:#5045F2;'>About Dermsage:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Dermsage is an AI-powered skin disease detection service. We aim to provide quick and accurate skin disease diagnosis to our users.</p>", unsafe_allow_html=True)

# Benefits
st.sidebar.markdown("<h2 style='color:#5045F2;'>Benefits:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<ul><li>Fast and reliable skin disease detection.</li><li>Accessible from anywhere.</li><li>Support for multiple skin conditions.</li></ul>", unsafe_allow_html=True)

# Contact Us
st.sidebar.markdown("<h2 style='color:#5045F2;'>Contact Us:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>If you have any questions or feedback, please email us at dermsage@gmail.com.</p>", unsafe_allow_html=True)

# Terms of Service
st.sidebar.markdown("<h2 style='color:#5045F2;'>Terms of Service:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Please read and agree to our Terms of Service before using this service.")

# Privacy Policy
st.sidebar.markdown("<h2 style='color:#5045F2;'>Privacy Policy:</h2>", unsafe_allow_html=True)
st.sidebar.markdown("We take your privacy seriously. Please read our Privacy Policy.")

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
st.markdown(
    """
    <style>
    /* Custom CSS for the sidebar */
    .sidebar .markdown-text-container {
        background-color: #F5F5F5; /* Background color for the sidebar */
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
    }

    .sidebar h2 {
        color: #5045F2; /* Header text color */
        font-size: 20px; /* Header text size */
    }

    .sidebar p {
        color: #333; /* Text color */
        font-size: 14px; /* Text size */
    }

    /* Custom CSS for the main content area */
    .main-content {
        margin-left: 20px; /* Create some space between sidebar and content */
    }

    /* Custom CSS for the header */
    .header {
        background-color: #5045F2; /* Header background color */
        color: white; /* Header text color */
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }

    /* Custom CSS for buttons */
    .sidebar button {
        background-color: #5045F2; /* Button background color */
        color: white; /* Button text color */
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }

    /* Custom CSS for button hover effect */
    .sidebar button:hover {
        background-color: #322EDF; /* Hover background color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

