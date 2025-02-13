import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import base64

# Load the trained model
model = tf.keras.models.load_model('final_model_saved')

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Showing the logo
with open("logo_1.png", "rb") as icon_image_file:
    icon_image_data = icon_image_file.read()

st.image(icon_image_data, use_column_width=True)

# Title
st.markdown("<p class='header' style='font-size:20px;'><strong>Welcome to Dermsage!</strong></p>", unsafe_allow_html=True)

# User Input: Name
user_name = st.text_input("Enter your name:")

# User Input: Image Upload
uploaded_image = st.file_uploader("Drag or upload image here", type=["jpg", "jpeg", "png"])

# User Input: Image URL
image_url = st.text_input("Or enter an image URL:")

# Function to process the image
def process_image(image):
    image = image.convert("RGB")
    image_for_prediction = image.resize((224, 224))
    image_for_prediction = np.asarray(image_for_prediction) / 255.0  # Normalize
    return np.expand_dims(image_for_prediction, axis=0)

image = None
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif image_url:
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        else:
            st.error("Failed to load image from URL.")
    except Exception as e:
        st.error(f"Error loading image: {e}")

if image:
    image_for_prediction = process_image(image)
    prediction = model.predict(image_for_prediction)
    predicted_class_index = np.argmax(prediction)
    
    # Class labels
    class_names = [
        "Acne / Rosacea", "Eczema", "Normal Skin", "Psoriasis/Lichen Planus", "Fungal Infections", "Vitiligo"
    ]
    predicted_class = class_names[predicted_class_index]
    
    # Links to conditions
    links = {
        "Vitiligo": "https://www.mayoclinic.org/diseases-conditions/vitiligo/symptoms-causes/syc-20355912",
        "Psoriasis/Lichen Planus": "https://www.mayoclinic.org/diseases-conditions/psoriasis/symptoms-causes/syc-20355840",
        "Acne / Rosacea": "https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047",
        "Eczema": "https://my.clevelandclinic.org/health/diseases/9998-eczema",
        "Fungal Infections": "https://www.mayoclinic.org/diseases-conditions/ringworm-body/symptoms-causes/syc-20353780"
    }
    
    anchor = links.get(predicted_class, "#")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"<p style='font-size: 24px; color: #5045F2; text-align: center;'><b>{user_name}, here is your predicted skin condition:</b> <a href={anchor}>{predicted_class}</a></p>", unsafe_allow_html=True)
