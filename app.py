# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load the model
model = load_model("mnist_model.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert the image (white digit on black)
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model

    st.image(image, caption='Processed Image', width=150)
    prediction = np.argmax(model.predict(img_array))
    st.write(f"Predicted Digit: **{prediction}**")
