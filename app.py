# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model("mnist_model.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    st.image(image, caption='Processed Image', width=150)
    prediction = np.argmax(model.predict(img_array))
    st.write(f"Predicted Digit: **{prediction}**")
