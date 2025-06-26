import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model("mnist_model.h5")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below!")

canvas_result = st.canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
    img = img.resize((28, 28)).convert('L')
    img = ImageOps.invert(img)
    img = np.array(img).reshape(1, 28, 28) / 255.0

    if st.button("Predict"):
        prediction = model.predict(img)
        st.write(f"Prediction: **{np.argmax(prediction)}**")
