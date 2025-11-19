import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Cartoonizer 🎨")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original Image", width=300)

    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detect edges
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    # Smooth colors
    color = cv2.bilateralFilter(img, 9, 200, 200)

    # Combine edges + color
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    st.image(cartoon, caption="Cartoonized Image", width=300)
