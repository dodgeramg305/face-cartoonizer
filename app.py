import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Title
# -------------------------------
st.title("🎨 Face Fun Factory – 5 Transformations + Cartoonish")


# -------------------------------
# Upload Image
# -------------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.success("Image uploaded successfully!")

    # --------------------------------
    # Transformation 1 – Original
    # --------------------------------
    original = img_np.copy()

    # --------------------------------
    # Transformation 2 – Pixelated Face
    # --------------------------------
    def pixelate(image, scale=0.15):
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    pixelated = pixelate(img_np.copy())

    # --------------------------------
    # Transformation 3 – Natural Color (NO BLUE FILTER)
    # --------------------------------
    def natural_color(image):
        return image.copy()   # leaves image untouched

    natural = natural_color(img_np.copy())

    # --------------------------------
    # Transformation 4 – Hide Eyes (circles)
    # --------------------------------
    def hide_eyes(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        faces = face.detectMultiScale(gray, 1.1, 4)
        out = image.copy()

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            detected_eyes = eyes.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in detected_eyes:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                radius = int(ew * 0.7)
                cv2.circle(out, center, radius, (0, 0, 0), -1)

        return out

    eyes_hidden = hide_eyes(img_np.copy())

    # --------------------------------
    # Transformation 5 – Cartoonish Effect
    # --------------------------------
    def cartoon_effect(image):
        # Smooth colors
        color = cv2.bilateralFilter(image, 9, 200, 200)

        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            7
        )

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(color, edges)

    cartoonish = cartoon_effect(img_np.copy())


    # -------------------------------
    # Display Results (Grid Layout)
    # -------------------------------
    st.write("### Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption="Original", use_column_width=True)
    with col2:
        st.image(pixelated, caption="Pixelated Face", use_column_width=True)
    with col3:
        st.image(natural, caption="Natural Color", use_column_width=True)

    col4, col5 = st.columns(2)
    with col4:
        st.image(eyes_hidden, caption="Eyes Hidden", use_column_width=True)
    with col5:
        st.image(cartoonish, caption="Cartoonish", use_column_width=True)

else:
    st.info("👆 Upload a JPG or PNG image to get started.")
