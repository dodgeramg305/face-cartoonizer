import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🎨 Face Fun Factory – 5 Transformations + Bunny Ears")

# Load bunny ears
BUNNY_PATH = "assets/bunny_ears.png"
bunny = Image.open(BUNNY_PATH)
bunny = np.array(bunny)

# File upload
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.success("Image uploaded successfully!")

    img = np.array(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 1. FACE DETECTION USING HAAR
    # -----------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(img_rgb, 1.3, 5)

    # -----------------------------
    # 2. PIXELATION
    # -----------------------------
    def pixelate(image, size=15):
        h, w = image.shape[:2]
        temp = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    pixelated = img_rgb.copy()
    for (x, y, w, h) in faces:
        face = pixelated[y:y+h, x:x+w]
        face = pixelate(face, size=20)
        pixelated[y:y+h, x:x+w] = face

    # -----------------------------
    # 3. BLUE FACE
