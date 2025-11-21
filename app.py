import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Function: Pixelate (lighter)
# -----------------------------
def pixelate(img, blocks=30):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# -----------------------------
# Function: Pencil Sketch
# -----------------------------
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (31, 31), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# -----------------------------
# Function: Blur Eyes Region (STATIC, ALWAYS WORKS)
# -----------------------------
def blur_eyes(img):
    h, w = img.shape[:2]

    # Define region where eyes typically are
    y1 = int(h * 0.22)
    y2 = int(h * 0.42)

    blurred = img.copy()

    # Extract region
    eye_region = blurred[y1:y2, :]

    # Apply blur
    eye_region = cv2.GaussianBlur(eye_region, (51, 51), 30)

    # Put blurred region back
    blurred[y1:y2, :] = eye_region
    return blurred

# -----------------------------
# Function: Blur Background
# -----------------------------
def blur_background(img):
    h, w = img.shape[:2]

    # center face crop (30% width, 40% height)
    top = int(h * 0.20)
    bottom = int(h * 0.75)
    left = int(w * 0.20)
    right = int(w * 0.80)

    output = cv2.GaussianBlur(img, (55, 55), 45)

    # restore face region
    output[top:bottom, left:right] = img[top:bottom, left:right]

    return output


# -----------------------------
# DISPLAY RESULTS
# -----------------------------
if uploaded:

    img = Image.open(uploaded).convert("RGB")
    img = np.array(img)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    # ORIGINAL
    with col1:
        st.image(img, caption="Original", use_column_width=True)

    # PIXELATED
    with col2:
        pix = pixelate(img_bgr)
        st.image(cv2.cvtColor(pix, cv2.COLOR_BGR2RGB),
                 caption="Pixelated Face", use_column_width=True)

    # BLURRED EYES (safe always)
    with col3:
        hidden = blur_eyes(img_bgr)
        st.image(cv2.cvtColor(hidden, cv2.COLOR_BGR2RGB),
                 caption="Eyes Blurred", use_column_width=True)

    # PENCIL SKETCH
    with col4:
        sketch = pencil_sketch(img_bgr)
        st.image(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB),
                 caption="Pencil Sketch", use_column_width=True)

    # BLUR BACKGROUND
    with col5:
        bb = blur_background(img_bgr)
        st.image(cv2.cvtColor(bb, cv2.COLOR_BGR2RGB),
                 caption="Blur Background", use_column_width=True)
