import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ------------------------------------------------
# Pixelate face (light pixelation)
# ------------------------------------------------
def pixelate(img, blocks=28):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# ------------------------------------------------
# Pencil Sketch (fixed, OpenCV only)
# ------------------------------------------------
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (25, 25), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# ------------------------------------------------
# Blur Background (keeps face sharp)
# ------------------------------------------------
def blur_background(img):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (55, 55), 0)

    # Keep face sharp by extracting a square around center
    size = min(h, w) // 2
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    x2 = x1 + size
    y2 = y1 + size

    result = blurred.copy()
    result[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return result


# ------------------------------------------------
# Display Results
# ------------------------------------------------
if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        px = pixelate(img_bgr)
        st.image(cv2.cvtColor(px, cv2.COLOR_BGR2RGB),
                 caption="Pixelated Face", use_column_width=True)

    with col3:
        sketch = pencil_sketch(img_bgr)
        st.image(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB),
                 caption="Pencil Sketch", use_column_width=True)

    with col4:
        blur = blur_background(img_bgr)
        st.image(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),
                 caption="Blur Background", use_column_width=True)
