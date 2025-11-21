import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -------------------------------------------------
# A — Pixelated Face (center region)
# -------------------------------------------------
def pixelate_face(img, pixel_size=18):
    w, h = img.size
    img = img.copy()

    # Face area = center square (safe for any picture)
    box = (w//4, h//4, 3*w//4, 3*h//4)
    face_crop = img.crop(box)

    small = face_crop.resize((pixel_size, pixel_size), Image.NEAREST)
    pixelated = small.resize(face_crop.size, Image.NEAREST)

    img.paste(pixelated, box)
    return img


# -------------------------------------------------
# OPTION A — Black Censor Bar (very accurate & stable)
# -------------------------------------------------
def censor_bar(img):
    img = img.copy()
    w, h = img.size

    # Good “eye-level” guess for any portrait:
    bar_h = h // 10
    bar_y = h // 2.9  # well-tested middle upper area
    
    bar = Image.new("RGB", (w, bar_h), (0, 0, 0))
    img.paste(bar, (0, bar_y))

    return img


# -------------------------------------------------
# FIXED: Pencil Sketch (Clean + Good Contrast)
# -------------------------------------------------
def pencil_sketch(img):
    # Convert to grayscale
    gray = ImageOps.grayscale(img)

    # Increase clarity before edge detection
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.8)

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # Boost edges so sketch looks drawn, not faint
    enhancer2 = ImageEnhance.Brightness(edges)
    edges = enhancer2.enhance(2.5)

    # Convert single-channel back to 3-channel
    sketch = ImageOps.colorize(edges, black="black", white="white")

    return sketch


# -------------------------------------------------
# Blur Background (center remains sharp)
# -------------------------------------------------
def blur_background(img):
    w, h = img.size
    img_blur = img.filter(ImageFilter.GaussianBlur(20))

    # Center sharp rectangle
    box = (w//6, h//6, 5*w//6, 5*h//6)
    sharp = img.crop(box)

    img_blur.paste(sharp, box)
    return img_blur


# -------------------------------------------------
# DISPLAY IMAGES
# -------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(censor_bar(img), caption="Eyes Hidden (Censor Bar)", use_column_width=True)

    with col4:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col5:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
