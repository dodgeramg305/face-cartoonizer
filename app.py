import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -------------------------------------------------
# A — Pixelated Face (center area pixelation)
# -------------------------------------------------
def pixelate_face(img, pixel_size=12):
    w, h = img.size
    img = img.copy()

    # Face area = center square
    box = (w//4, h//4, 3*w//4, 3*h//4)
    face_crop = img.crop(box)

    small = face_crop.resize((pixel_size, pixel_size), Image.NEAREST)
    pixelated = small.resize(face_crop.size, Image.NEAREST)

    img.paste(pixelated, box)
    return img


# -------------------------------------------------
# B — Hide Eyes (simple censor bar)
# Works for *all* images — no AI needed
# -------------------------------------------------
def hide_eyes(img):
    img = img.copy()
    w, h = img.size

    # Bar across top-middle of face
    bar_h = h // 10
    bar_y = h // 3

    censor = Image.new("RGB", (w, bar_h), (0, 0, 0))
    img.paste(censor, (0, bar_y))

    return img


# -------------------------------------------------
# C — Pencil Sketch (PIL)
# -------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blur = inverted.filter(ImageFilter.GaussianBlur(25))
    sketch = ImageOps.colorize(ImageOps.invert(Image.blend(gray, blur, 0.75)),
                               black="black", white="white")
    return sketch


# -------------------------------------------------
# E — Blur Background (center sharp)
# -------------------------------------------------
def blur_background(img):
    w, h = img.size
    img_blur = img.filter(ImageFilter.GaussianBlur(18))

    # Center crop stays sharp
    box = (w//6, h//6, 5*w//6, 5*h//6)
    sharp = img.crop(box)

    img_blur.paste(sharp, box)
    return img_blur


# -------------------------------------------------
# DISPLAY
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
        st.image(hide_eyes(img), caption="Eyes Hidden", use_column_width=True)

    with col4:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col5:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
