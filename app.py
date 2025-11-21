import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import face_recognition

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# ----------------------------------------------------
# A — Pixelate Only The Face Area
# ----------------------------------------------------
def pixelate_face(img):
    pil = img.copy()
    np_img = np.array(pil)

    face_locations = face_recognition.face_locations(np_img)

    if not face_locations:
        return pil  # no face detected

    top, right, bottom, left = face_locations[0]

    face = pil.crop((left, top, right, bottom))
    face_small = face.resize((20, 20), resample=Image.NEAREST)
    face_pixelated = face_small.resize(face.size, Image.NEAREST)

    pil.paste(face_pixelated, (left, top))
    return pil


# ----------------------------------------------------
# B — Eyes Hidden (Black Censor Bar)
# ----------------------------------------------------
def censor_bar(img):
    pil = img.copy()
    np_img = np.array(pil)

    face_locations = face_recognition.face_locations(np_img)

    if not face_locations:
        return pil

    top, right, bottom, left = face_locations[0]

    face_height = bottom - top
    bar_height = int(face_height * 0.18)

    bar_top = top + int(face_height * 0.30)
    bar_bottom = bar_top + bar_height

    black_bar = Image.new("RGB", (right - left, bar_height), (0, 0, 0))

    pil.paste(black_bar, (left, bar_top))
    return pil


# ----------------------------------------------------
# C — Pencil Sketch (High Quality)
# ----------------------------------------------------
def pencil_sketch(img):
    pil = img.convert("L")

    inverted = ImageOps.invert(pil)
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=30))

    sketch = Image.blend(pil, blurred, alpha=0.5)
    return sketch.convert("RGB")


# ----------------------------------------------------
# D — Blur Background
# ----------------------------------------------------
def blur_background(img):
    pil = img.copy()
    np_img = np.array(pil)

    face_locations = face_recognition.face_locations(np_img)
    if not face_locations:
        return pil

    top, right, bottom, left = face_locations[0]

    blurred = pil.filter(ImageFilter.GaussianBlur(radius=25))
    face = pil.crop((left, top, right, bottom))

    blurred.paste(face, (left, top))
    return blurred


# ----------------------------------------------------
# Display Results
# ----------------------------------------------------
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

