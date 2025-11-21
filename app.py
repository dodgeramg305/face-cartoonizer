import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -----------------------------
# PIXELATE JUST THE FACE AREA
# -----------------------------
def pixelate_face(img):
    w, h = img.size

    # define a center box (face area estimate)
    box = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))

    face = img.crop(box)

    # pixelate
    small = face.resize((25, 25), resample=Image.NEAREST)
    pixelated = small.resize(face.size, Image.NEAREST)

    output = img.copy()
    output.paste(pixelated, box)
    return output


# -----------------------------
# CENSOR BAR
# -----------------------------
def censor_bar(img):
    w, h = img.size
    bar_height = int(h * 0.10)        # 10% of image height
    bar_y = int(h * 0.30)             # position around eye level

    bar = Image.new("RGB", (w, bar_height), (0, 0, 0))

    output = img.copy()
    output.paste(bar, (0, bar_y))
    return output


# -----------------------------
# PENCIL SKETCH (PIL)
# -----------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)

    blur = inverted.filter(ImageFilter.GaussianBlur(25))

    sketch = ImageOps.colorize(ImageOps.autocontrast(ImageChops.dodge(gray, blur)), black="black", white="white")
    return sketch


# -----------------------------
# BLUR BACKGROUND (keep center sharp)
# -----------------------------
def blur_background(img):
    w, h = img.size

    # estimated face area box
    box = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))

    # blur full image
    blurred = img.filter(ImageFilter.GaussianBlur(15))

    # paste sharp face
    face_region = img.crop(box)
    blurred.paste(face_region, box)
    return blurred


# -----------------------------
# DISPLAY ALL RESULTS
# -----------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    # Original
    with col1:
        st.image(img, caption="Original", use_column_width=True)

    # Pixelated Face
    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face", use_column_width=True)

    # Censor Bar
    with col3:
        st.image(censor_bar(img), caption="Eyes Hidden (Censor Bar)", use_column_width=True)

    # Pencil Sketch
    with col4:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    # Blur Background
    with col5:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
