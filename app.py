import streamlit as st
from PIL import Image, ImageFilter
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------
# Helper: Convert to NumPy
# ----------------------------------------------------
def to_np(img):
    return np.array(img)

# ----------------------------------------------------
# A – PIXELATE THE FACE AREA ONLY
# ----------------------------------------------------
def pixelate_face(img):
    np_img = np.array(img)
    h, w, _ = np_img.shape

    # Crop face area (middle vertical band)
    top = int(h * 0.20)
    bottom = int(h * 0.80)
    left = int(w * 0.25)
    right = int(w * 0.75)

    face_crop = img.crop((left, top, right, bottom))
    face_small = face_crop.resize((20, 20), resample=Image.NEAREST)
    face_pixelated = face_small.resize(face_crop.size, Image.NEAREST)

    out = img.copy()
    out.paste(face_pixelated, (left, top))
    return out

# ----------------------------------------------------
# B – HIDE EYES (approximate based on image geometry)
# ----------------------------------------------------
def hide_eyes(img):
    np_img = np.array(img)
    h, w = np_img.shape[:2]

    # Approximate eye positions
    eye_y = int(h * 0.40)
    left_eye_x = int(w * 0.33)
    right_eye_x = int(w * 0.66)

    radius = int(min(h, w) * 0.05)

    out = img.copy()
    draw = Image.new("RGBA", out.size)
    draw_np = np.array(draw)

    yy, xx = np.ogrid[:h, :w]
    mask_left = (xx - left_eye_x)**2 + (yy - eye_y)**2 <= radius**2
    mask_right = (xx - right_eye_x)**2 + (yy - eye_y)**2 <= radius**2

    np_img[mask_left] = [0, 0, 0]
    np_img[mask_right] = [0, 0, 0]

    return Image.fromarray(np_img)

# ----------------------------------------------------
# C – PENCIL SKETCH (strong + clean)
# ----------------------------------------------------
def pencil_sketch(img):
    gray = img.convert("L")
    inv = Image.fromarray(255 - np.array(gray))
    blur = inv.filter(ImageFilter.GaussianBlur(25))

    sketch_np = (np.array(gray) * 255 / (np.array(blur) + 1))
    sketch_np = sketch_np.clip(0, 255).astype(np.uint8)

    return Image.fromarray(sketch_np).convert("RGB")

# ----------------------------------------------------
# E – BLUR BACKGROUND ONLY
# ----------------------------------------------------
def blur_background(img):
    w, h = img.size
    blurred = img.filter(ImageFilter.GaussianBlur(12))

    # Face area = center 50%
    left = int(w * 0.25)
    right = int(w * 0.75)
    top = int(h * 0.20)
    bottom = int(h * 0.80)

    sharp_face = img.crop((left, top, right, bottom))
    blurred.paste(sharp_face, (left, top))

    return blurred

# ----------------------------------------------------
# DISPLAY RESULTS
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
        st.image(hide_eyes(img), caption="Eyes Hidden", use_column_width=True)

    with col4:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col5:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
