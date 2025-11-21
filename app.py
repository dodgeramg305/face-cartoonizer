import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np

# -----------------------------
# Streamlit page settings
# -----------------------------
st.set_page_config(
    page_title="Face Fun Factory – Transformations",
    layout="wide"
)

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -----------------------------
# Helper: get "face" box
# (center rectangle – works well for typical portrait photos)
# -----------------------------
def get_face_box(img_width, img_height):
    """
    Approximate face region as a central rectangle.
    This avoids needing heavy face-detection libraries
    that don’t work on Streamlit Cloud.
    """

    x1 = int(img_width * 0.25)
    x2 = int(img_width * 0.75)
    y1 = int(img_height * 0.15)
    y2 = int(img_height * 0.85)

    # make sure box is valid
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return x1, y1, x2, y2


# -----------------------------
# A – Pixelated face (only middle region)
# -----------------------------
def pixelate_face(img: Image.Image, blocks: int = 32) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = get_face_box(w, h)

    face = img.crop((x1, y1, x2, y2))

    # downscale then upscale to create a pixelated effect
    small = face.resize((blocks, blocks), resample=Image.BILINEAR)
    pixelated = small.resize(face.size, resample=Image.NEAREST)

    out = img.copy()
    out.paste(pixelated, (x1, y1, x2, y2))
    return out


# -----------------------------
# B – Hide eyes with black circles
# -----------------------------
def hide_eyes(img: Image.Image) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = get_face_box(w, h)

    face_width = x2 - x1
    face_height = y2 - y1

    # approximate positions of eyes relative to face box
    eye_y = int(y1 + face_height * 0.35)
    left_eye_x = int(x1 + face_width * 0.32)
    right_eye_x = int(x1 + face_width * 0.68)

    # radius proportional to face size
    radius = int(min(face_width, face_height) * 0.10)

    out = img.copy()
    draw = ImageDraw.Draw(out)

    for cx in (left_eye_x, right_eye_x):
        draw.ellipse(
            (cx - radius, eye_y - radius, cx + radius, eye_y + radius),
            fill="black"
        )

    return out


# -----------------------------
# C – Pencil sketch
# -----------------------------
def pencil_sketch(img: Image.Image) -> Image.Image:
    # grayscale
    gray = ImageOps.grayscale(img)

    # invert
    inv = ImageOps.invert(gray)

    # blur inverted image
    blur = inv.filter(ImageFilter.GaussianBlur(radius=25))

    # color dodge blend: result = base * 255 / (255 - blend)
    g = np.array(gray).astype("float")
    b = np.array(blur).astype("float")

    # avoid division by zero
    result = np.minimum(255, (g * 255.0) / (255.0 - b + 1e-6))
    result = result.clip(0, 255).astype("uint8")

    sketch = Image.fromarray(result)
    return sketch.convert("RGB")


# -----------------------------
# E – Blur background, keep face sharp
# -----------------------------
def blur_background(img: Image.Image, blur_radius: int = 18) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = get_face_box(w, h)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = blurred.copy()

    face_region = img.crop((x1, y1, x2, y2))
    out.paste(face_region, (x1, y1, x2, y2))

    return out


# -----------------------------
# Main display logic
# -----------------------------
if uploaded:
    # open as PIL image and convert to RGB
    img = Image.open(uploaded).convert("RGB")

    # TOP ROW: Original, Pixelated Face, Eyes Hidden
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        pix = pixelate_face(img)
        st.image(pix, caption="Pixelated Face", use_column_width=True)

    with col3:
        eyes = hide_eyes(img)
        st.image(eyes, caption="Eyes Hidden", use_column_width=True)

    # BOTTOM ROW: Pencil Sketch, Blur Background
    col4, col5 = st.columns(2)

    with col4:
        sketch = pencil_sketch(img)
        st.image(sketch, caption="Pencil Sketch", use_column_width=True)

    with col5:
        blurred_bg = blur_background(img)
        st.image(blurred_bg, caption="Blur Background", use_column_width=True)
else:
    st.info("Upload a close-up portrait photo to see the transformations.")
