import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

st.set_page_config(
    page_title="Face Fun Factory – Transformations",
    layout="wide"
)

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -------------------------------------------------
# Helper: compute a “face center” box for effects
# (roughly eyes + nose + mouth area)
# -------------------------------------------------
def get_face_box(img):
    w, h = img.size
    cx = w // 2
    cy = int(h * 0.45)          # a bit above exact center
    fw = int(w * 0.50)          # width of face box
    fh = int(h * 0.55)          # height of face box

    left = max(0, cx - fw // 2)
    top = max(0, cy - fh // 2)
    right = min(w, cx + fw // 2)
    bottom = min(h, cy + fh // 2)

    return left, top, right, bottom


# -------------------------------------------------
# A – Pixelate face center (eyes + nose + mouth)
# -------------------------------------------------
def pixelate_face_center(img, pixel_fraction=0.12):
    """
    Pixelates only the central face region.
    pixel_fraction controls how chunky the blocks are.
    """
    img = img.convert("RGB")
    w, h = img.size
    left, top, right, bottom = get_face_box(img)

    face = img.crop((left, top, right, bottom))
    fw, fh = face.size

    # Downscale then upscale using NEAREST to create pixelation
    small_w = max(6, int(fw * pixel_fraction))
    small_h = max(6, int(fh * pixel_fraction))
    small = face.resize((small_w, small_h), resample=Image.BILINEAR)
    pixelated = small.resize((fw, fh), resample=Image.NEAREST)

    out = img.copy()
    out.paste(pixelated, (left, top))
    return out


# -------------------------------------------------
# B – Hide eyes with small circles (approx positions)
# -------------------------------------------------
def hide_eyes_with_circles(img):
    img = img.convert("RGB")
    w, h = img.size
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Approximate eye locations as fractions of image size
    eye_y = int(h * 0.35)
    left_x = int(w * 0.35)
    right_x = int(w * 0.65)

    radius = int(min(w, h) * 0.06)  # smaller circles than before

    for x in (left_x, right_x):
        draw.ellipse(
            (x - radius, eye_y - radius, x + radius, eye_y + radius),
            fill=(0, 0, 0)
        )

    return out


# -------------------------------------------------
# C – Pencil Sketch (no black output)
# -------------------------------------------------
def pencil_sketch(img):
    img = img.convert("RGB")
    gray = img.convert("L")
    gray_np = np.array(gray).astype(np.float32)

    # Invert & blur
    inverted = 255 - gray_np
    inverted_img = Image.fromarray(inverted.astype(np.uint8))
    blurred = inverted_img.filter(ImageFilter.GaussianBlur(radius=15))
    blurred_np = np.array(blurred).astype(np.float32)

    # Color dodge blend: sketch = gray * 255 / (255 - blurred)
    dodge = gray_np * 255.0 / (255.0 - blurred_np + 1e-5)
    dodge = np.clip(dodge, 0, 255).astype(np.uint8)

    sketch = Image.fromarray(dodge, mode="L")
    return sketch.convert("RGB")


# -------------------------------------------------
# E – Blur background, keep face center sharp
# -------------------------------------------------
def blur_background_keep_face_sharp(img):
    img = img.convert("RGB")
    w, h = img.size
    left, top, right, bottom = get_face_box(img)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=12))

    face = img.crop((left, top, right, bottom))
    out = blurred.copy()
    out.paste(face, (left, top))
    return out


# -------------------------------------------------
# Main UI
# -------------------------------------------------
if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    # Top row: Original, Pixelated Face, Eyes Hidden
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(pil_img, caption="Original", use_column_width=True)

    with col2:
        pix = pixelate_face_center(pil_img)
        st.image(pix, caption="Pixelated Face Center", use_column_width=True)

    with col3:
        eyes_hidden = hide_eyes_with_circles(pil_img)
        st.image(eyes_hidden, caption="Eyes Hidden", use_column_width=True)

    # Bottom row: Pencil Sketch, Blur Background
    col4, col5 = st.columns(2)

    with col4:
        sketch_img = pencil_sketch(pil_img)
        st.image(sketch_img, caption="Pencil Sketch", use_column_width=True)

    with col5:
        blurred_bg = blur_background_keep_face_sharp(pil_img)
        st.image(blurred_bg, caption="Blur Background (Face Sharp)", use_column_width=True)
