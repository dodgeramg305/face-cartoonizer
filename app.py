import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------
# UTIL: Estimate face crop (center area of image)
# ----------------------------------------------------
def estimate_face_region(img):
    w, h = img.size
    face_width = int(w * 0.45)
    face_height = int(h * 0.50)

    cx, cy = w // 2, int(h * 0.45)

    x1 = max(0, cx - face_width // 2)
    y1 = max(0, cy - face_height // 2)
    x2 = min(w, cx + face_width // 2)
    y2 = min(h, cy + face_height // 2)

    return (x1, y1, x2, y2)

# ----------------------------------------------------
# PIXELATE ONLY FACE
# ----------------------------------------------------
def pixelate_face_only(img, scale=25):
    img_copy = img.copy()
    x1, y1, x2, y2 = estimate_face_region(img)

    face_crop = img_copy.crop((x1, y1, x2, y2))

    small = face_crop.resize(
        (face_crop.width // scale, face_crop.height // scale),
        resample=Image.BILINEAR
    )
    pixelated = small.resize(face_crop.size, Image.NEAREST)

    img_copy.paste(pixelated, (x1, y1))
    return img_copy

# ----------------------------------------------------
# CLEAN BRIGHT PENCIL SKETCH
# ----------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)

    blur = inv.filter(ImageFilter.GaussianBlur(radius=22))
    dodge = ImageOps.blend(gray, blur, 0.1)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge, edges, 0.25)
    return final

# ----------------------------------------------------
# BLUR BACKGROUND (same square format)
# ----------------------------------------------------
def blur_background(img):
    output_size = (600, 600)

    face_center_crop = img.resize(output_size)

    blurred_bg = img.filter(
        ImageFilter.GaussianBlur(radius=30)
    ).resize(output_size)

    mask = Image.new("L", output_size, 0)
    cx, cy = output_size[0] // 2, output_size[1] // 2
    r = int(output_size[0] * 0.40)

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                mask.putpixel((x, y), 255)

    final = Image.composite(face_center_crop, blurred_bg, mask)
    return final

# ----------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face_only(img), caption="Pixelated Face (Face Only)", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
