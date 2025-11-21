import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------
# Pixelate ONLY the face — DO NOT TOUCH (you liked this)
# -----------------------
def pixelate_face_only(img, block_size=20):
    w, h = img.size
    img_np = np.array(img)

    # Rough face bounding box estimation (center vertical region)
    face_top = int(h * 0.20)
    face_bottom = int(h * 0.75)
    face_left = int(w * 0.25)
    face_right = int(w * 0.75)

    face_region = img_np[face_top:face_bottom, face_left:face_right]

    # Pixelation
    small = Image.fromarray(face_region).resize(
        (face_region.shape[1] // block_size, face_region.shape[0] // block_size),
        resample=Image.NEAREST
    )
    pixelated = small.resize((face_region.shape[1], face_region.shape[0]), Image.NEAREST)
    img_np[face_top:face_bottom, face_left:face_right] = np.array(pixelated)

    return Image.fromarray(img_np)


# -----------------------
# Pencil Sketch (fixed version, not dark, Streamlit-safe)
# -----------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=18))

    # Convert to numpy
    g = np.array(gray).astype("float")
    b = np.array(blur).astype("float")

    # Dodge blend formula
    dodge = g * 255 / (255 - b + 1)
    dodge = np.clip(dodge, 0, 255).astype("uint8")

    return Image.fromarray(dodge).convert("RGB")


# -----------------------
# Blur Background (same size as others, centered)
# -----------------------
def blur_background(img):
    # Output size should match grid size
    out_size = (400, 400)

    # Resize background
    blurred = img.filter(ImageFilter.GaussianBlur(18)).resize(out_size)

    # Create mask (circle)
    mask = Image.new("L", out_size, 0)
    r = int(out_size[0] * 0.38)
    cx, cy = out_size[0] // 2, out_size[1] // 2

    for x in range(out_size[0]):
        for y in range(out_size[1]):
            if (x - cx)**2 + (y - cy)**2 <= r*r:
                mask.putpixel((x, y), 255)

    # Crop and resize original face
    face = img.resize(out_size)

    return Image.composite(face, blurred, mask)


# -----------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face_only(img), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
