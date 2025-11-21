import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------
# A — PIXELATE ONLY THE FACE
# -----------------------------------------------------
def pixelate_face(img, block_size=15):
    w, h = img.size
    img_np = np.array(img)

    # Approximate face box = center 60%
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    y1 = int(h * 0.20)
    y2 = int(h * 0.80)

    face_region = img_np[y1:y2, x1:x2]

    # Pixelation
    small = Image.fromarray(face_region).resize(
        (max(1, (x2 - x1) // block_size), max(1, (y2 - y1) // block_size)),
        resample=Image.NEAREST
    )

    pixelated = small.resize((x2 - x1, y2 - y1), Image.NEAREST)

    img_np[y1:y2, x1:x2] = np.array(pixelated)
    return Image.fromarray(img_np)

# -----------------------------------------------------
# B — PENCIL SKETCH (PIL-ONLY)
# -----------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)

    # strong edges
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    # blend with grayscale to soften it
    final = Image.blend(gray, edges, alpha=0.6)

    return final.convert("RGB")

# -----------------------------------------------------
# C — BLUR BACKGROUND SAME SIZE
# -----------------------------------------------------
def blur_background(img):
    output_size = (600, 600)

    base = img.resize(output_size)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=20)).resize(output_size)

    # create circular mask
    mask = Image.new("L", output_size, 0)
    draw = ImageDraw.Draw(mask)

    r = int(output_size[0] * 0.38)
    cx, cy = output_size[0] // 2, output_size[1] // 2
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

    final = Image.composite(base, blurred, mask)
    return final

# -----------------------------------------------------
# DISPLAY
# -----------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face (Face Only)", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
