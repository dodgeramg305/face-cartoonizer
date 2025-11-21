import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------
# Soft Pixelation
# -----------------------------------------------------
def pixelate(img, scale=12):
    w, h = img.size
    img_small = img.resize((max(1, w // scale), max(1, h // scale)), Image.BILINEAR)
    return img_small.resize((w, h), Image.NEAREST)


# -----------------------------------------------------
# Pencil Sketch (True Dodge Blend)
# -----------------------------------------------------
def dodge(front, back):
    """Real dodge blend: front / (255 - back)"""
    result = ImageChops.dodge(front, back)
    return result


def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blurred = inverted.filter(ImageFilter.GaussianBlur(18))

    # true dodge
    sketch = dodge(gray, blurred)

    # add edges to make stronger sketch effect
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(sketch, edges, alpha=0.35)
    return final.convert("RGB")


# -----------------------------------------------------
# Blur Background v2 (same size as others)
# -----------------------------------------------------
def blur_background(img):
    w, h = img.size
    output_size = (w, h)

    # blurred base
    blurred = img.filter(ImageFilter.GaussianBlur(22))

    # circular mask (smaller / more natural)
    mask = Image.new("L", (w, h), 0)
    r = int(min(w, h) * 0.38)  # circle radius smaller now
    cx, cy = w // 2, h // 2

    for x in range(w):
        for y in range(h):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    # final mix (circle is original, background is blurred)
    return Image.composite(img, blurred, mask)


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
        st.image(pixelate(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
