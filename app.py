import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# ---------------------------------------------------
#  A) PIXELATE FACE (lighter pixelation)
# ---------------------------------------------------
def pixelate(img, block_size=18):
    pil = Image.fromarray(img)
    w, h = pil.size

    small = pil.resize((w // block_size, h // block_size), Image.NEAREST)
    result = small.resize((w, h), Image.NEAREST)

    return result


# ---------------------------------------------------
#  B) PENCIL SKETCH (clean + stable version)
# ---------------------------------------------------
def pencil_sketch(img):
    pil = Image.fromarray(img).convert("L")  # grayscale
    inverted = ImageOps.invert(pil)
    blurred = inverted.filter(ImageFilter.GaussianBlur(22))

    # Dodge blend
    blend = Image.blend(pil, blurred, 0.2)

    return blend.convert("RGB")  # return RGB for Streamlit


# ---------------------------------------------------
#  C) BLUR BACKGROUND (circular sharp face)
# ---------------------------------------------------
def blur_background(img):
    pil = Image.fromarray(img)

    blurred = pil.filter(ImageFilter.GaussianBlur(25))

    mask = Image.new("L", pil.size, 0)
    draw = ImageDraw.Draw(mask)

    cx, cy = pil.size[0] // 2, pil.size[1] // 2
    r = min(pil.size) // 3

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)

    result = Image.composite(pil, blurred, mask)

    return result


# ---------------------------------------------------
#            DISPLAY OUTPUT
# ---------------------------------------------------
if uploaded:

    img = Image.open(uploaded)
    img_np = np.array(img)

    col1, col2, col3 = st.columns(3)

    # ORIGINAL
    with col1:
        st.image(img, caption="Original", use_column_width=True)

    # PIXELATED FACE
    with col2:
        pix = pixelate(img_np)
        st.image(pix, caption="Pixelated Face", use_column_width=True)

    # PENCIL SKETCH
    with col3:
        sketch = pencil_sketch(img_np)
        st.image(sketch, caption="Pencil Sketch", use_column_width=True)

    # BLUR BACKGROUND (full width)
    st.subheader("")
    st.image(blur_background(img_np), caption="Blur Background", use_column_width=True)
