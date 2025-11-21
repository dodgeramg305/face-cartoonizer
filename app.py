import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------
# Pixelation (soft)
# -----------------------
def pixelate(img, scale=22):
    w, h = img.size
    img_small = img.resize((w // scale, h // scale), resample=Image.BILINEAR)
    return img_small.resize((w, h), Image.NEAREST)

# -----------------------
# Pencil Sketch (fixed - no ImageOps.dodge)
# -----------------------

def dodge(front, back):
    """Manual dodge blend using numpy."""
    front = np.asarray(front).astype('float')
    back = np.asarray(back).astype('float')

    result = back * 255 / (255 - front + 1)
    result[result > 255] = 255
    result[front == 255] = 255  # avoid division issues

    return Image.fromarray(result.astype('uint8'))

def pencil_sketch(img):
    gray = ImageOps.grayscale(img)

    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(18))

    # dodge blend
    sketch = dodge(gray, blur)

    # add stronger edges
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(sketch, edges, alpha=0.40)
    return final.convert("RGB")

# -----------------------
# Blur Background (square)
# -----------------------
def blur_background(img):
    output_size = (600, 600)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=22)).resize(output_size)

    # mask circle
    mask = Image.new("L", output_size, 0)
    r = int(output_size[0] * 0.33)
    cx, cy = output_size[0] // 2, output_size[1] // 2

    # draw circle mask manually
    mask_pixels = mask.load()
    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask_pixels[x, y] = 255

    face = img.resize(output_size)
    return Image.composite(face, blurred, mask)

# -----------------------------------------------------
# DISPLAY OUTPUT
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
