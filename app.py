import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------------
# FACE ESTIMATOR (PIL-only approximate method)
# This avoids cv2. It finds the brightest area (usually the face).
# -------------------------------------------------------------------
def approximate_face_box(img):
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)

    # Find bright region by threshold
    thresh = np.percentile(arr, 75)
    mask = arr > thresh

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return (0, 0, img.size[0], img.size[1])

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Expand box a bit
    pad = int(min(img.size) * 0.08)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.size[0], x2 + pad)
    y2 = min(img.size[1], y2 + pad)

    return (x1, y1, x2, y2)

# -------------------------------------------------------------------
# Pixelate only face
# -------------------------------------------------------------------
def pixelate_face(img, scale=20):
    img2 = img.copy()
    w, h = img.size

    # estimate face box
    x1, y1, x2, y2 = approximate_face_box(img)

    face = img2.crop((x1, y1, x2, y2))
    fw, fh = face.size

    face_small = face.resize((max(1, fw//scale), max(1, fh//scale)), Image.NEAREST)
    face_pix = face_small.resize((fw, fh), Image.NEAREST)

    img2.paste(face_pix, (x1, y1, x2, y2))
    return img2

# -------------------------------------------------------------------
# Pencil sketch
# -------------------------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(15))

    # dodge
    dodge = ImageOps.blend(gray, blur, 0.5)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge, edges, 0.45)
    return final.convert("RGB")

# -------------------------------------------------------------------
# Blur background (same size as others, with circular focus)
# -------------------------------------------------------------------
def blur_background(img):
    output = img.resize((600, 600))
    blurred = output.filter(ImageFilter.GaussianBlur(25))

    w, h = output.size
    mask = Image.new("L", (w, h), 0)

    # circular mask
    cx, cy = w//2, h//2
    r = int(w * 0.35)

    for x in range(w):
        for y in range(h):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    final = Image.composite(output, blurred, mask)
    return final

# -------------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------------
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
