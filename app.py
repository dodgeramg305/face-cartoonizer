import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import face_recognition

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --------------------------------------------------
# FACE DETECTION using face_recognition
# --------------------------------------------------
def get_face_box(img):
    np_img = np.array(img)
    boxes = face_recognition.face_locations(np_img)

    if len(boxes) == 0:
        return None

    # top, right, bottom, left
    t, r, b, l = boxes[0]
    return (l, t, r, b)


# --------------------------------------------------
# PIXELATE ONLY THE FACE
# --------------------------------------------------
def pixelate_face(img, pixel_size=10):
    box = get_face_box(img)
    if not box:
        return img

    l, t, r, b = box
    face = img.crop((l, t, r, b))

    # shrink → expand = pixelation
    w, h = face.size
    face_small = face.resize((w // pixel_size, h // pixel_size), Image.NEAREST)
    face_pix = face_small.resize((w, h), Image.NEAREST)

    result = img.copy()
    result.paste(face_pix, (l, t))
    return result


# --------------------------------------------------
# PENCIL SKETCH (safe method)
# --------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)

    inverted = ImageOps.invert(gray)
    blurred = inverted.filter(ImageFilter.GaussianBlur(18))

    # Manual dodge (fixes the crash)
    gray_np = np.array(gray).astype("float")
    blurred_np = np.array(blurred).astype("float")

    dodge = np.minimum(255, (gray_np * 255) / (255 - blurred_np + 1))
    dodge_img = Image.fromarray(dodge.astype("uint8"))

    # Add edges overlay
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge_img, edges, alpha=0.35)
    return final.convert("RGB")


# --------------------------------------------------
# BLUR BACKGROUND (same square size)
# --------------------------------------------------
def blur_background(img, size=400):
    img = img.resize((size, size))
    blurred = img.filter(ImageFilter.GaussianBlur(22))

    # create circular mask
    mask = Image.new("L", (size, size), 0)
    r = int(size * 0.38)
    cx, cy = size // 2, size // 2

    for x in range(size):
        for y in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                mask.putpixel((x, y), 255)

    return Image.composite(img, blurred, mask)


# --------------------------------------------------
# SHOW RESULTS
# --------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
