import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# =======================================================
#  FACE DETECTION (simple heuristic)
# =======================================================
def detect_face_bbox(img):
    img_small = img.resize((256, 256)).convert("L")
    arr = np.array(img_small)

    ys, xs = np.where(arr > np.percentile(arr, 60))
    if len(xs) == 0:
        return None

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    W, H = img.size
    return (
        int(x1 * (W/256)),
        int(y1 * (H/256)),
        int(x2 * (W/256)),
        int(y2 * (H/256)),
    )


# =======================================================
#  PIXELATE ONLY THE FACE
# =======================================================
def pixelate_face_only(img, blocks=12):
    box = detect_face_bbox(img)
    if box is None:
        return img

    x1, y1, x2, y2 = box
    face = img.crop((x1, y1, x2, y2))

    w, h = face.size
    small = face.resize((max(1, w//blocks), max(1, h//blocks)), Image.NEAREST)
    pix = small.resize((w, h), Image.NEAREST)

    out = img.copy()
    out.paste(pix, (x1, y1))
    return out


# =======================================================
#  PENCIL SKETCH (clean)
# =======================================================
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(18))

    gray_np = np.array(gray).astype(float)
    blur_np = np.array(blur).astype(float)

    dodge = gray_np * 255 / (255 - blur_np + 1)
    dodge = np.clip(dodge, 0, 255).astype("uint8")

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edges_np = np.array(edges).astype(float)

    final = (0.75 * dodge + 0.25 * edges_np).astype("uint8")
    return Image.fromarray(final).convert("RGB")


# =======================================================
#  BLUR BACKGROUND
# =======================================================
def blur_background(img):
    return img.filter(ImageFilter.GaussianBlur(20))


# =======================================================
#  DISPLAY — RESTORED LAYOUT
# =======================================================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # -----------------------
    #  TOP: 3 small boxes
    # -----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face_only(img), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)

    # Space
    st.markdown("## ")

    # -----------------------
    #  BOTTOM: MEDIUM BOX
    # -----------------------
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(pencil_sketch(img), caption="Pencil Sketch", width=450)
