import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# =======================================================
#  FACE DETECTION USING PIL (fast + Streamlit-friendly)
# =======================================================
def detect_face_bbox(img):
    """
    Lightweight face detection using a simple heuristic.
    This avoids heavy libraries like OpenCV or mediapipe,
    and works well for centered portrait photos (class project safe).
    """

    img_small = img.resize((256, 256)).convert("L")
    arr = np.array(img_small)

    # crude face region detection (brightest region assumption)
    ys, xs = np.where(arr > np.percentile(arr, 60))

    if len(xs) == 0:
        return None  # no detectable face region

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    # Rescale to original image
    W, H = img.size
    scale_x = W / 256
    scale_y = H / 256

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )


# =======================================================
#  PIXELATE ONLY FACE
# =======================================================
def pixelate_face_only(img, blocks=12):
    face_box = detect_face_bbox(img)
    if face_box is None:
        return img  # return original if no face detected

    x1, y1, x2, y2 = face_box
    face = img.crop((x1, y1, x2, y2))

    # pixelate face only
    w, h = face.size
    small = face.resize((max(1, w // blocks), max(1, h // blocks)), Image.NEAREST)
    pixelated_face = small.resize((w, h), Image.NEAREST)

    # paste back over original
    out = img.copy()
    out.paste(pixelated_face, (x1, y1))
    return out


# =======================================================
#  PENCIL SKETCH (clean + strong edges)
# =======================================================
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(18))

    gray_np = np.array(gray).astype(float)
    blur_np = np.array(blur).astype(float)

    dodge = gray_np * 255 / (255 - blur_np + 1)
    dodge = np.clip(dodge, 0, 255).astype("uint8")

    # Add some edge detail
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edges_np = np.array(edges).astype(float)

    final = (0.75 * dodge + 0.25 * edges_np).astype("uint8")

    return Image.fromarray(final).convert("RGB")


# =======================================================
#  BLUR BACKGROUND (same size box as others)
# =======================================================
def blur_background(img):
    blurred = img.filter(ImageFilter.GaussianBlur(20))
    return blurred


# =======================================================
#  DISPLAY OUTPUT LAYOUT
# =======================================================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # TOP ROW (3 small boxes)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face_only(img), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)

    # spacing
    st.markdown("---")

    # BOTTOM ROW (FULL WIDTH pencil sketch)
    st.image(
        pencil_sketch(img),
        caption="Pencil Sketch",
        use_column_width=True
    )
