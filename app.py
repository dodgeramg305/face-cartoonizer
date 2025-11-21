import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import mediapipe as mp

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Face Fun Factory – Transformations",
    layout="wide"
)

st.title("Face Fun Factory – Transformations")
st.write("Upload a face photo to see multiple fun transformations at once!")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --------------------------------------------------
# MediaPipe setup (for landmarks)
# --------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh


def detect_face_landmarks(img_pil):
    """
    Runs MediaPipe Face Mesh on a PIL image and returns:
      - face_landmarks: MediaPipe landmarks object (or None)
      - face_box: (x1, y1, x2, y2) bounding box of the face in pixel coords (or None)
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    h, w, _ = img_rgb.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, None

    face = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    x1 = int(min(xs) * w)
    x2 = int(max(xs) * w)
    y1 = int(min(ys) * h)
    y2 = int(max(ys) * h)

    # Small padding around the face
    pad_x = int(0.05 * (x2 - x1))
    pad_y = int(0.05 * (y2 - y1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return face, (x1, y1, x2, y2)


# --------------------------------------------------
# A. Pixelate features (eyes + mouth region)
# --------------------------------------------------
def pixelate_region(img_pil, box, block_size=12):
    """
    Pixelates only the specified rectangle region in the image.
    box = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return img_pil

    face_crop = img_pil.crop(box)

    # Downsample then upsample (nearest) to get big blocks
    small_w = max(1, face_crop.width // block_size)
    small_h = max(1, face_crop.height // block_size)
    small = face_crop.resize((small_w, small_h), Image.NEAREST)
    pixelated = small.resize(face_crop.size, Image.NEAREST)

    out = img_pil.copy()
    out.paste(pixelated, box)
    return out


def get_feature_box_from_face(face_box):
    """
    From the full face bounding box, shrink vertically so we
    mainly cover eyes + nose + mouth (middle band of the face).
    """
    x1, y1, x2, y2 = face_box
    height = y2 - y1
    # keep the central 60% of the face height
    new_y1 = int(y1 + 0.20 * height)
    new_y2 = int(y1 + 0.80 * height)
    return x1, new_y1, x2, new_y2


# --------------------------------------------------
# B. Hide eyes (small black circles)
# --------------------------------------------------
def hide_eyes(img_pil, face_landmarks):
    """
    Draws smaller black circles over each eye, based on landmarks.
    """
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def to_px(lm):
        return int(lm.x * w), int(lm.y * h)

    # Key landmarks for eyes from MediaPipe mesh
    left_inner = face_landmarks.landmark[33]
    left_outer = face_landmarks.landmark[133]
    right_inner = face_landmarks.landmark[362]
    right_outer = face_landmarks.landmark[263]

    # Left eye
    lx1, ly1 = to_px(left_inner)
    lx2, ly2 = to_px(left_outer)
    lcx = (lx1 + lx2) // 2
    lcy = (ly1 + ly2) // 2
    lr = int(np.hypot(lx1 - lx2, ly1 - ly2) * 0.6)  # 0.6 = fairly tight

    # Right eye
    rx1, ry1 = to_px(right_inner)
    rx2, ry2 = to_px(right_outer)
    rcx = (rx1 + rx2) // 2
    rcy = (ry1 + ry2) // 2
    rr = int(np.hypot(rx1 - rx2, ry1 - ry2) * 0.6)

    # Draw circles
    draw.ellipse((lcx - lr, lcy - lr, lcx + lr, lcy + lr), fill="black")
    draw.ellipse((rcx - rr, rcy - rr, rcx + rr, rcy + rr), fill="black")

    return img


# --------------------------------------------------
# C. Pencil sketch (PIL version, no OpenCV)
# --------------------------------------------------
def pencil_sketch(img_pil):
    """
    Creates a pencil sketch effect using grayscale + inverted blur + color dodge.
    """
    gray = img_pil.convert("L")
    gray_np = np.array(gray).astype("float")

    inverted = 255 - gray_np
    inverted_img = Image.fromarray(inverted.astype("uint8"))
    blurred = inverted_img.filter(ImageFilter.GaussianBlur(radius=25))
    blurred_np = np.array(blurred).astype("float")

    # Color dodge blend
    dodge = np.minimum(255, (gray_np * 255) / (255 - blurred_np + 1e-6))
    sketch_np = dodge.astype("uint8")

    sketch_img = Image.fromarray(sketch_np).convert("RGB")
    return sketch_img


# --------------------------------------------------
# E. Blur background (keep face sharp)
# --------------------------------------------------
def blur_background(img_pil, face_box, blur_radius=18):
    if face_box is None:
        # Just blur the whole image if no face detected
        return img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    x1, y1, x2, y2 = face_box

    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    face_region = img_pil.crop(face_box)
    blurred.paste(face_region, (x1, y1, x2, y2))
    return blurred


# --------------------------------------------------
# MAIN: When user uploads an image
# --------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Get face landmarks & bounding box once
    face_landmarks, face_box = detect_face_landmarks(img)

    if face_landmarks is None:
        st.warning("I couldn't detect a face very well. "
                   "I'll still show transformations, but some may be less precise.")

    # Layout: 3 on top, 2 on bottom
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    # 1️⃣ Original
    with col1:
        st.image(img, caption="Original", use_column_width=True)

    # 2️⃣ Pixelated features (eyes + mouth)
    with col2:
        if face_box is not None:
            feature_box = get_feature_box_from_face(face_box)
            pix_img = pixelate_region(img, feature_box, block_size=10)
        else:
            pix_img = pixelate_region(img, (0, 0, img.width, img.height), block_size=10)
        st.image(pix_img, caption="Pixelated Features", use_column_width=True)

    # 3️⃣ Eyes Hidden
    with col3:
        if face_landmarks is not None:
            hidden = hide_eyes(img, face_landmarks)
        else:
            hidden = img
        st.image(hidden, caption="Eyes Hidden", use_column_width=True)

    # 4️⃣ Pencil Sketch
    with col4:
        sketch = pencil_sketch(img)
        st.image(sketch, caption="Pencil Sketch", use_column_width=True)

    # 5️⃣ Blur Background (face sharp)
    with col5:
        blurred_bg = blur_background(img, face_box)
        st.image(blurred_bg, caption="Blurred Background (Face in Focus)", use_column_width=True)
