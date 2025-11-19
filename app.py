import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import face_recognition

# ---------------------------
# Streamlit App Settings
# ---------------------------
st.set_page_config(page_title="Face Cartoonizer", layout="wide")

st.title("🎨 Face Cartoonizer – Facial Recognition Edition")

st.write("""
Upload an image and this app will:

1. **Detect faces** using *face_recognition*  
2. **Cartoonize** the entire image or only the detected faces  
3. Let you download the cartoonized result  

This demonstrates Facial Recognition concepts using bounding-box detection.
""")

# ---------------------------
# Cartoon Effect (Pillow-based)
# ---------------------------
def cartoonize_pillow(pil_img, smooth=2, edge_enhance=2):
    # Convert to numpy
    img = np.array(pil_img)

    # Edge detection mask (Pillow)
    edges = pil_img.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    edges = edges.filter(ImageFilter.SMOOTH_MORE)

    # Smooth color areas
    result = pil_img.filter(ImageFilter.SMOOTH_MORE)
    for _ in range(smooth):
        result = result.filter(ImageFilter.SMOOTH)

    # Enhance edges
    for _ in range(edge_enhance):
        result = Image.blend(result, edges.convert("RGB"), alpha=0.25)

    return result


# ---------------------------
# Face Cartoonization Logic
# ---------------------------
def cartoonize_faces_only(img_pil, face_locations):
    img_np = np.array(img_pil).copy()

    for (top, right, bottom, left) in face_locations:
        face_roi = img_pil.crop((left, top, right, bottom))
        cartoon_face = cartoonize_pillow(face_roi)

        # Paste cartoon face back into the image
        img_pil.paste(cartoon_face, (left, top))

    return img_pil


# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

mode = st.sidebar.radio(
    "Cartoonization Mode",
    ["Cartoonize entire image", "Cartoonize faces only"]
)

smooth = st.sidebar.slider("Smoothness", 1, 5, 2)
edges = st.sidebar.slider("Edge Strength", 1, 5, 2)

# ---------------------------
# Main Processing
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Detect faces
    face_locations = face_recognition.face_locations(img_np)

    st.subheader(f"Detected {len(face_locations)} face(s).")

    # Draw bounding boxes preview
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for (top, right, bottom, left) in face_locations:
        draw.rectangle([left, top, right, bottom], outline="red", width=4)

    st.image(preview, caption="Detected Faces", use_container_width=True)

    # Apply cartoon effect
    if mode == "Cartoonize entire image":
        cartoon = cartoonize_pillow(image, smooth=smooth, edge_enhance=edges)
    else:
        cartoon = cartoonize_faces_only(image.copy(), face_locations)

    st.subheader("Cartoonized Image")
    st.image(cartoon, use_container_width=True)

    # Download button
    st.download_button(
        "Download Cartoon Image",
        data=cartoon.tobytes(),
        file_name="cartoonized.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image to get started.")
