
import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from model import UNet
import io

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Satellite Flood Detection AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🛰️ Satellite Flood Detection System")
st.markdown(
    "An interactive **deep learning application** that uses a "
    "U-Net model to segment potential flood regions in satellite imagery."
)


# ============================================================
# 2. DEVICE AND MODEL LOADING
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "flood_model_finetuned_backup.pth"
)


@st.cache_resource
def load_model(model_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model = UNet(n_channels=3, n_classes=1)

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


try:
    model = load_model(str(MODEL_PATH))
    st.sidebar.success(f"Model loaded: {DEVICE.type.upper()}")
except Exception as error:
    st.error(f"Could not load the U-Net model: {error}")
    st.info(
        "Keep app.py, model.py, and flood_model_final.pth "
        "in the same project directory."
    )
    st.stop()

# ============================================================
# 3. PREPROCESSING
# Must match the training notebook
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def predict_flood(image, threshold):
    """Return probability map, binary mask, and flood-pixel percentage."""

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary_mask = (probabilities >= threshold).astype(np.uint8)
    flood_percentage = float(binary_mask.mean() * 100)

    return probabilities, binary_mask, flood_percentage


def create_overlay(image, mask):
    """Highlight predicted flood pixels in red on the original image."""

    image_array = np.array(image).copy()

    # Resize the predicted mask back to the original image dimensions
    mask_pil = Image.fromarray(mask * 255).resize(
        image.size,
        Image.Resampling.NEAREST
    )
    mask_original = np.array(mask_pil) > 0

    overlay = image_array.copy()
    red = np.array([255, 40, 40], dtype=np.float32)

    overlay[mask_original] = (
        0.55 * image_array[mask_original].astype(np.float32)
        + 0.45 * red
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay), mask_pil


def create_probability_image(probabilities, size):
    """Convert model probabilities into an 8-bit grayscale image."""

    probability_image = Image.fromarray(
        (probabilities * 255).clip(0, 255).astype(np.uint8)
    )

    return probability_image.resize(
        size,
        Image.Resampling.BILINEAR
    )


# ============================================================
# 4. SIDEBAR SETTINGS
# ============================================================
st.sidebar.header("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Flood detection threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.30,
    step=0.05,
    help=(
        "Lower thresholds generally identify more potential flood pixels. "
        "Higher thresholds produce more conservative predictions."
    )
)

st.sidebar.caption(f"Current threshold: {threshold:.2f}")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** U-Net")
st.sidebar.write("**Task:** Binary image segmentation")
st.sidebar.write("**Input resolution:** 256 × 256")

# ============================================================
# 5. IMAGE UPLOAD
# ============================================================
st.subheader("1. Upload a satellite image")

uploaded_file = st.file_uploader(
    "Choose an image in JPG or PNG format",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info(
        "Upload a satellite image to begin. For meaningful results, "
        "use imagery similar to the dataset used to train this model."
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption=f"Original image — {image.width} × {image.height} pixels",
            use_container_width=True
        )

        st.subheader("2. Run flood segmentation")

        if st.button(
            "🌊 Detect Flood Regions",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Running U-Net inference..."):
                probabilities, mask, flood_percentage = predict_flood(
                    image,
                    threshold
                )

                overlay, mask_image = create_overlay(image, mask)
                probability_image = create_probability_image(
                    probabilities,
                    image.size
                )

            st.success("Segmentation completed.")

            # ------------------------------------------------
            # 6. RESULTS
            # ------------------------------------------------
            st.subheader("3. Prediction results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    mask_image,
                    caption="Binary flood mask — white = predicted flood",
                    use_container_width=True
                )

            with col2:
                st.image(
                    overlay,
                    caption="Flood overlay — red = predicted flood",
                    use_container_width=True
                )

            st.subheader("4. Prediction summary")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Predicted flood pixels",
                f"{flood_percentage:.2f}%"
            )

            metric2.metric(
                "Detection threshold",
                f"{threshold:.2f}"
            )

            metric3.metric(
                "Model input",
                "256 × 256"
            )

            st.caption(
                "Flood-pixel percentage is calculated from the model's "
                "256 × 256 prediction. It is not a measurement of actual "
                "flooded land area."
            )

            # ------------------------------------------------
            # 7. PROBABILITY MAP
            # ------------------------------------------------
            with st.expander("View flood probability heatmap"):
                st.image(
                    probability_image,
                    caption=(
                        "Grayscale probability map: brighter pixels indicate "
                        "higher predicted flood probability."
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------
            # 8. DOWNLOADABLE OUTPUTS
            # ------------------------------------------------
            st.subheader("5. Download results")

            download_col1, download_col2, download_col3 = st.columns(3)

            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format="PNG")

            overlay_buffer = io.BytesIO()
            overlay.save(overlay_buffer, format="PNG")

            probability_buffer = io.BytesIO()
            probability_image.save(probability_buffer, format="PNG")

            with download_col1:
                st.download_button(
                    "⬇️ Download mask",
                    data=mask_buffer.getvalue(),
                    file_name="predicted_flood_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download_col2:
                st.download_button(
                    "⬇️ Download overlay",
                    data=overlay_buffer.getvalue(),
                    file_name="flood_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download_col3:
                st.download_button(
                    "⬇️ Download probability map",
                    data=probability_buffer.getvalue(),
                    file_name="flood_probability.png",
                    mime="image/png",
                    use_container_width=True
                )

            # ------------------------------------------------
            # 9. LIMITATIONS
            # ------------------------------------------------
            with st.expander("About this model and its limitations"):
                st.markdown(
                    """
                    - The model predicts pixel-level flood regions from an image.
                    - Prediction quality depends on the training data and image conditions.
                    - The selected threshold changes the precision–recall trade-off.
                    - The displayed percentage represents classified pixels, not
                      geographic flood extent.
                    - Predictions have not been validated for emergency or
                      operational use.
                    """
                )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
