import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

def assess_risk(box, img_width, img_height, category):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = img_width * img_height
    relative_size = box_area / frame_area

    moving_objects = ['car', 'truck', 'bus', 'pedestrian', 'rider', 'motorcycle', 'bicycle', 'train']
    if category not in moving_objects:
        return "N/A", relative_size

    if relative_size > 0.15:
        return "DANGER", relative_size
    elif relative_size > 0.05:
        return "WARNING", relative_size
    else:
        return "SAFE", relative_size

def process_image(img_array, conf_threshold=0.25):
    img_height, img_width = img_array.shape[:2]
    results = model(img_array, conf=conf_threshold, verbose=False)

    color_map = {
        "SAFE": (0, 200, 0), "WARNING": (0, 200, 255),
        "DANGER": (0, 0, 255), "N/A": (200, 200, 200)
    }
    risk_counts = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
    class_counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        risk, rel_size = assess_risk(xyxy, img_width, img_height, cls_name)
        if risk in risk_counts:
            risk_counts[risk] += 1

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = color_map[risk]
        label = f"{cls_name} {conf:.2f} {risk}" if risk != "N/A" else f"{cls_name} {conf:.2f}"
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_array, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_array, risk_counts, class_counts

st.set_page_config(page_title="Real-Time Object Detection - Autonomous Driving", layout="wide")

# ===== Dashboard-style theme (mobile-responsive) =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {
    background: linear-gradient(180deg, #0b0f14 0%, #10161d 100%);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.dash-header {
    padding: 4px 0 18px 0;
    border-bottom: 1px solid #232b34;
    margin-bottom: 20px;
}
.dash-title {
    font-size: clamp(1.4rem, 5vw, 2.1rem);
    font-weight: 700;
    color: #eef2f5;
    margin: 0;
    letter-spacing: -0.5px;
}
.dash-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #5fe0c7;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #131a21;
    border: 1px solid #232b34;
    border-radius: 10px;
    padding: 14px 10px 10px 10px;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
}

/* Slider + uploader containers */
[data-testid="stFileUploaderDropzone"] {
    background: #131a21;
    border: 1px dashed #2f3a44;
    border-radius: 10px;
}

/* Footer / signature plate */
.sig-plate {
    margin-top: 40px;
    padding-top: 14px;
    border-top: 1px solid #232b34;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #5c6773;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 6px;
}
.sig-plate .tag {
    color: #5fe0c7;
    letter-spacing: 1px;
}

/* Mobile tweaks */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
    }
    [data-testid="stMetric"] {
        padding: 10px 6px 8px 6px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    .sig-plate {
        flex-direction: column;
        text-align: left;
    }
}
</style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown("""
<div class="dash-header">
    <p class="dash-title">🚗 Real-Time Object Detection</p>
    <p class="dash-sub">Autonomous Driving &middot; YOLOv8 &middot; BDD100K Fine-Tuned &middot; Proximity Risk Overlay</p>
</div>
""", unsafe_allow_html=True)

conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
uploaded_file = st.file_uploader("Upload a driving scene", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection..."):
        result_img, risk_counts, class_counts = process_image(img_array_bgr, conf_threshold)

    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="Detection + Risk Overlay", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 SAFE", risk_counts["SAFE"])
    col2.metric("🟡 WARNING", risk_counts["WARNING"])
    col3.metric("🔴 DANGER", risk_counts["DANGER"])

    st.write("**Detected classes in this image:**", class_counts)
else:
    st.info("Upload a driving image above to start detection")

# ===== Signature footer =====
st.markdown("""
<div class="sig-plate">
    <span><span class="tag">PROJECT</span> &nbsp; Real-Time Object Detection for Autonomous Driving</span>
    <span><span class="tag">AUTHOR</span> &nbsp; Muhammad Shahbaz Khan &middot; Matricola 6669708</span>
</div>
""", unsafe_allow_html=True)
