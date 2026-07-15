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

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp { background: #f3f2ee; }

.block-container {
    max-width: 760px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-radius: 14px;
}

.app-title {
    font-size: clamp(1.1rem, 4vw, 1.5rem);
    font-weight: 700;
    color: #24231f;
    margin: 0;
}
.app-sub {
    font-size: 0.85rem;
    color: #6b6a63;
    margin-top: 4px;
    margin-bottom: 4px;
}

.risk-row { display: flex; gap: 10px; margin: 6px 0 18px 0; }
.risk-card { flex: 1; border-radius: 10px; padding: 12px 12px 10px 12px; }
.risk-card.safe { background: #eaf3de; }
.risk-card.safe .risk-label { color: #3b6d11; }
.risk-card.safe .risk-value { color: #27500a; }
.risk-card.warn { background: #faeeda; }
.risk-card.warn .risk-label { color: #854f0b; }
.risk-card.warn .risk-value { color: #633806; }
.risk-card.danger { background: #fcebeb; }
.risk-card.danger .risk-label { color: #a32d2d; }
.risk-card.danger .risk-value { color: #791f1f; }
.risk-label { font-size: 12.5px; margin: 0; }
.risk-value { font-size: 22px; font-weight: 700; margin: 2px 0 0; }

@media (max-width: 640px) {
    .risk-row { flex-direction: column; gap: 8px; }
    .risk-value { font-size: 18px; }
}

[data-testid="stFileUploaderDropzone"] {
    background: #f7f6f2;
    border: 1.5px dashed #c9c6ba;
    border-radius: 10px;
}

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 4px 0; }
.chip {
    background: #f7f6f2;
    border: 1px solid #e3e1d9;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12.5px;
    color: #24231f;
    white-space: nowrap;
}

.sig-footer {
    margin-top: 20px;
    padding-top: 12px;
    border-top: 1px solid #e3e1d9;
    font-size: 0.75rem;
    color: #9c9a90;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 4px;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.2rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        min-width: 100% !important;
    }
    [data-testid="stMetric"] { padding: 8px 6px 6px 6px; margin-bottom: 6px; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    .sig-footer { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("""
    <p class="app-title">🚗 Real-time object detection</p>
    <p class="app-sub">Autonomous driving &middot; YOLOv8 fine-tuned on BDD100K &middot; proximity risk overlay</p>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    st.caption("Only show detections the model is at least this confident about. Lower to catch more objects, at the risk of false alarms.")

    uploaded_file = st.file_uploader("Upload a driving scene", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            result_img, risk_counts, class_counts = process_image(img_array_bgr, conf_threshold)

        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img_rgb, caption="Detection + risk overlay", use_container_width=True)

        st.markdown(f"""
        <div class="risk-row">
            <div class="risk-card safe"><p class="risk-label">Safe</p><p class="risk-value">{risk_counts['SAFE']}</p></div>
            <div class="risk-card warn"><p class="risk-label">Warning</p><p class="risk-value">{risk_counts['WARNING']}</p></div>
            <div class="risk-card danger"><p class="risk-label">Danger</p><p class="risk-value">{risk_counts['DANGER']}</p></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Detected in this image**")
        if class_counts:
            chips_html = "".join(
                f'<span class="chip">{name.replace("_", " ").capitalize()} &middot; {count}</span>'
                for name, count in sorted(class_counts.items(), key=lambda x: -x[1])
            )
            st.markdown(f'<div class="chip-row">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No objects detected at this confidence threshold.")
    else:
        st.info("Upload a driving image above to start detection.")

    st.markdown("""
    <div class="sig-footer">
        <span>Real-time object detection for autonomous driving</span>
        <span>Muhammad Shahbaz Khan &middot; Matricola 6669708</span>
    </div>
    """, unsafe_allow_html=True)
