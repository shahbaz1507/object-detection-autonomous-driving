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
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
 
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-radius: 14px;
}
 
.header-block { text-align: center; margin-bottom: 6px; }
.app-title-row { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 4px; }
.app-title-icon { font-size: 24px; }
.app-title-text { font-size: 20px; font-weight: 700; margin: 0; color: #24231f; }
.app-sub { font-size: 12.5px; color: #6b6a63; margin: 0 0 10px 0; text-align: center; }
 
.section-title { font-size: 12.5px; font-weight: 600; margin: 0 0 8px; color: #24231f; }
 
.helper-text-custom { font-size: 11px; color: #9c9a90; margin: 2px 0 12px; line-height: 1.4; }
 
[data-testid="stFileUploaderDropzone"] {
    background: #f7f6f2;
    border: 1.5px dashed #c9c6ba;
    border-radius: 10px;
}
 
.risk-row { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.risk-card { border-radius: 10px; padding: 10px 12px; display: flex; align-items: center; justify-content: space-between; }
.risk-card.safe { background: #eaf3de; }
.risk-card.warn { background: #faeeda; }
.risk-card.danger { background: #fcebeb; }
.risk-label { font-size: 12.5px; margin: 0; }
.risk-card.safe .risk-label { color: #3b6d11; }
.risk-card.warn .risk-label { color: #854f0b; }
.risk-card.danger .risk-label { color: #a32d2d; }
.risk-value { font-size: 18px; font-weight: 700; margin: 0; }
.risk-card.safe .risk-value { color: #27500a; }
.risk-card.warn .risk-value { color: #633806; }
.risk-card.danger .risk-value { color: #791f1f; }
 
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.chip { background: #f7f6f2; border: 1px solid #e3e1d9; border-radius: 999px; padding: 5px 12px; font-size: 12px; color: #24231f; white-space: nowrap; }
 
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
        padding-top: 3.5rem !important;
    }
    .app-title-text { font-size: 16px; }
    .app-sub { font-size: 10.5px; }
    .risk-value { font-size: 15px; }
    .sig-footer { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)
 
with st.container(border=True):
    st.markdown("""
    <div class="header-block">
        <div class="app-title-row">
            <span class="app-title-icon">🚗</span>
            <p class="app-title-text">Real-time object detection</p>
        </div>
        <p class="app-sub">Autonomous driving &middot; YOLOv8 fine-tuned on BDD100K &middot; proximity risk overlay</p>
    </div>
    """, unsafe_allow_html=True)
 
    col_left, col_right = st.columns(2)
 
    with col_left:
        conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
        st.markdown('<p class="helper-text-custom">Only show detections the model is at least this confident about.</p>', unsafe_allow_html=True)
 
        uploaded_file = st.file_uploader("Upload a driving scene", type=["jpg", "jpeg", "png"])
 
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
 
            with st.spinner("Running detection..."):
                result_img, risk_counts, class_counts = process_image(img_array_bgr, conf_threshold)
 
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(result_img_rgb)
            preview.thumbnail((360, 240))
            st.image(preview, caption="Detection + risk overlay")
        else:
            risk_counts = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
            class_counts = {}
            st.info("Upload a driving image to start detection.")
 
    with col_right:
        st.markdown('<p class="section-title">Risk summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-row">
            <div class="risk-card safe"><span class="risk-label">Safe</span><span class="risk-value">{risk_counts['SAFE']}</span></div>
            <div class="risk-card warn"><span class="risk-label">Warning</span><span class="risk-value">{risk_counts['WARNING']}</span></div>
            <div class="risk-card danger"><span class="risk-label">Danger</span><span class="risk-value">{risk_counts['DANGER']}</span></div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<p class="section-title">Detected in this image</p>', unsafe_allow_html=True)
        if class_counts:
            chips_html = "".join(
                f'<span class="chip">{name.replace("_", " ").capitalize()} &middot; {count}</span>'
                for name, count in sorted(class_counts.items(), key=lambda x: -x[1])
            )
            st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No objects detected yet.")
 
    st.markdown("""
    <div class="sig-footer">
        <span>Real-time object detection for autonomous driving</span>
        <span>Muhammad Shahbaz Khan &middot; Matricola 6669708</span>
    </div>
    """, unsafe_allow_html=True)
