import streamlit as st
import torch
import cv2
import numpy as np
import os
from model import UNet

NUM_CLASSES = 23

st.set_page_config(page_title="CityScape Segmentation", layout="wide")
st.title("CityScape Image Segmentation")

page = st.sidebar.selectbox("Page", ["Training Plots", "Prediction"])

if page == "Training Plots":
    st.header("Training Metrics")

    if os.path.exists("plots/loss.png"):
        st.image("plots/loss.png", caption="Training Loss Curve")
    if os.path.exists("plots/miou.png"):
        st.image("plots/miou.png", caption="Training mIOU")
    if os.path.exists("plots/dice.png"):
        st.image("plots/dice.png", caption="Training mDice")

    st.header("Test Set Scores")
    if os.path.exists("test_scores.txt"):
        with open("test_scores.txt", "r") as f:
            lines = f.readlines()
            test_miou = float(lines[0].strip())
            test_mdice = float(lines[1].strip())
        st.metric("Test mIOU", f"{test_miou:.4f}")
        st.metric("Test mDice", f"{test_mdice:.4f}")

else:
    st.header("Upload Test Images for Prediction")

    model = UNet(NUM_CLASSES)
    model.load_state_dict(torch.load("saved_model.pth", map_location="cpu"))
    model.eval()

    files = st.file_uploader("Upload 4 images from test set", type=["png", "jpg"],
                             accept_multiple_files=True)

    if files:
        for file in files:
            raw = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_NEAREST)
            img_norm = img_resized.astype(np.float32) / 255.0

            x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
            pred = torch.argmax(out, dim=1).squeeze().numpy()

            gt_name = file.name
            gt_path = os.path.join("data/CameraMask", gt_name)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(img_resized, caption="Input Image")

            with col2:
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path)
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                    gt_mask = cv2.resize(gt_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                    gt_label = np.max(gt_mask, axis=-1)
                    gt_colored = (gt_label * 10).astype(np.uint8)
                    st.image(gt_colored, caption="Ground Truth Mask")
                else:
                    st.write("Ground truth not found for:", gt_name)

            with col3:
                pred_colored = (pred * 10).astype(np.uint8)
                st.image(pred_colored, caption="Predicted Mask")

            st.markdown("---")