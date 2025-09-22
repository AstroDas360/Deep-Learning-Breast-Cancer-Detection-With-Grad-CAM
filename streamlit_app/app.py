# app.py — Streamlit UI

import os, json
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np, cv2, streamlit as st, tensorflow as tf, keras

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from app_bundle_v5.preprocess import trim_borders
from app_bundle_v5.gradcam_utils import gradcam_overlay


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "app_bundle_v6"
MODEL_FILE = BUNDLE / "model.keras"
MODEL_DIR  = BUNDLE / "model_saved"
CFG_PATH   = BUNDLE / "config.json"

st.set_page_config(page_title="Mini-DDSM Classifier", layout="wide")

@st.cache_resource
def load_assets():
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
    else:
        cfg = {"img_size":[224,224], "last_conv":"block5_conv3",
               "class_names":["Non-malignant","Malignant"], "threshold":0.5}
    img_size  = tuple(cfg["img_size"])
    last_conv = cfg["last_conv"]
    classes   = cfg["class_names"]
    thr_def   = float(cfg["threshold"])

    can_cam, src = False, "None"
    if MODEL_FILE.exists():
        model = tf.keras.models.load_model(str(MODEL_FILE), compile=False)
        can_cam, src = True, ".keras"
    else:
        layer = keras.layers.TFSMLayer(str(MODEL_DIR), call_endpoint="serving_default")
        class Wrapper:
            def __init__(self, l): self.l = l
            def predict(self, x, verbose=0):
                out = self.l(x, training=False); out = list(out.values())[0] if isinstance(out, dict) else out
                return out.numpy()
        model, src = Wrapper(layer), "SavedModel"
    return model, can_cam, src, img_size, last_conv, classes, thr_def

def preprocess(bgr, img_size, gamma=1.2):
    # grayscale + trim
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = trim_borders(g)

    # letterbox
    W, H = int(img_size[0]), int(img_size[1])
    h, w = g.shape[:2]
    r = min(W / max(w, 1), H / max(h, 1))
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    rsz = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), dtype=rsz.dtype)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rsz

    # display image
    g01 = canvas.astype("float32") / 255.0
    disp = g01 if abs(gamma - 1.0) < 1e-6 else np.clip(g01 ** (1.0 / gamma), 0, 1)
    disp_rgb = (np.stack([disp, disp, disp], -1) * 255).astype("uint8")

    x = np.stack([g01, g01, g01], -1) * 255.0
    xb = vgg_preprocess(x[None, ...].astype("float32"))
    return xb, disp_rgb

def main():
    st.markdown("### Mini-DDSM Breast Image Classifier")
    model, can_cam, src, img_size, last_conv, classes, thr_def = load_assets()
    st.caption(f"Model: **{src}**")

    thr   = st.sidebar.slider("Threshold (malignant)", 0.0, 1.0, thr_def, 0.01)
    alpha = st.sidebar.slider("Grad-CAM alpha", 0.0, 1.0, 0.45, 0.05)
    gamma = st.sidebar.slider("Display gamma", 0.6, 2.0, 1.2, 0.1)

    up = st.file_uploader("Upload mammogram", type=["png","jpg","jpeg"])
    if not up: return

    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    xb, disp_rgb = preprocess(bgr, img_size, gamma)

    prob = float(model.predict(xb, verbose=0).ravel()[0])
    pred = classes[1] if prob >= thr else classes[0]

    c1, c2 = st.columns(2)
    c1.subheader("Input (trimmed & resized)")
    c1.image(disp_rgb, use_container_width=True)

    c2.subheader("Grad-CAM overlay")
    if can_cam and isinstance(model, tf.keras.Model):
        overlay, _ = gradcam_overlay(model, xb, disp_rgb, layer_name=last_conv, alpha=float(alpha))
        c2.image(overlay, use_container_width=True)
    else:
        c2.image(disp_rgb, caption="Grad-CAM not available (SavedModel)", use_container_width=True)

    st.markdown(f"**Prediction:** {pred}  \nP(Malignant) = `{prob:.3f}`  |  Threshold = `{thr:.2f}`")

if __name__ == "__main__":
    main()