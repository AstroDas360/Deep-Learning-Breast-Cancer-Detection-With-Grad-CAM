import numpy as np
import cv2
import tensorflow as tf

def gradcam_overlay(model, x_bhwc, disp_rgb, layer_name="block5_conv3", alpha=0.45, floor=0.02):
    # Build grad-model (simple & explicit)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, preds = grad_model(x_bhwc, training=False)  # conv: (1,h,w,C)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv)
    w     = tf.reduce_mean(grads, axis=(1, 2))[0]                   # (C,)
    cam   = tf.reduce_sum(conv[0] * w[None, None, :], axis=-1).numpy()
    cam   = np.maximum(cam, 0)                                       # ReLU

    # resize -> normalize -> floor for dark background
    H, W = int(x_bhwc.shape[1]), int(x_bhwc.shape[2])
    cam  = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam  = (cam - cam.min()) / (cam.max() + 1e-8)
    if floor and floor > 0: cam[cam < float(floor)] = 0.0

    hm   = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
    if disp_rgb.ndim == 2:
        disp_rgb = cv2.cvtColor(disp_rgb, cv2.COLOR_GRAY2RGB)
    if disp_rgb.dtype != np.uint8:
        disp_rgb = np.clip(disp_rgb, 0, 255).astype(np.uint8)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(hm, float(alpha), disp_rgb, float(1.0 - alpha), 0.0)
    return overlay, float(preds.numpy().ravel()[0])