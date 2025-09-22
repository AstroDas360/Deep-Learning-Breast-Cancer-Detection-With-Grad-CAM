import cv2, numpy as np

def trim_borders(img, frac=0.05, pad_ratio=0.015, close_kernel=7):
    # Ensure grayscale uint8
    if img is None:
        return img
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    dy = max(1, int(h * frac))
    dx = max(1, int(w * frac))
    y0, y1 = dy, max(dy + 1, h - dy)
    x0, x1 = dx, max(dx + 1, w - dx)
    img = img[y0:y1, x0:x1]

    g = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decide foreground polarity (breast should be "white")
    if (th > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8), 1
    )

    # **Key line**: make absolutely sure it's 8-bit single-channel for findContours
    th = (th > 0).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img

    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    pad = int(pad_ratio * max(bw, bh))
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + bw + pad);  y1 = min(img.shape[0], y + bh + pad)

    if (x1 - x0) > 20 and (y1 - y0) > 20:
        return img[y0:y1, x0:x1]
    return img

def preprocess_path(path, img_size):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        return np.zeros((1, img_size[1], img_size[0], 3), dtype="float32")
    g = trim_borders(g)
    g = cv2.resize(g, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)
    g01 = g.astype("float32") / 255.0
    x = np.stack([g01, g01, g01], axis=-1)
    m, s = x.mean(), x.std() + 1e-7
    x = (x - m) / s
    return x[None, ...]  # (1, H, W, 3)
