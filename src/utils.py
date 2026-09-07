import cv2
import math
import os
import numpy as np


def dist(p1, p2):
    """Menghitung jarak Euclidean 2D antara dua titik."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def resolve_path(*candidates):
    """Mencari path file/direktori pertama yang ditemukan dari daftar kandidat."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


def overlay_image_alpha(img, img_overlay, pos, size=None):
    """Menempelkan sticker/gambar overlay ke frame secara aman dengan transparansi dan kualitas tinggi (LANCZOS)."""
    x, y = int(pos[0]), int(pos[1])
    if size is not None:
        sw, sh = max(1, int(size[0])), max(1, int(size[1]))
        img_overlay = cv2.resize(img_overlay, (sw, sh), interpolation=cv2.INTER_LANCZOS4)

    h_ov, w_ov = img_overlay.shape[:2]
    h, w = img.shape[:2]

    if x >= w or y >= h or x + w_ov <= 0 or y + h_ov <= 0:
        return img

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_ov, w), min(y + h_ov, h)

    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    overlay_crop = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2]

    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        img[y1:y2, x1:x2] = (1.0 - alpha) * img[y1:y2, x1:x2] + alpha * overlay_crop[:, :, :3]
    else:
        img[y1:y2, x1:x2] = overlay_crop

    return img


def draw_hud_badge(img, fps, w, h, verified_names, detected_faces, current_expressions, gesture_name, ui_scale):
    """Menggambar kotak status HUD elegan dan ringkas di pojok kiri atas."""
    font_scale_sub = 0.45 * ui_scale
    badge_w = int(360 * ui_scale)
    badge_h = int(130 * ui_scale)

    hud_bg = img.copy()
    cv2.rectangle(hud_bg, (10, 10), (10 + badge_w, 10 + badge_h), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.50, img, 0.50, 0, img)
    cv2.rectangle(img, (10, 10), (10 + badge_w, 10 + badge_h), (80, 80, 80), 1, cv2.LINE_AA)

    pad_x = int(22 * ui_scale)
    cv2.putText(img, f"FPS: {int(fps)} | Resolusi: {w}x{h}", (pad_x, int(35 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 0, 255), 1, cv2.LINE_AA)

    if verified_names:
        id_status = f"ID: {', '.join(verified_names)}"
        id_color = (0, 255, 120)
    elif len(detected_faces) > 0:
        id_status = "ID: Tidak Dikenal"
        id_color = (0, 0, 255)
    else:
        id_status = "ID: Mencari Wajah..."
        id_color = (180, 180, 180)

    cv2.putText(img, id_status, (pad_x, int(60 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, id_color, 1, cv2.LINE_AA)

    if current_expressions:
        expr_hud_text = f"Ekspresi: {', '.join(current_expressions)}"
        expr_hud_color = (0, 255, 255)
    else:
        expr_hud_text = "Ekspresi: -"
        expr_hud_color = (160, 160, 160)

    cv2.putText(img, expr_hud_text, (pad_x, int(85 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, expr_hud_color, 1, cv2.LINE_AA)

    gesture_color = (0, 215, 255) if "CINEMA" in gesture_name else ((0, 0, 255) if "fuck" in gesture_name else (0, 255, 120))
    cv2.putText(img, f"Tangan: {gesture_name}", (pad_x, int(110 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, gesture_color, 1, cv2.LINE_AA)
