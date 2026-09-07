import cv2
import time
import os
import sys
import argparse
from datetime import datetime

from src.utils import resolve_path, overlay_image_alpha, draw_hud_badge, dist
from src.face_recognition import FaceIdentitySystem
from src.face_expression import FaceExpressionDetector
from src.hand_tracking import HandGestureTracker

# Pastikan output konsol Windows mendukung UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DEFAULT_CAMERA_SOURCE = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Live HD Computer Vision: Face Recognition, Expressions & Memes")
    parser.add_argument("-s", "--source", type=str, default=str(DEFAULT_CAMERA_SOURCE), help="Sumber kamera (0, 1, 2 atau URL)")
    parser.add_argument("-i", "--ip", type=str, default=None, help="IP DroidCam WiFi (misal: 192.168.1.15)")
    parser.add_argument("-W", "--width", type=int, default=1920, help="Lebar frame HD (default: 1920)")
    parser.add_argument("-H", "--height", type=int, default=1080, help="Tinggi frame HD (default: 1080)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS kamera (default: 30)")
    return parser.parse_args()


def init_camera(args):
    cam_source = f"http://{args.ip.strip()}:4747/video" if args.ip else (int(args.source) if args.source.isdigit() else args.source)
    print("\n==================================================")
    print(f"  MENGHUBUNGKAN KE KAMERA: {cam_source}")
    print("==================================================")

    if isinstance(cam_source, int):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera {cam_source}!")
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[OK] Kamera HD Aktif: {actual_w}x{actual_h} @ {actual_fps} FPS\n")
    return cap


def main():
    args = parse_arguments()
    cap = init_camera(args)
    if cap is None:
        return

    # Inisialisasi Modul AI
    face_system = FaceIdentitySystem()
    expression_detector = FaceExpressionDetector()
    hand_tracker = HandGestureTracker()

    # Inisialisasi Gambar Meme Overlays
    cinema_logo_path = resolve_path(
        os.path.join("assets", "images", "cinema_logo.png"),
        os.path.join("assets", "cinema_logo.png"),
        "cinema_logo.png"
    )
    cinema_logo_img = cv2.imread(cinema_logo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(cinema_logo_path) else None

    manface_path = resolve_path(
        os.path.join("assets", "images", "meme_senyum.jpg"),
        os.path.join("assets", "images", "meme_senyum.png"),
        os.path.join("assets", "meme_senyum.jpg"),
        "meme_senyum.jpg"
    )
    manface_sticker_img = cv2.imread(manface_path) if os.path.exists(manface_path) else None

    window_name = "HD Vision: Face Recognition & Memes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    p_time = 0
    frame_count = 0
    detected_faces = []
    is_fullscreen = False
    is_mirrored = True

    print("==================================================")
    print("   HD VISION: IDENTITAS + EKSPRESI + MEME CINEMA  ")
    print("==================================================")
    print(" - Deteksi Ekspresi Wajah (Senyum -> Meme Roblox Man Face)")
    print(" - Pose 10 Jari (Angkat 2 Tangan) -> MEME ABSOLUTE CINEMA")
    print(" - Pose Jari Tengah -> 'fuck you'")
    print(" - 'm' : Toggle Mirror | 'r' : Reload DB | 's' : Screenshot | 'f' : Fullscreen | 'q' : Keluar\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[!] Frame tidak terbaca atau streaming terputus.")
            break

        if is_mirrored:
            img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_count += 1
        ui_scale = max(0.9, min(2.0, w / 1280.0))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Deteksi Identitas Wajah
        if frame_count % 2 == 0 or len(detected_faces) == 0:
            detected_faces = face_system.recognize_faces_in_frame(img)

        # 2. Deteksi Ekspresi Wajah
        face_expressions = expression_detector.process(img_rgb, w, h)

        # 3. Deteksi Pose Tangan & Gesture
        hand_data = hand_tracker.process(img, img_rgb, w, h, ui_scale)

        # 4. Gambar Kotak Identitas & Ekspresi Wajah
        verified_names = []
        current_expressions = []
        for face in detected_faces:
            fx1, fy1, fx2, fy2 = face["box"]
            name = face["name"]
            fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

            matching_expr = "Netral"
            expr_color = (0, 255, 120)
            if face_expressions:
                best_expr_match = min(face_expressions, key=lambda fe: dist((fcx, fcy), fe["center"]))
                matching_expr = best_expr_match["expression"]
                expr_color = best_expr_match["color"]
                current_expressions.append(matching_expr)

            is_known = name != "Tidak Dikenal"
            box_color = (0, 255, 0) if is_known else (0, 0, 255)
            if is_known:
                verified_names.append(name)

            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), box_color, max(1, int(2 * ui_scale)), cv2.LINE_AA)
            cv2.putText(img, f"ID: {name}", (fx1, max(int(22 * ui_scale), fy1 - int(8 * ui_scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * ui_scale, box_color, max(1, int(2 * ui_scale)), cv2.LINE_AA)
            cv2.putText(img, f"Ekspresi: {matching_expr}", (fx1, min(h - int(10 * ui_scale), fy2 + int(22 * ui_scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, expr_color, max(1, int(1.5 * ui_scale)), cv2.LINE_AA)

        # 5. Efek Stiker Roblox Man Face saat Senyum / Tertawa
        is_smiling_active = any(expr in ["Tersenyum", "Tertawa"] for expr in current_expressions)
        if is_smiling_active and manface_sticker_img is not None:
            card_w = int(160 * ui_scale)
            card_h = int(card_w * (manface_sticker_img.shape[0] / manface_sticker_img.shape[1]))
            c_x = w - card_w - int(20 * ui_scale)
            c_y = int(20 * ui_scale)
            cv2.rectangle(img, (c_x - 2, c_y - 2), (c_x + card_w + 2, c_y + card_h + 2), (255, 255, 255), 2, cv2.LINE_AA)
            overlay_image_alpha(img, manface_sticker_img, (c_x, c_y), (card_w, card_h))

        # 6. Efek Logo Absolute Cinema
        if hand_data["is_absolute_cinema"] and cinema_logo_img is not None:
            logo_w = int(w * 0.45)
            logo_h = int(logo_w * (cinema_logo_img.shape[0] / cinema_logo_img.shape[1]))
            logo_x = (w - logo_w) // 2
            logo_y = max(0, h - logo_h - int(15 * ui_scale))
            overlay_image_alpha(img, cinema_logo_img, (logo_x, logo_y), (logo_w, logo_h))

        # 7. Tampilan HUD & Status Ringkas
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        draw_hud_badge(img, fps, w, h, verified_names, detected_faces, current_expressions, hand_data["gesture_name"], ui_scale)
        cv2.imshow(window_name, img)

        # Event Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            is_mirrored = not is_mirrored
        elif key == ord('r'):
            face_system.load_known_identities()
        elif key == ord('s'):
            os.makedirs("outputs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("outputs", f"capture_{timestamp}.png")
            cv2.imwrite(save_path, img)
            print(f"[OK] Screenshot tersimpan: {save_path}")
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
