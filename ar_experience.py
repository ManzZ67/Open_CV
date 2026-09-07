import cv2
import time
import os
import sys
import argparse
from datetime import datetime

import mediapipe as mp
from src.hand_tracking import get_fingers_state
from src.ar_filters import ARAirCanvas, ARFaceFilter, AR3DHologramCube

# Pastikan output konsol Windows mendukung UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DEFAULT_CAMERA_SOURCE = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Augmented Reality (AR) Studio: Air Canvas, Face Filters & 3D Hologram")
    parser.add_argument("-s", "--source", type=str, default=str(DEFAULT_CAMERA_SOURCE), help="Sumber kamera (0, 1, 2 atau URL)")
    parser.add_argument("-i", "--ip", type=str, default=None, help="IP DroidCam WiFi (misal: 192.168.1.15)")
    parser.add_argument("-W", "--width", type=int, default=1920, help="Lebar frame HD (default: 1920)")
    parser.add_argument("-H", "--height", type=int, default=1080, help="Tinggi frame HD (default: 1080)")
    return parser.parse_args()


def init_camera(args):
    cam_source = f"http://{args.ip.strip()}:4747/video" if args.ip else (int(args.source) if args.source.isdigit() else args.source)
    print("\n==================================================")
    print(f"  MENGHUBUNGKAN KE KAMERA AR: {cam_source}")
    print("==================================================")

    if isinstance(cam_source, int):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera {cam_source}!")
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Kamera AR HD Aktif: {actual_w}x{actual_h} @ 30 FPS\n")
    return cap


def main():
    args = parse_arguments()
    cap = init_camera(args)
    if cap is None:
        return

    # Inisialisasi Modul AR
    air_canvas = ARAirCanvas()
    face_filter = ARFaceFilter()
    holo_cube = AR3DHologramCube()

    # Inisialisasi MediaPipe Hands & Face Mesh
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    window_name = "Augmented Reality (AR) Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Modes: "CANVAS", "FACE_FILTER", "3D_CUBE"
    current_ar_mode = "CANVAS"
    is_mirrored = True
    is_fullscreen = False
    p_time = time.time()

    print("==================================================")
    print("      ✨ AUGMENTED REALITY (AR) STUDIO            ")
    print("==================================================")
    print(" [Mode AR]:")
    print(" - Tekan '1' : Mode AR Air Canvas (Lukis di Udara dengan Jari)")
    print(" - Tekan '2' : Mode AR Face Filters (Kacamata Cyber / Mahkota)")
    print(" - Tekan '3' : Mode AR 3D Hologram Cube di Telapak Tangan")
    print("\n [Kontrol Tambahan]:")
    print(" - Tekan 'c' : Bersihkan Gambar Canvas (Mode 1)")
    print(" - Tekan 't' : Ganti Aksesoris Wajah (Mode 2)")
    print(" - Tekan 's' : Simpan Screenshot Hasil AR ke outputs/")
    print(" - Tekan 'm' : Toggle Mirror Kamera")
    print(" - Tekan 'f' : Fullscreen On / Off")
    print(" - Tekan 'q' atau 'ESC' : Keluar\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[!] Frame kamera terputus.")
            break

        if is_mirrored:
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        ui_scale = max(0.9, min(2.0, w / 1280.0))
        c_time = time.time()
        dt = c_time - p_time
        p_time = c_time

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ======================================================================
        # MODE 1: AR AIR CANVAS (MELUKIS DI UDARA)
        # ======================================================================
        if current_ar_mode == "CANVAS":
            air_canvas.draw_palette(img, ui_scale)
            hand_results = hands.process(img_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                    fingers = get_fingers_state(lm_list)
                    air_canvas.process_hand(img, lm_list, fingers, ui_scale)
            img = air_canvas.render_canvas(img)

        # ======================================================================
        # MODE 2: AR FACE FILTERS (CYBER VISOR & GOLDEN CROWN)
        # ======================================================================
        elif current_ar_mode == "FACE_FILTER":
            face_results = face_mesh.process(img_rgb)
            if face_results.multi_face_landmarks:
                for face_lms in face_results.multi_face_landmarks:
                    if face_filter.filter_mode == "CYBER_VISOR":
                        face_filter.draw_cyber_visor(img, face_lms.landmark, w, h, ui_scale)
                    elif face_filter.filter_mode == "GOLDEN_CROWN":
                        face_filter.draw_golden_crown(img, face_lms.landmark, w, h, ui_scale)

        # ======================================================================
        # MODE 3: AR 3D HOLOGRAM CUBE FLOATING ON PALM
        # ======================================================================
        elif current_ar_mode == "3D_CUBE":
            hand_results = hands.process(img_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                    wrist, idx_mcp, pinky_mcp = lm_list[0], lm_list[5], lm_list[17]
                    palm_cx = (wrist[0] + idx_mcp[0] + pinky_mcp[0]) // 3
                    palm_cy = (wrist[1] + idx_mcp[1] + pinky_mcp[1]) // 3
                    holo_cube.draw_cube_on_palm(img, (palm_cx, palm_cy), dt=dt, ui_scale=ui_scale)

        # Bottom-Left AR Info Badge
        badge_w, badge_h = int(320 * ui_scale), int(70 * ui_scale)
        cv2.rectangle(img, (10, h - badge_h - 10), (10 + badge_w, h - 10), (20, 20, 20), -1)
        cv2.rectangle(img, (10, h - badge_h - 10), (10 + badge_w, h - 10), (0, 255, 255), 1, cv2.LINE_AA)
        
        mode_label = {
            "CANVAS": "Mode: AR Air Canvas (Lukis Udara)",
            "FACE_FILTER": f"Mode: AR Filter ({face_filter.filter_mode})",
            "3D_CUBE": "Mode: AR 3D Hologram Cube"
        }.get(current_ar_mode, "AR Mode")

        cv2.putText(img, mode_label, (20, h - int(45 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui_scale, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Tekan '1', '2', '3' untuk Ganti Mode", (20, h - int(22 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui_scale, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            current_ar_mode = "CANVAS"
            print("[*] Beralih ke Mode: AR Air Canvas (Lukis di Udara)")
        elif key == ord('2'):
            current_ar_mode = "FACE_FILTER"
            print("[*] Beralih ke Mode: AR Face Filter Aksesoris")
        elif key == ord('3'):
            current_ar_mode = "3D_CUBE"
            print("[*] Beralih ke Mode: AR 3D Hologram Cube")
        elif key == ord('t') and current_ar_mode == "FACE_FILTER":
            face_filter.filter_mode = "GOLDEN_CROWN" if face_filter.filter_mode == "CYBER_VISOR" else "CYBER_VISOR"
            print(f"[*] Filter Wajah: {face_filter.filter_mode}")
        elif key == ord('c') and current_ar_mode == "CANVAS":
            air_canvas.clear()
            print("[*] Canvas dibersihkan!")
        elif key == ord('m'):
            is_mirrored = not is_mirrored
        elif key == ord('s'):
            os.makedirs("outputs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("outputs", f"ar_snapshot_{timestamp}.png")
            cv2.imwrite(save_path, img)
            print(f"[OK] Foto AR tersimpan di: {save_path}")
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
