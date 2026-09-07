import cv2
import time
import os
import sys
import argparse
import mediapipe as mp

from src.utils import dist
from src.ar_math_drag import ARMathDragEngine

# Pastikan output konsol Windows mendukung UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DEFAULT_CAMERA_SOURCE = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="AR Math Game: Pinch & Drag AR Number Blocks")
    parser.add_argument("-s", "--source", type=str, default=str(DEFAULT_CAMERA_SOURCE), help="Sumber kamera (0, 1, 2 atau URL)")
    parser.add_argument("-i", "--ip", type=str, default=None, help="IP DroidCam WiFi (misal: 192.168.1.15)")
    parser.add_argument("-W", "--width", type=int, default=1920, help="Lebar frame HD (default: 1920)")
    parser.add_argument("-H", "--height", type=int, default=1080, help="Tinggi frame HD (default: 1080)")
    return parser.parse_args()


def init_camera(args):
    cam_source = f"http://{args.ip.strip()}:4747/video" if args.ip else (int(args.source) if args.source.isdigit() else args.source)
    print("\n==================================================")
    print(f"  MENGHUBUNGKAN KE KAMERA AR MATH: {cam_source}")
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
    print(f"[OK] Kamera AR Math HD Aktif: {actual_w}x{actual_h} @ 30 FPS\n")
    return cap


def main():
    args = parse_arguments()
    cap = init_camera(args)
    if cap is None:
        return

    # Inisialisasi Engine Game AR Math
    game = ARMathDragEngine()

    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hand_landmark_style = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=-1, circle_radius=4)
    hand_connection_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Cukup 1 tangan aktif untuk kontrol yang presisi
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    window_name = "AR Math Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    is_mirrored = True
    is_fullscreen = False

    print("==================================================")
    print("      🎯 AR MATH GAME: PINCH & DRAG BALOK         ")
    print("==================================================")
    print(" - Cubit (Pinch Jempol + Telunjuk) untuk memegang balok angka.")
    print(" - Geser (Drag) balok ke kotak slot yang bertanda '?'.")
    print(" - Lepas cubitan (Drop) untuk menaruh angka!")
    print(" - 'r' : Reset / Buat soal baru")
    print(" - 'm' : Toggle Mirror Kamera")
    print(" - 'f' : Fullscreen On / Off")
    print(" - 'q' atau 'ESC' : Keluar\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[!] Frame kamera terputus.")
            break

        if is_mirrored:
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        ui_scale = max(0.9, min(2.0, w / 1280.0))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Deteksi Tangan & Posisi Pinch
        hand_results = hands.process(img_rgb)
        pinch_pt = None
        is_pinching = False

        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_style,
                    connection_drawing_spec=hand_connection_style
                )

                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                # Titik tengah pinch & jarak antara jempol dan telunjuk
                pinch_pt = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                pinch_dist = dist(thumb_tip, index_tip)

                # Jika jarak < 42 pixel -> SEDANG PINCH / MEMEGANG
                if pinch_dist < 42 * ui_scale:
                    is_pinching = True

        # 2. Update Logika Pinch & Drag Balok AR
        game.update_hand_interaction(pinch_pt, is_pinching, w, h)

        # 3. Render Tampilan Visual Game AR
        game.draw_game_scene(img, pinch_pt, is_pinching, ui_scale)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            game.generate_new_stage()
            print("[*] Soal baru dimuat!")
        elif key == ord('m'):
            is_mirrored = not is_mirrored
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
