import cv2
import time
import os
import sys
import argparse
from datetime import datetime

from src.hand_tracking import HandGestureTracker
from src.math_game import MathGameEngine

# Pastikan output konsol Windows mendukung UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DEFAULT_CAMERA_SOURCE = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Math Vision: Hand Gesture Math Quiz Game")
    parser.add_argument("-s", "--source", type=str, default=str(DEFAULT_CAMERA_SOURCE), help="Sumber kamera (0, 1, 2 atau URL)")
    parser.add_argument("-i", "--ip", type=str, default=None, help="IP DroidCam WiFi (misal: 192.168.1.15)")
    parser.add_argument("-W", "--width", type=int, default=1920, help="Lebar frame HD (default: 1920)")
    parser.add_argument("-H", "--height", type=int, default=1080, help="Tinggi frame HD (default: 1080)")
    parser.add_argument("--time", type=float, default=6.0, help="Batas waktu per soal dalam detik (default: 6.0)")
    return parser.parse_args()


def init_camera(args):
    cam_source = f"http://{args.ip.strip()}:4747/video" if args.ip else (int(args.source) if args.source.isdigit() else args.source)
    print("\n==================================================")
    print(f"  MENGHUBUNGKAN KE KAMERA GAME: {cam_source}")
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
    print(f"[OK] Kamera Game HD Aktif: {actual_w}x{actual_h} @ 30 FPS\n")
    return cap


def main():
    args = parse_arguments()
    cap = init_camera(args)
    if cap is None:
        return

    # Inisialisasi Hand Tracker & Math Game Engine
    hand_tracker = HandGestureTracker(max_num_hands=2)
    game = MathGameEngine(time_limit_per_question=args.time)

    window_name = "AI Math Vision: Hand Gesture Quiz Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    is_mirrored = True
    is_fullscreen = False

    print("==================================================")
    print("   🎮 AI MATH VISION: HAND GESTURE QUIZ GAME     ")
    print("==================================================")
    print(" - Jawab soal matematika dengan mengangkat jari (0-10)!")
    print(" - Tahan posisi jari 1 detik untuk mengunci jawaban.")
    print(" - Tekan 'SPACE' : Mulai / Restart Game")
    print(" - Tekan 'm' : Toggle Mirror Kamera")
    print(" - Tekan 'f' : Fullscreen On / Off")
    print(" - Tekan 'q' atau 'ESC' : Keluar dari Game\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[!] Frame kamera terputus.")
            break

        if is_mirrored:
            img = cv2.flip(img, 1)

        h, w, _ = img.shape
        ui_scale = max(0.9, min(2.0, w / 1280.0))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Deteksi Tangan & Hitung Jari Pemain
        hand_data = hand_tracker.process(img, img_rgb, w, h, ui_scale)
        detected_fingers = hand_data["finger_count_total"] if hand_data["hands_count"] > 0 else -1

        # 2. Update Game State
        game.update(detected_fingers)

        # 3. Render Game UI
        game.draw_game_ui(img, detected_fingers, ui_scale)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("[*] Keluar dari game...")
            break
        elif key in (ord(' '), ord('r')):
            game.start_game()
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
