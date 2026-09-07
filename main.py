import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import sys
import argparse
import torch
from facenet_pytorch import InceptionResnetV1

# Pastikan output konsol Windows mendukung UTF-8 tanpa crash
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==============================================================================
# KONFIGURASI SUMBER KAMERA (WEBCAM LAPTOP / DROIDCAM)
# ==============================================================================
# - Default: 0 (Kamera bawaan Laptop)
# - DroidCam Client PC: 1 (atau 2)
# - DroidCam WiFi / IP: "http://<IP_HP>:4747/video" (misal: python main.py -i 192.168.1.15)
DEFAULT_CAMERA_SOURCE = 0  # 0 = Kamera Laptop, 1/2 = DroidCam Client, atau URL IP



def dist(p1, p2):
    """Menghitung jarak Euclidean 2D antara dua titik."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_fingers_state(lm_list):
    """
    Deteksi status 5 jari terbuka (1) atau tertutup (0):
    Index: [0: Jempol, 1: Telunjuk, 2: Tengah, 3: Manis, 4: Kelingking]
    """
    fingers = [0, 0, 0, 0, 0]
    wrist = lm_list[0]
    pinky_mcp = lm_list[17]

    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    mcp_ids = [2, 5, 9, 13, 17]

    for i in range(1, 5):
        tip = lm_list[tip_ids[i]]
        pip = lm_list[pip_ids[i]]
        mcp = lm_list[mcp_ids[i]]

        if (tip[1] < pip[1] and tip[1] < mcp[1]) or (dist(tip, wrist) > dist(pip, wrist) * 1.08):
            fingers[i] = 1
        else:
            fingers[i] = 0

    thumb_tip = lm_list[4]
    thumb_mcp = lm_list[2]

    d_thumb_tip_to_pinky_base = dist(thumb_tip, pinky_mcp)
    d_thumb_base_to_pinky_base = dist(thumb_mcp, pinky_mcp)

    if d_thumb_tip_to_pinky_base > (d_thumb_base_to_pinky_base * 1.02):
        fingers[0] = 1
    else:
        fingers[0] = 0

    return fingers

def overlay_image_alpha(img, img_overlay, pos, size=None):
    """Menempelkan sticker/gambar overlay ke frame secara aman dengan transparansi."""
    x, y = int(pos[0]), int(pos[1])
    if size is not None:
        sw, sh = max(1, int(size[0])), max(1, int(size[1]))
        img_overlay = cv2.resize(img_overlay, (sw, sh))

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

# --- SISTEM REKOGNISI IDENTITAS WAJAH (FACENET + MEDIAPIPE) ---
class FaceIdentitySystem:
    def __init__(self, identities_dir="identitas"):
        self.identities_dir = identities_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Menginisialisasi Model FaceNet pada: {self.device}")
        
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        self.known_embeddings = {}
        self.load_known_identities()

    def extract_embedding_from_crop(self, face_rgb):
        if face_rgb.size == 0:
            return None
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        with torch.no_grad():
            emb = self.resnet(face_tensor.unsqueeze(0).to(self.device))
            emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        return emb.cpu().numpy()[0]

    def load_known_identities(self):
        self.known_embeddings.clear()
        if not os.path.exists(self.identities_dir):
            os.makedirs(self.identities_dir, exist_ok=True)
            return

        print(f"\n==========================================")
        print(f"  MEMUAT DATABASE IDENTITAS DARI '{self.identities_dir}'")
        print(f"==========================================")
        
        for root, dirs, files in os.walk(self.identities_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(root, file)
                    rel_dir = os.path.relpath(root, self.identities_dir)
                    person_name = rel_dir if rel_dir != "." else os.path.splitext(file)[0]
                    
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    scale = 1000 / max(h, w) if max(h, w) > 1000 else 1.0
                    img_work = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
                    wh, ww = img_work.shape[:2]

                    rgb = cv2.cvtColor(img_work, cv2.COLOR_BGR2RGB)
                    res = self.mp_face.process(rgb)

                    if res.detections:
                        best_det = max(res.detections, key=lambda d: d.score[0])
                        bbox = best_det.location_data.relative_bounding_box
                        x1 = max(0, int(bbox.xmin * ww))
                        y1 = max(0, int(bbox.ymin * wh))
                        x2 = min(ww, int((bbox.xmin + bbox.width) * ww))
                        y2 = min(wh, int((bbox.ymin + bbox.height) * wh))

                        face_crop = rgb[y1:y2, x1:x2]
                        emb = self.extract_embedding_from_crop(face_crop)
                        if emb is not None:
                            if person_name not in self.known_embeddings:
                                self.known_embeddings[person_name] = []
                            self.known_embeddings[person_name].append(emb)
                            print(f" [OK] Terdaftar: '{person_name}' (File: {file})")

        print(f"Total Identitas Terdaftar: {len(self.known_embeddings)} orang\n")

    def recognize_faces_in_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(frame_rgb)
        
        detected_faces = []
        if not res.detections:
            return detected_faces

        for det in res.detections:
            score = det.score[0]
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            mx = int((x2 - x1) * 0.1)
            my = int((y2 - y1) * 0.1)
            cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
            cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

            face_crop = frame_rgb[cy1:cy2, cx1:cx2]
            emb = self.extract_embedding_from_crop(face_crop)

            best_match_name = "Tidak Dikenal"
            best_similarity = 0.0

            if emb is not None and self.known_embeddings:
                for name, emb_list in self.known_embeddings.items():
                    for known_emb in emb_list:
                        sim = float(np.dot(emb, known_emb))
                        if sim > best_similarity:
                            best_similarity = sim
                            if sim >= 0.62:
                                best_match_name = name

            detected_faces.append({
                "box": (x1, y1, x2, y2),
                "name": best_match_name,
                "similarity": best_similarity,
                "score": score
            })

        return detected_faces

def get_facial_expression(landmarks, w, h):
    """
    Analisis ekspresi wajah presisi & akurat berdasarkan MediaPipe FaceMesh:
    - Tersenyum (Senyum Nyata / Smirk Nyata) -> Memunculkan Sticker Roblox Man Face
    - Tertawa (Mulut Terbuka + Senyum)
    - Cemberut
    - Kaget / Mulut Terbuka
    - Mengedip Kiri / Kanan
    - Mata Terpejam
    - Netral
    """
    def get_pt(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    # 1. Skala Dimensi Wajah
    face_height = max(1.0, dist(get_pt(10), get_pt(152)))
    face_width = max(1.0, dist(get_pt(234), get_pt(454)))

    # 2. Bibir & Mulut
    p_lip_top = get_pt(13)
    p_lip_bottom = get_pt(14)
    p_corner_r = get_pt(61)
    p_corner_l = get_pt(291)

    mouth_height = dist(p_lip_top, p_lip_bottom)
    mouth_width = dist(p_corner_r, p_corner_l)
    
    mouth_open_ratio = mouth_height / face_height
    mouth_width_ratio = mouth_width / face_width

    # Elevasi Sudut Bibir dihitung dari garis tengah bibir
    lip_center_y = (p_lip_top[1] + p_lip_bottom[1]) / 2.0
    corner_lift_r = (lip_center_y - p_corner_r[1]) / face_height
    corner_lift_l = (lip_center_y - p_corner_l[1]) / face_height
    avg_corner_lift = (corner_lift_r + corner_lift_l) / 2.0
    max_corner_lift = max(corner_lift_r, corner_lift_l)
    min_corner_lift = min(corner_lift_r, corner_lift_l)

    # 3. Mata (Eye Aspect Ratio - EAR)
    r_eye_h = dist(get_pt(159), get_pt(145))
    r_eye_w = max(1.0, dist(get_pt(133), get_pt(33)))
    right_ear = r_eye_h / r_eye_w

    l_eye_h = dist(get_pt(386), get_pt(374))
    l_eye_w = max(1.0, dist(get_pt(362), get_pt(263)))
    left_ear = l_eye_h / l_eye_w

    # 4. Alis
    r_brow_eye = dist(get_pt(66), get_pt(159)) / face_height
    l_brow_eye = dist(get_pt(296), get_pt(386)) / face_height
    avg_brow_lift = (r_brow_eye + l_brow_eye) / 2.0

    # --- KLASIFIKASI EKSPRESI ---
    # 1. Kaget / Mulut Terbuka
    if mouth_open_ratio > 0.16:
        if mouth_open_ratio > 0.22 or avg_brow_lift > 0.17:
            return "Kaget", (0, 200, 255)
        else:
            return "Mulut Terbuka", (0, 255, 255)

    # 2. Tertawa (Mulut terbuka + senyum lebar)
    if mouth_open_ratio > 0.08 and (mouth_width_ratio > 0.48 or avg_corner_lift > 0.022):
        return "Tertawa", (0, 255, 120)

    # 3. Tersenyum (Senyum 2 Sisi atau Smirk 1 Sisi Nyata)
    is_two_side_smile = (avg_corner_lift > 0.020 and mouth_width_ratio > 0.44) or (mouth_width_ratio > 0.49 and avg_corner_lift > 0.008)
    is_smirk = (max_corner_lift > 0.035 and (max_corner_lift - min_corner_lift) > 0.025)

    if is_two_side_smile or is_smirk:
        return "Tersenyum", (0, 255, 0)

    # 4. Cemberut (Sudut bibir melengkung jelas ke bawah)
    if avg_corner_lift < -0.022 and mouth_open_ratio < 0.07:
        return "Cemberut", (0, 100, 255)

    # 5. Mata Terpejam
    if left_ear < 0.12 and right_ear < 0.12:
        return "Mata Terpejam", (200, 200, 200)

    # 6. Mengedip
    if left_ear < 0.12 and right_ear >= 0.18:
        return "Mengedip Kiri", (255, 200, 0)
    if right_ear < 0.12 and left_ear >= 0.18:
        return "Mengedip Kanan", (255, 200, 0)

    # Default: Netral
    return "Netral", (0, 255, 120)

def main():
    # Inisialisasi Argumen CLI (DroidCam / Webcam)
    parser = argparse.ArgumentParser(description="OpenCV Face & Gesture Detection dengan Dukungan DroidCam")
    parser.add_argument(
        "-s", "--source",
        default=str(DEFAULT_CAMERA_SOURCE),
        help="Sumber video: Index webcam/DroidCam (0, 1, 2) atau URL DroidCam (misal: http://192.168.1.15:4747/video)"
    )
    parser.add_argument(
        "-i", "--ip",
        default=None,
        help="IP HP DroidCam (misal: 192.168.1.15) -> otomatis menggunakan http://<IP>:4747/video"
    )
    args = parser.parse_args()

    if args.ip:
        cam_source = f"http://{args.ip.strip()}:4747/video"
    else:
        cam_source = args.source

    # Cek apakah source berupa angka (index webcam)
    if isinstance(cam_source, str) and cam_source.isdigit():
        cam_source = int(cam_source)

    print(f"\n==================================================")
    print(f"  MENGHUBUNGKAN KE KAMERA: {cam_source}")
    print(f"==================================================")

    cap = cv2.VideoCapture(cam_source)
    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verifikasi koneksi kamera & fallback cerdas jika index tidak ditemukan
    if not cap.isOpened():
        print(f"\n[!] PERINGATAN: Gagal membuka kamera pada sumber '{cam_source}'!")
        if isinstance(cam_source, int) and cam_source != 0:
            print(f"[i] Mencoba beralih ke Index Kamera 0 (Default Webcam)...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cam_source = 0
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"[OK] Berhasil terhubung ke Index Kamera 0!")
        
        if not cap.isOpened():
            print("\n[X] ERROR: Tidak dapat mengakses kamera!")
            print("----------------------------------------------------------------------")
            print("PANDUAN KONEKSI DROIDCAM:")
            print("1. CARA WIFI / IP (Tanpa Client PC):")
            print("   - Buka aplikasi DroidCam di HP (pastikan HP & PC satu jaringan WiFi).")
            print("   - Lihat 'WiFi IP' di layar HP (contoh: 192.168.1.15).")
            print("   - Jalankan: python main.py -i 192.168.1.15")
            print("   - Atau: python main.py -s http://192.168.1.15:4747/video")
            print("2. CARA DROIDCAM CLIENT PC (USB / WiFi Virtual Cam):")
            print("   - Buka DroidCam Client di Windows, lalu tekan 'Start'.")
            print("   - Jalankan: python main.py -s 1 (atau -s 2)")
            print("----------------------------------------------------------------------\n")
            return

    print(f"[OK] Kamera aktif ({cam_source})\n")

    face_system = FaceIdentitySystem("identitas")

    # Inisialisasi Logo Teks Absolute Cinema (Transparan)
    cinema_logo_path = "cinema_logo.png"
    cinema_logo_img = cv2.imread(cinema_logo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(cinema_logo_path) else None
    if cinema_logo_img is not None:
        print("Logo Teks Absolute Cinema siap ditampilkan!")

    # Inisialisasi Sticker Roblox Man Face
    manface_path = "meme_senyum.jpg"
    if not os.path.exists(manface_path):
        manface_path = "meme_senyum.png"
    manface_sticker_img = cv2.imread(manface_path) if os.path.exists(manface_path) else None
    if manface_sticker_img is not None:
        print("Sticker Roblox Man Face siap ditampilkan!")

    # Inisialisasi Hand Tracking MediaPipe (Dukungan 2 Tangan)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hand_landmark_style = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=-1, circle_radius=4)
    hand_connection_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    # Inisialisasi Face Mesh MediaPipe untuk Deteksi Ekspresi Wajah
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    window_name = "frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    p_time = 0
    frame_count = 0
    detected_faces = []
    is_fullscreen = False
    is_mirrored = True  # Mode mirror kamera (tekan 'm' untuk switch)

    print("==================================================")
    print("   IDENTITAS WAJAH + EKSPRESI + MEME CINEMA & ROBLOX")
    print("==================================================")
    print(" - Deteksi Ekspresi Wajah (Senyum -> Meme Roblox Man Face, Cemberut, Kaget, Netral)")
    print(" - Pose 10 Jari (Angkat 2 Tangan 🖐️🖐️) -> MEME ABSOLUTE CINEMA")
    print(" - Pose Jari Tengah 🖕 -> 'fuck you'")
    print(" - Tekan 'm' : Toggle Mirror / Flip Kamera")
    print(" - Tekan 'r' : Reload database identitas")
    print(" - Tekan 'f' : Fullscreen On / Off")
    print(" - Tekan 'q' atau 'ESC' : Keluar\n")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("[!] Frame tidak terbaca atau streaming terputus.")
            break

        if is_mirrored:
            img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_count += 1

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Deteksi Identitas Wajah
        if frame_count % 2 == 0 or len(detected_faces) == 0:
            detected_faces = face_system.recognize_faces_in_frame(img)

        # 2. Deteksi Ekspresi Wajah (Face Mesh)
        mesh_results = face_mesh.process(img_rgb)
        face_expressions = []

        if mesh_results.multi_face_landmarks:
            for face_lms in mesh_results.multi_face_landmarks:
                expr_text, expr_color = get_facial_expression(face_lms.landmark, w, h)
                
                xs = [int(lm.x * w) for lm in face_lms.landmark]
                ys = [int(lm.y * h) for lm in face_lms.landmark]
                f_bx1, f_bx2 = max(0, min(xs)), min(w, max(xs))
                f_by1, f_by2 = max(0, min(ys)), min(h, max(ys))

                face_expressions.append({
                    "box": (f_bx1, f_by1, f_bx2, f_by2),
                    "center": ((f_bx1 + f_bx2) // 2, (f_by1 + f_by2) // 2),
                    "expression": expr_text,
                    "color": expr_color
                })

        # 3. Hand Tracking (Bisa 2 Tangan)
        hand_results = hands.process(img_rgb)

        finger_count_total = 0
        hands_detected_count = 0
        gesture_name = "Siap"
        is_middle_finger = False
        is_absolute_cinema = False

        if hand_results.multi_hand_landmarks:
            hands_detected_count = len(hand_results.multi_hand_landmarks)

            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_style,
                    connection_drawing_spec=hand_connection_style
                )

                lm_list = []
                x_list, y_list = [], []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))
                    x_list.append(cx)
                    y_list.append(cy)

                # Kotak Hijau Tangan
                hx_min, hx_max = max(0, min(x_list) - 20), min(w, max(x_list) + 20)
                hy_min, hy_max = max(0, min(y_list) - 20), min(h, max(y_list) + 20)

                fingers = get_fingers_state(lm_list)
                finger_count = sum(fingers)
                finger_count_total += finger_count

                # Gesture Per Tangan (Deteksi Jari Tengah)
                if fingers[2] == 1 and fingers[1] == 0 and fingers[3] == 0 and fingers[4] == 0:
                    gesture_name = "fuck you"
                    is_middle_finger = True
                    hand_box_color = (0, 0, 255)
                    
                    mid_tip_x, mid_tip_y = lm_list[12]
                    cv2.putText(
                        img,
                        "fuck you",
                        (mid_tip_x - 35, mid_tip_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )
                else:
                    hand_box_color = (0, 255, 0)

                cv2.rectangle(img, (hx_min, hy_min), (hx_max, hy_max), hand_box_color, 1)

            # Cek Pose ABSOLUTE CINEMA (Angkat 2 Tangan Telapak Terbuka / 10 Jari)
            if hands_detected_count == 2 and finger_count_total >= 9:
                is_absolute_cinema = True
                gesture_name = "ABSOLUTE CINEMA 🎬"
            elif not is_middle_finger:
                if finger_count_total == 1:
                    gesture_name = "MENUNJUK (1 Jari) 👆"
                elif finger_count_total == 2:
                    gesture_name = "2 JARI / PEACE ✌️"
                elif finger_count_total == 5:
                    gesture_name = "5 JARI (Telapak Terbuka) 🖐️"
                else:
                    gesture_name = f"{finger_count_total} Jari Terbuka"

        # 4. GAMBAR IDENTITAS & EKSPRESI WAJAH (Kotak Bersih Minimalis)
        verified_names = []
        current_expressions = []

        for face in detected_faces:
            fx1, fy1, fx2, fy2 = face["box"]
            name = face["name"]
            fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

            # Cari ekspresi yang sesuai lokasi wajah
            matching_expr = "Netral"
            expr_color = (0, 255, 120)
            if face_expressions:
                best_expr_match = min(face_expressions, key=lambda fe: dist((fcx, fcy), fe["center"]))
                matching_expr = best_expr_match["expression"]
                expr_color = best_expr_match["color"]
                current_expressions.append(matching_expr)

            is_known = name != "Tidak Dikenal"
            box_color = (0, 255, 0) if is_known else (0, 0, 255)
            label_text = f"ID: {name}" if is_known else "Tidak Dikenal"

            if is_known:
                verified_names.append(name)

            # Kotak Wajah Minimalis (Garis Tipis)
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), box_color, 1)

            # Label Identitas (Atas Kotak)
            cv2.putText(
                img,
                label_text,
                (fx1, max(18, fy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                box_color,
                1
            )

            # Label Ekspresi (Bawah Kotak)
            cv2.putText(
                img,
                f"Ekspresi: {matching_expr}",
                (fx1, min(h - 8, fy2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                expr_color,
                1
            )

        # 5. EFEK STICKER ROBLOX MAN FACE SAAT SENYUM / SMIRK (Kecil & Rapi)
        is_smiling_active = any(expr in ["Tersenyum", "Tertawa"] for expr in current_expressions)
        if is_smiling_active and manface_sticker_img is not None:
            card_w = 140
            card_h = int(card_w * (manface_sticker_img.shape[0] / manface_sticker_img.shape[1]))
            c_x = w - card_w - 15
            c_y = 15
            
            # Border Putih Rapi di Sekeliling Sticker
            cv2.rectangle(img, (c_x - 2, c_y - 2), (c_x + card_w + 2, c_y + card_h + 2), (255, 255, 255), 2)
            overlay_image_alpha(img, manface_sticker_img, (c_x, c_y), (card_w, card_h))

        # 6. EFEK LOGO TEKS ABSOLUTE CINEMA DI BAWAH TENGAH (Lebih Ramping)
        if is_absolute_cinema and cinema_logo_img is not None:
            logo_w = int(w * 0.42)
            logo_h = int(logo_w * (cinema_logo_img.shape[0] / cinema_logo_img.shape[1]))
            logo_x = (w - logo_w) // 2
            logo_y = max(0, h - logo_h - 10)
            overlay_image_alpha(img, cinema_logo_img, (logo_x, logo_y), (logo_w, logo_h))

        # 7. TAMPILAN HUD & STATUS (Minimalis, Ringkas, Card Transparan Elegan)
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Background Card Transparan untuk HUD
        hud_bg = img.copy()
        cv2.rectangle(hud_bg, (10, 10), (320, 110), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.45, img, 0.55, 0, img)
        cv2.rectangle(img, (10, 10), (320, 110), (70, 70, 70), 1)

        # FPS
        cv2.putText(img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Status Identitas Wajah
        if verified_names:
            id_status = f"ID: {', '.join(verified_names)}"
            id_color = (0, 255, 120)
        elif len(detected_faces) > 0:
            id_status = "ID: Tidak Dikenal"
            id_color = (0, 0, 255)
        else:
            id_status = "ID: Mencari Wajah..."
            id_color = (180, 180, 180)

        cv2.putText(img, id_status, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, id_color, 1)

        # Status Ekspresi di HUD
        if current_expressions:
            expr_hud_text = f"Ekspresi: {', '.join(current_expressions)}"
            expr_hud_color = (0, 255, 255)
        else:
            expr_hud_text = "Ekspresi: -"
            expr_hud_color = (160, 160, 160)

        cv2.putText(img, expr_hud_text, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, expr_hud_color, 1)

        # Status Gesture Tangan
        if hand_results.multi_hand_landmarks:
            if is_absolute_cinema:
                status_text = "Tangan: 🎬 ABSOLUTE CINEMA"
                gesture_color = (0, 215, 255)
            elif is_middle_finger:
                status_text = "Tangan: fuck you 🖕"
                gesture_color = (0, 0, 255)
            else:
                status_text = f"Tangan: {gesture_name} ({finger_count_total} Jari)"
                gesture_color = (0, 255, 120)

            cv2.putText(
                img,
                status_text,
                (20, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                gesture_color,
                1
            )
        else:
            cv2.putText(
                img,
                "Tangan: Siap",
                (20, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 160),
                1
            )

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            is_mirrored = not is_mirrored
            print(f"[i] Mirror / Flip Kamera: {'ON' if is_mirrored else 'OFF'}")
        elif key == ord('r'):
            face_system.load_known_identities()
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
