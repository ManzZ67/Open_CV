import cv2
import math
import numpy as np
from src.utils import dist, overlay_image_alpha


# ==============================================================================
# 1. AR AIR CANVAS (MENGGAMBAR DI UDARA DENGAN JARI)
# ==============================================================================
class ARAirCanvas:
    def __init__(self):
        self.colors = [
            ("CYAN", (255, 255, 0)),
            ("GREEN", (0, 255, 128)),
            ("ORANGE", (0, 140, 255)),
            ("MAGENTA", (255, 0, 255)),
            ("ERASER", (0, 0, 0))
        ]
        self.current_color_idx = 0
        self.brush_thickness = 8
        self.eraser_thickness = 45
        self.canvas = None
        self.xp, self.yp = 0, 0

    def draw_palette(self, img, ui_scale=1.0):
        h, w = img.shape[:2]
        bar_h = int(65 * ui_scale)
        num_colors = len(self.colors)
        slot_w = w // num_colors

        # Background palette
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (25, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.line(img, (0, bar_h), (w, bar_h), (80, 80, 80), 2, cv2.LINE_AA)

        for i, (name, col) in enumerate(self.colors):
            x1 = i * slot_w
            x2 = (i + 1) * slot_w
            is_active = (i == self.current_color_idx)

            if is_active:
                cv2.rectangle(img, (x1 + 4, 4), (x2 - 4, bar_h - 4), (255, 255, 255), 2, cv2.LINE_AA)
            
            box_col = (50, 50, 50) if name == "ERASER" else col
            cv2.rectangle(img, (x1 + 10, 10), (x2 - 10, bar_h - 10), box_col, -1)
            
            # Label
            text_col = (255, 255, 255) if name == "ERASER" else (0, 0, 0)
            cv2.putText(img, name, (x1 + int(slot_w * 0.25), int(bar_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, text_col, 2, cv2.LINE_AA)

    def process_hand(self, img, lm_list, fingers, ui_scale=1.0):
        h, w = img.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), np.uint8)

        # Index tip (8) and Middle tip (12)
        x1, y1 = lm_list[8]
        x2, y2 = lm_list[12]

        # Selection Mode: 2 Jari terangkat (Telunjuk + Jari Tengah)
        if fingers[1] == 1 and fingers[2] == 1:
            self.xp, self.yp = 0, 0
            cv2.circle(img, (x1, y1), int(12 * ui_scale), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (x2, y2), int(12 * ui_scale), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

            # Cek jika memilih di menu palette atas
            if y1 < int(65 * ui_scale):
                slot_w = w // len(self.colors)
                chosen_idx = min(len(self.colors) - 1, max(0, x1 // slot_w))
                self.current_color_idx = chosen_idx

        # Drawing Mode: Hanya Jari Telunjuk terangkat
        elif fingers[1] == 1 and fingers[2] == 0:
            active_color = self.colors[self.current_color_idx][1]
            is_eraser = (self.colors[self.current_color_idx][0] == "ERASER")
            th = self.eraser_thickness if is_eraser else self.brush_thickness
            th = int(th * ui_scale)

            # Brush cursor
            cursor_col = (200, 200, 200) if is_eraser else active_color
            cv2.circle(img, (x1, y1), int(8 * ui_scale), cursor_col, -1, cv2.LINE_AA)

            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1

            if is_eraser:
                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), (0, 0, 0), th)
                cv2.line(img, (self.xp, self.yp), (x1, y1), (255, 255, 255), int(2 * ui_scale), cv2.LINE_AA)
            else:
                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), active_color, th, cv2.LINE_AA)

            self.xp, self.yp = x1, y1
        else:
            self.xp, self.yp = 0, 0

    def render_canvas(self, img):
        if self.canvas is None:
            return img
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)
        return img

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)


# ==============================================================================
# 2. AR FACE ACCESSORIES & CYBER VISOR
# ==============================================================================
class ARFaceFilter:
    def __init__(self):
        self.filter_mode = "CYBER_VISOR"  # "CYBER_VISOR", "GOLDEN_CROWN"

    def draw_cyber_visor(self, img, landmarks, w, h, ui_scale=1.0):
        def get_p(idx):
            return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

        # Mata Kiri (33) & Kanan (263), Titik Hidung Atas (168)
        p_left = get_p(33)
        p_right = get_p(263)
        p_mid = get_p(168)
        eye_dist = dist(p_left, p_right)

        # Hitung Sudut Kemiringan Kepala
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        angle = math.degrees(math.atan2(dy, dx))

        visor_w = int(eye_dist * 2.4)
        visor_h = int(eye_dist * 0.75)

        # Gambar Kacamata Visor Holografik Neon
        cx, cy = p_mid[0], p_mid[1]
        pts = [
            (int(cx - visor_w * 0.5), int(cy - visor_h * 0.4)),
            (int(cx + visor_w * 0.5), int(cy - visor_h * 0.4)),
            (int(cx + visor_w * 0.42), int(cy + visor_h * 0.5)),
            (int(cx - visor_w * 0.42), int(cy + visor_h * 0.5))
        ]

        # Rotasikan titik-titik sesuai orientasi kepala
        rot_pts = []
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for px, py in pts:
            rx = int(cx + (px - cx) * cos_a - (py - cy) * sin_a)
            ry = int(cy + (px - cx) * sin_a + (py - cy) * cos_a)
            rot_pts.append((rx, ry))

        poly = np.array(rot_pts, np.int32)
        
        # Transparent visor glass
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (255, 200, 0))  # Cyan Neon Visor
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [poly], True, (0, 255, 255), max(1, int(3 * ui_scale)), cv2.LINE_AA)

        # Cyber Visor Tech Line
        p_in1 = ((rot_pts[0][0] + rot_pts[3][0]) // 2, (rot_pts[0][1] + rot_pts[3][1]) // 2)
        p_in2 = ((rot_pts[1][0] + rot_pts[2][0]) // 2, (rot_pts[1][1] + rot_pts[2][1]) // 2)
        cv2.line(img, p_in1, p_in2, (255, 255, 255), max(1, int(2 * ui_scale)), cv2.LINE_AA)

    def draw_golden_crown(self, img, landmarks, w, h, ui_scale=1.0):
        def get_p(idx):
            return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

        p_top = get_p(10)  # Puncak dahi
        p_l = get_p(234)
        p_r = get_p(454)
        head_w = dist(p_l, p_r)

        cx, cy = p_top[0], p_top[1] - int(head_w * 0.25)
        cw = int(head_w * 0.8)
        ch = int(cw * 0.6)

        # Mahkota Emas 3 Puncak
        crown_pts = [
            (cx - cw // 2, cy),
            (cx - cw // 2, cy - ch // 2),
            (cx - cw // 4, cy - ch // 4),
            (cx, cy - ch),
            (cx + cw // 4, cy - ch // 4),
            (cx + cw // 2, cy - ch // 2),
            (cx + cw // 2, cy)
        ]
        poly = np.array(crown_pts, np.int32)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (0, 215, 255))
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
        cv2.polylines(img, [poly], True, (255, 255, 255), max(1, int(2 * ui_scale)), cv2.LINE_AA)

        # Permata di 3 puncak mahkota
        for pt in [crown_pts[1], crown_pts[3], crown_pts[5]]:
            cv2.circle(img, pt, int(6 * ui_scale), (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, int(6 * ui_scale), (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# 3. AR 3D HOLOGRAPHIC CUBE FLOATING ON PALM
# ==============================================================================
class AR3DHologramCube:
    def __init__(self):
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

    def draw_cube_on_palm(self, img, palm_center, dt=0.03, size=55, ui_scale=1.0):
        cx, cy = int(palm_center[0]), int(palm_center[1])
        self.angle_x += dt * 1.8
        self.angle_y += dt * 2.4
        self.angle_z += dt * 1.2
        s = size * ui_scale

        # 8 Titik Sudut Kubus 3D
        nodes = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * s

        # Matriks Rotasi 3D
        rad_x, rad_y, rad_z = self.angle_x, self.angle_y, self.angle_z
        Rx = np.array([[1, 0, 0], [0, math.cos(rad_x), -math.sin(rad_x)], [0, math.sin(rad_x), math.cos(rad_x)]])
        Ry = np.array([[math.cos(rad_y), 0, math.sin(rad_y)], [0, 1, 0], [-math.sin(rad_y), 0, math.cos(rad_y)]])
        Rz = np.array([[math.cos(rad_z), -math.sin(rad_z), 0], [math.sin(rad_z), math.cos(rad_z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        # Proyeksi 3D ke 2D
        proj_pts = []
        for node in nodes:
            r_node = R @ node
            # Efek perspektif sederhana
            fov = 300.0
            z_factor = fov / (fov + r_node[2])
            px = int(cx + r_node[0] * z_factor)
            py = int(cy - int(70 * ui_scale) + r_node[1] * z_factor)
            proj_pts.append((px, py))

        # 12 Garis Rangka Kubus 3D
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Laser Tether dari telapak tangan ke kubus
        cv2.line(img, (cx, cy), (cx, cy - int(70 * ui_scale)), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(10 * ui_scale), (0, 255, 255), 2, cv2.LINE_AA)

        for p1_i, p2_i in edges:
            cv2.line(img, proj_pts[p1_i], proj_pts[p2_i], (255, 220, 0), max(1, int(2 * ui_scale)), cv2.LINE_AA)

        for pt in proj_pts:
            cv2.circle(img, pt, max(2, int(4 * ui_scale)), (0, 255, 255), -1, cv2.LINE_AA)
