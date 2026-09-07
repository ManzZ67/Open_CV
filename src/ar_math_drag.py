import cv2
import random
import math
import numpy as np
from src.utils import dist


class ARBlock:
    """Balok angka AR yang bisa di-pinch (cubit), digeser (drag), dan ditaruh (drop)."""
    def __init__(self, value, pos, size=(90, 90), color=(220, 100, 20)):
        self.value = value
        self.x, self.y = int(pos[0]), int(pos[1])
        self.w, self.h = int(size[0]), int(size[1])
        self.orig_x, self.orig_y = self.x, self.y
        self.color = color
        self.is_dragging = False

    def contains(self, px, py, margin=35):
        """Mengecek apakah jari berada di atas balok (dengan toleransi grab yang nyaman)."""
        return (self.x - self.w // 2 - margin <= px <= self.x + self.w // 2 + margin and
                self.y - self.h // 2 - margin <= py <= self.y + self.h // 2 + margin)

    def draw(self, img, ui_scale=1.0):
        w = int(self.w * ui_scale)
        h = int(self.h * ui_scale)
        x1 = self.x - w // 2
        y1 = self.y - h // 2
        x2 = self.x + w // 2
        y2 = self.y + h // 2

        # Efek bayangan / glow saat di-drag
        if self.is_dragging:
            cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 255), 4, cv2.LINE_AA)
            bg_color = (min(255, self.color[0] + 50), min(255, self.color[1] + 50), min(255, self.color[2] + 50))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
            bg_color = self.color

        # Kotak Balok AR Solid Elegan
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

        # Angka di tengah balok
        text = str(self.value)
        font_scale = 1.2 * ui_scale
        thickness = max(2, int(3 * ui_scale))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = self.x - tw // 2
        ty = self.y + th // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


class ARMathDragEngine:
    def __init__(self):
        self.stage = 1
        self.score = 0
        self.dragged_block = None
        self.offset_x = 0
        self.offset_y = 0
        self.lost_frames = 0
        
        # Soal & Balok Pilihan
        self.num1 = 10
        self.op = "+"
        self.num2 = 6
        self.ans = 16
        self.missing_target = "ans"  # "num1", "num2", atau "ans"
        self.blocks = []
        self.success_animation_time = 0
        self.wrong_feedback_time = 0
        
        self.generate_new_stage()

    def generate_new_stage(self):
        """Membuat soal matematika yang seru dan balok-balok pilihan melayang."""
        self.op = random.choice(["+", "-"])
        
        if self.op == "+":
            self.ans = random.randint(6, 12 + self.stage * 3)
            self.num1 = random.randint(1, self.ans - 1)
            self.num2 = self.ans - self.num1
        else:
            self.num1 = random.randint(6, 12 + self.stage * 3)
            self.num2 = random.randint(1, self.num1 - 1)
            self.ans = self.num1 - self.num2

        # Tentukan posisi slot yang kosong
        self.missing_target = random.choice(["num2", "ans"])
        correct_val = self.num2 if self.missing_target == "num2" else self.ans

        # Buat 3 balok pilihan melayang (1 benar + 2 pengalih)
        distractor1 = max(0, correct_val + random.choice([-3, -2, -1, 1, 2, 3]))
        distractor2 = max(0, correct_val + random.choice([-5, -4, 4, 5]))
        while distractor2 == distractor1 or distractor2 == correct_val:
            distractor2 = correct_val + random.randint(2, 6)

        options = [correct_val, distractor1, distractor2]
        random.shuffle(options)

        self.blocks = []
        # Posisi mengambang di kanan atas
        spawn_positions = [(860, 150), (1000, 150), (1140, 150)]
        colors = [(220, 110, 30), (30, 130, 220), (160, 50, 200)]

        for i, val in enumerate(options):
            pos = spawn_positions[i]
            col = colors[i % len(colors)]
            self.blocks.append(ARBlock(val, pos, size=(85, 85), color=col))

        self.dragged_block = None
        self.lost_frames = 0

    def update_hand_interaction(self, pinch_pt, is_pinching, w, h, ui_scale=1.0):
        """Memproses logika pinch & drag dengan proteksi anti-lepas saat melewati wajah."""
        # Jika tangan hilang sesaat (misal terhalang muka), tahan balok di posisi terakhir sampai 12 frame (~0.4s)
        if pinch_pt is None:
            if self.dragged_block is not None:
                self.lost_frames += 1
                if self.lost_frames > 15:
                    self.dragged_block.is_dragging = False
                    self.dragged_block = None
                    self.lost_frames = 0
            return

        self.lost_frames = 0
        px, py = pinch_pt

        # 1. Mulai Drag jika sedang Pinch pada sebuah Balok
        if is_pinching:
            if self.dragged_block is None:
                for block in reversed(self.blocks):
                    if block.contains(px, py, margin=int(35 * ui_scale)):
                        self.dragged_block = block
                        block.is_dragging = True
                        self.offset_x = block.x - px
                        self.offset_y = block.y - py
                        break
            else:
                # Pergerakan mulus (Smooth drag)
                target_x = px + self.offset_x
                target_y = py + self.offset_y
                self.dragged_block.x = int(self.dragged_block.x + 0.8 * (target_x - self.dragged_block.x))
                self.dragged_block.y = int(self.dragged_block.y + 0.8 * (target_y - self.dragged_block.y))
        else:
            # 2. Lepas Pinch (Drop)
            if self.dragged_block is not None:
                self.dragged_block.is_dragging = False
                self.check_drop_target(w, h, ui_scale)
                self.dragged_block = None

    def get_slot_rects(self, w, h, ui_scale=1.0):
        """Menghitung koordinat kotak slot persamaan matematika di tengah bawah."""
        slot_w = int(105 * ui_scale)
        slot_h = int(105 * ui_scale)
        gap = int(18 * ui_scale)
        total_w = 5 * slot_w + 4 * gap
        start_x = (w - total_w) // 2
        start_y = h - slot_h - int(90 * ui_scale)

        slots = {
            "num1": (start_x, start_y, slot_w, slot_h),
            "op": (start_x + (slot_w + gap), start_y, slot_w, slot_h),
            "num2": (start_x + 2 * (slot_w + gap), start_y, slot_w, slot_h),
            "eq": (start_x + 3 * (slot_w + gap), start_y, slot_w, slot_h),
            "ans": (start_x + 4 * (slot_w + gap), start_y, slot_w, slot_h),
        }
        return slots

    def check_drop_target(self, w, h, ui_scale=1.0):
        """Mengecek apakah balok yang di-drop berada di dalam atau dekat slot kosong yang tepat."""
        slots = self.get_slot_rects(w, h, ui_scale)
        tx, ty, tw, th = slots[self.missing_target]
        slot_cx = tx + tw // 2
        slot_cy = ty + th // 2

        bx, by = self.dragged_block.x, self.dragged_block.y
        d_to_slot = dist((bx, by), (slot_cx, slot_cy))

        # Toleransi drop fleksibel & magnetik (radius 110px atau berada di sekitar slot)
        drop_margin = int(50 * ui_scale)
        is_near_slot = (d_to_slot < 110 * ui_scale) or (
            (tx - drop_margin <= bx <= tx + tw + drop_margin) and
            (ty - drop_margin <= by <= ty + th + drop_margin)
        )

        if is_near_slot:
            correct_val = self.num2 if self.missing_target == "num2" else self.ans
            if self.dragged_block.value == correct_val:
                # JAWABAN BENAR!
                self.score += 50
                self.stage += 1
                self.success_animation_time = 25
                self.generate_new_stage()
            else:
                # Jawaban Salah -> Kembalikan ke posisi awal
                self.wrong_feedback_time = 20
                self.dragged_block.x = self.dragged_block.orig_x
                self.dragged_block.y = self.dragged_block.orig_y
        else:
            # Tidak ditaruh di slot -> Kembalikan ke posisi awal
            self.dragged_block.x = self.dragged_block.orig_x
            self.dragged_block.y = self.dragged_block.orig_y

    def draw_game_scene(self, img, pinch_pt, is_pinching, ui_scale=1.0):
        h, w = img.shape[:2]

        # ======================================================================
        # 1. TOP-LEFT HUD CARD (Tahap & Skor)
        # ======================================================================
        card_w, card_h = int(240 * ui_scale), int(100 * ui_scale)
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + card_w, 20 + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
        cv2.rectangle(img, (20, 20), (20 + card_w, 20 + card_h), (60, 60, 60), 2, cv2.LINE_AA)

        cv2.putText(img, f"Tahap: {self.stage}", (35, int(60 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * ui_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Skor: {self.score}", (35, int(98 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * ui_scale, (0, 255, 120), 2, cv2.LINE_AA)

        # ======================================================================
        # 2. PERSAMAAN MATEMATIKA (SLOT KOTAK DI TENGAH BAWAH)
        # ======================================================================
        slots = self.get_slot_rects(w, h, ui_scale)
        tx, ty, tw, th = slots[self.missing_target]
        slot_cx, slot_cy = tx + tw // 2, ty + th // 2

        # Indikator Magnetik
        is_hovering_target = False
        if self.dragged_block is not None:
            d_hover = dist((self.dragged_block.x, self.dragged_block.y), (slot_cx, slot_cy))
            if d_hover < 140 * ui_scale:
                is_hovering_target = True
                cv2.line(img, (self.dragged_block.x, self.dragged_block.y), (slot_cx, slot_cy), (0, 255, 120), 2, cv2.LINE_AA)

        def draw_slot(rect, text, is_empty=False):
            sx, sy, sw, sh = rect
            if is_empty:
                slot_border_col = (0, 255, 0) if is_hovering_target else (0, 255, 255)
                thickness = 4 if is_hovering_target else 3
                
                if is_hovering_target:
                    slot_bg = img.copy()
                    cv2.rectangle(slot_bg, (sx, sy), (sx + sw, sy + sh), (0, 180, 0), -1)
                    cv2.addWeighted(slot_bg, 0.40, img, 0.60, 0, img)
                
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), slot_border_col, thickness, cv2.LINE_AA)
                cv2.putText(img, "?", (sx + int(sw * 0.35), sy + int(sh * 0.68)), cv2.FONT_HERSHEY_DUPLEX, 1.4 * ui_scale, slot_border_col, 3, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (40, 150, 40), -1)
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2, cv2.LINE_AA)
                font_scale = 1.3 * ui_scale
                (tw_t, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
                cv2.putText(img, text, (sx + (sw - tw_t) // 2, sy + (sh + th_t) // 2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        draw_slot(slots["num1"], str(self.num1))
        draw_slot(slots["op"], self.op)
        draw_slot(slots["num2"], str(self.num2), is_empty=(self.missing_target == "num2"))
        draw_slot(slots["eq"], "=")
        draw_slot(slots["ans"], str(self.ans), is_empty=(self.missing_target == "ans"))

        # ======================================================================
        # 3. BALOK-BALOK PILIHAN AR YANG MELAYANG
        # ======================================================================
        for block in self.blocks:
            block.draw(img, ui_scale)

        # ======================================================================
        # 4. PINCH CURSOR FEEDBACK (LINGKARAN KURSOR JARI)
        # ======================================================================
        if pinch_pt is not None:
            px, py = pinch_pt
            cursor_color = (0, 255, 0) if is_pinching else (0, 255, 255)
            cv2.circle(img, (px, py), int(18 * ui_scale), cursor_color, 2, cv2.LINE_AA)
            if is_pinching:
                cv2.circle(img, (px, py), int(8 * ui_scale), (0, 255, 0), -1, cv2.LINE_AA)

        # ======================================================================
        # 5. BOTTOM HINT TEXT
        # ======================================================================
        hint_text = "Pinch balok, drag ke slot kosong | 'r' reset | 'q' keluar"
        (htw, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, 1)
        cv2.putText(img, hint_text, ((w - htw) // 2, h - int(25 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Feedback Animasi
        if self.success_animation_time > 0:
            self.success_animation_time -= 1
            cv2.putText(img, "BENAR! +50", (w // 2 - int(100 * ui_scale), int(120 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 1.2 * ui_scale, (0, 255, 120), 3, cv2.LINE_AA)
        
        if self.wrong_feedback_time > 0:
            self.wrong_feedback_time -= 1
            cv2.putText(img, "SALAH!", (w // 2 - int(60 * ui_scale), int(120 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 1.2 * ui_scale, (0, 0, 255), 3, cv2.LINE_AA)
