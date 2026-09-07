import cv2
import random
import time
import math
import numpy as np


class MathQuestionGenerator:
    def __init__(self):
        self.operations = ['+', '-']
        self.level = 1

    def generate(self, max_answer=10):
        """Membuat soal matematika dengan jawaban rentang 0 s.d. 10 (bisa dijawab jari tangan)."""
        op = random.choice(self.operations)
        if op == '+':
            ans = random.randint(1, min(10, 5 + self.level * 2))
            num1 = random.randint(0, ans)
            num2 = ans - num1
            question_text = f"{num1} + {num2} = ?"
            return question_text, ans
        elif op == '-':
            num1 = random.randint(1, min(10, 5 + self.level * 2))
            num2 = random.randint(0, num1)
            ans = num1 - num2
            question_text = f"{num1} - {num2} = ?"
            return question_text, ans
        return "1 + 1 = ?", 2


class MathGameEngine:
    def __init__(self, time_limit_per_question=6.0):
        self.generator = MathQuestionGenerator()
        self.time_limit = time_limit_per_question
        
        # Game State
        self.state = "START"  # "START", "PLAYING", "ROUND_RESULT", "GAME_OVER"
        self.score = 0
        self.streak = 0
        self.max_streak = 0
        self.lives = 3
        self.level = 1
        self.current_question = ""
        self.current_answer = 0
        
        # Timers
        self.question_start_time = 0
        self.hold_start_time = 0
        self.held_number = -1
        self.hold_duration_needed = 1.0  # Waktu tahan jari 1 detik untuk konfirmasi jawaban
        self.result_message = ""
        self.result_color = (0, 255, 0)
        self.round_result_end_time = 0

        # Highscore lokal
        self.high_score = 0

    def start_game(self):
        self.state = "PLAYING"
        self.score = 0
        self.streak = 0
        self.lives = 3
        self.level = 1
        self.next_question()

    def next_question(self):
        self.generator.level = self.level
        self.current_question, self.current_answer = self.generator.generate(max_answer=10)
        self.question_start_time = time.time()
        self.hold_start_time = 0
        self.held_number = -1
        self.state = "PLAYING"

    def update(self, detected_fingers):
        """Memproses input jari pemain dan memperbarui status game."""
        now = time.time()

        if self.state == "PLAYING":
            # 1. Cek Batas Waktu Soal
            elapsed = now - self.question_start_time
            time_left = max(0.0, self.time_limit - elapsed)

            if time_left <= 0:
                # Waktu Habis -> Salah
                self.handle_wrong(reason="WAKTU HABIS!")
                return

            # 2. Cek Jawaban Jari Pemain
            if detected_fingers >= 0:
                if detected_fingers == self.held_number:
                    # Sedang menahan angka yang sama
                    if self.hold_start_time > 0:
                        hold_time = now - self.hold_start_time
                        if hold_time >= self.hold_duration_needed:
                            # Jawaban Dikonfirmasi!
                            if detected_fingers == self.current_answer:
                                self.handle_correct()
                            else:
                                self.handle_wrong(reason=f"SALAH! JAWABAN: {self.current_answer}")
                else:
                    # Angka berubah, mulai hitung waktu tahan baru
                    self.held_number = detected_fingers
                    self.hold_start_time = now
            else:
                self.held_number = -1
                self.hold_start_time = 0

        elif self.state == "ROUND_RESULT":
            if now >= self.round_result_end_time:
                if self.lives <= 0:
                    self.state = "GAME_OVER"
                    if self.score > self.high_score:
                        self.high_score = self.score
                else:
                    self.next_question()

    def handle_correct(self):
        self.streak += 1
        self.max_streak = max(self.max_streak, self.streak)
        bonus = self.streak * 20
        points = 100 + bonus
        self.score += points
        
        # Naik level tiap kelipatan 5 streak
        if self.streak % 4 == 0:
            self.level += 1

        self.result_message = f"BENAR! +{points} PTS"
        self.result_color = (0, 255, 120)
        self.state = "ROUND_RESULT"
        self.round_result_end_time = time.time() + 1.2

    def handle_wrong(self, reason):
        self.lives -= 1
        self.streak = 0
        self.result_message = reason
        self.result_color = (0, 0, 255)
        self.state = "ROUND_RESULT"
        self.round_result_end_time = time.time() + 1.6

    def draw_game_ui(self, img, detected_fingers, ui_scale=1.0):
        h, w = img.shape[:2]
        now = time.time()

        # ======================================================================
        # 1. HEADER GAME CARD (Score, Lives, Streak)
        # ======================================================================
        header_h = int(70 * ui_scale)
        header_bg = img.copy()
        cv2.rectangle(header_bg, (0, 0), (w, header_h), (20, 20, 25), -1)
        cv2.addWeighted(header_bg, 0.70, img, 0.30, 0, img)
        cv2.line(img, (0, header_h), (w, header_h), (0, 255, 255), 2, cv2.LINE_AA)

        # Title
        cv2.putText(img, "MATH VISION GAME", (int(20 * ui_scale), int(45 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 0.9 * ui_scale, (0, 255, 255), 2, cv2.LINE_AA)

        # Score & Highscore
        cv2.putText(img, f"SCORE: {self.score}", (int(380 * ui_scale), int(45 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * ui_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Streak Combo
        if self.streak > 1:
            cv2.putText(img, f"STREAK x{self.streak} 🔥", (int(600 * ui_scale), int(45 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * ui_scale, (0, 165, 255), 2, cv2.LINE_AA)

        # Lives / Hearts (Teks ASCII Bersih)
        lives_text = "LIVES: " + ("<3 " * max(0, self.lives))
        cv2.putText(img, lives_text, (w - int(240 * ui_scale), int(45 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * ui_scale, (0, 50, 255), 2, cv2.LINE_AA)

        # ======================================================================
        # 2. KONTEN SESUAI STATE
        # ======================================================================
        if self.state == "START":
            # Start Screen Banner
            card_w, card_h = int(620 * ui_scale), int(260 * ui_scale)
            cx, cy = (w - card_w) // 2, (h - card_h) // 2
            
            card_bg = img.copy()
            cv2.rectangle(card_bg, (cx, cy), (cx + card_w, cy + card_h), (15, 15, 20), -1)
            cv2.addWeighted(card_bg, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, "TANTANGAN MATEMATIKA JARI", (cx + int(40 * ui_scale), cy + int(60 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * ui_scale, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "- Jawab soal matematika dengan mengangkat jari (0-10)", (cx + int(30 * ui_scale), cy + int(110 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "- Tahan posisi jari selama 1 detik untuk mengunci jawaban", (cx + int(30 * ui_scale), cy + int(145 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Tekan 'SPACE' atau Angkat 5 Jari untuk Mulai!", (cx + int(35 * ui_scale), cy + int(210 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui_scale, (0, 255, 120), 2, cv2.LINE_AA)

            if detected_fingers == 5:
                self.start_game()

        elif self.state in ["PLAYING", "ROUND_RESULT"]:
            # Kotak Soal Matematika di Tengah Atas
            box_w, box_h = int(480 * ui_scale), int(120 * ui_scale)
            bx, by = (w - box_w) // 2, header_h + int(20 * ui_scale)

            box_bg = img.copy()
            cv2.rectangle(box_bg, (bx, by), (bx + box_w, by + box_h), (15, 20, 30), -1)
            cv2.addWeighted(box_bg, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (0, 255, 255), 2, cv2.LINE_AA)

            # Teks Soal
            cv2.putText(img, self.current_question, (bx + int(80 * ui_scale), by + int(75 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 1.4 * ui_scale, (255, 255, 255), 3, cv2.LINE_AA)

            # Bar Waktu Mundur
            if self.state == "PLAYING":
                elapsed = now - self.question_start_time
                time_ratio = max(0.0, min(1.0, 1.0 - (elapsed / self.time_limit)))
                bar_w = int((box_w - 20) * time_ratio)
                bar_color = (0, 255, 0) if time_ratio > 0.4 else ((0, 215, 255) if time_ratio > 0.2 else (0, 0, 255))
                cv2.rectangle(img, (bx + 10, by + box_h - 14), (bx + 10 + bar_w, by + box_h - 6), bar_color, -1)

            # Hasil Ronde (Correct / Wrong Banner)
            if self.state == "ROUND_RESULT":
                cv2.putText(img, self.result_message, (bx + int(40 * ui_scale), by + box_h + int(45 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui_scale, self.result_color, 2, cv2.LINE_AA)

            # Feedback Jari Pemain & Bar Konfirmasi (Pojok Kiri Bawah)
            hud_w, hud_h = int(320 * ui_scale), int(90 * ui_scale)
            hx, hy = int(20 * ui_scale), h - hud_h - int(20 * ui_scale)
            cv2.rectangle(img, (hx, hy), (hx + hud_w, hy + hud_h), (20, 20, 20), -1)
            cv2.rectangle(img, (hx, hy), (hx + hud_w, hy + hud_h), (0, 255, 255), 1, cv2.LINE_AA)
            
            finger_str = f"Jari Terdeteksi: {detected_fingers}" if detected_fingers >= 0 else "Tangan: Siapkan Jari"
            cv2.putText(img, finger_str, (hx + int(15 * ui_scale), hy + int(35 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui_scale, (0, 255, 255), 2, cv2.LINE_AA)

            # Progress Bar Hold Konfirmasi Jawaban
            if self.state == "PLAYING" and self.hold_start_time > 0 and self.held_number >= 0:
                hold_ratio = min(1.0, (now - self.hold_start_time) / self.hold_duration_needed)
                fill_w = int((hud_w - 30) * hold_ratio)
                cv2.rectangle(img, (hx + 15, hy + int(55 * ui_scale)), (hx + 15 + fill_w, hy + int(70 * ui_scale)), (0, 255, 0), -1)
                cv2.rectangle(img, (hx + 15, hy + int(55 * ui_scale)), (hx + hud_w - 15, hy + int(70 * ui_scale)), (255, 255, 255), 1)

        elif self.state == "GAME_OVER":
            # Game Over Screen
            card_w, card_h = int(520 * ui_scale), int(260 * ui_scale)
            cx, cy = (w - card_w) // 2, (h - card_h) // 2

            card_bg = img.copy()
            cv2.rectangle(card_bg, (cx, cy), (cx + card_w, cy + card_h), (15, 15, 25), -1)
            cv2.addWeighted(card_bg, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(img, "GAME OVER", (cx + int(140 * ui_scale), cy + int(60 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 1.2 * ui_scale, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"SKOR AKHIR: {self.score}", (cx + int(120 * ui_scale), cy + int(115 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * ui_scale, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"MAX STREAK: {self.max_streak} 🔥", (cx + int(135 * ui_scale), cy + int(155 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui_scale, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Tekan 'SPACE' atau 'r' untuk Main Lagi", (cx + int(45 * ui_scale), cy + int(215 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui_scale, (0, 255, 0), 2, cv2.LINE_AA)
