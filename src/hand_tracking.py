import cv2
import mediapipe as mp
from src.utils import dist


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


class HandGestureTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_landmark_style = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=-1, circle_radius=5)
        self.hand_connection_style = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, img, img_rgb, w, h, ui_scale=1.0):
        hand_results = self.hands.process(img_rgb)
        finger_count_total = 0
        hands_detected_count = 0
        gesture_name = "Siap"
        is_middle_finger = False
        is_absolute_cinema = False

        if hand_results.multi_hand_landmarks:
            hands_detected_count = len(hand_results.multi_hand_landmarks)

            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.hand_landmark_style,
                    connection_drawing_spec=self.hand_connection_style
                )

                lm_list = []
                x_list, y_list = [], []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))
                    x_list.append(cx)
                    y_list.append(cy)

                hx_min = max(0, min(x_list) - int(20 * ui_scale))
                hx_max = min(w, max(x_list) + int(20 * ui_scale))
                hy_min = max(0, min(y_list) - int(20 * ui_scale))
                hy_max = min(h, max(y_list) + int(20 * ui_scale))

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
                        (mid_tip_x - int(35 * ui_scale), mid_tip_y - int(12 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65 * ui_scale,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    hand_box_color = (0, 255, 0)

                cv2.rectangle(img, (hx_min, hy_min), (hx_max, hy_max), hand_box_color, max(1, int(2 * ui_scale)), cv2.LINE_AA)

            # Cek Pose ABSOLUTE CINEMA
            if hands_detected_count == 2 and finger_count_total >= 9:
                is_absolute_cinema = True
                gesture_name = "ABSOLUTE CINEMA"
            elif not is_middle_finger:
                if finger_count_total == 1:
                    gesture_name = "1 Jari (Menunjuk)"
                elif finger_count_total == 2:
                    gesture_name = "2 Jari (Peace)"
                elif finger_count_total == 5:
                    gesture_name = "5 Jari (Terbuka)"
                else:
                    gesture_name = f"{finger_count_total} Jari Terbuka"

        return {
            "hands_count": hands_detected_count,
            "finger_count_total": finger_count_total,
            "gesture_name": gesture_name,
            "is_absolute_cinema": is_absolute_cinema,
            "is_middle_finger": is_middle_finger
        }
