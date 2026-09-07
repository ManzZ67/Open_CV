import mediapipe as mp
from src.utils import dist


def get_facial_expression(landmarks, w, h):
    """Menganalisis ekspresi wajah berdasarkan rasio landmark MediaPipe Face Mesh."""
    def get_pt(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    face_height = max(1.0, dist(get_pt(10), get_pt(152)))
    face_width = max(1.0, dist(get_pt(234), get_pt(454)))

    p_lip_top = get_pt(13)
    p_lip_bottom = get_pt(14)
    p_corner_r = get_pt(61)
    p_corner_l = get_pt(291)

    mouth_height = dist(p_lip_top, p_lip_bottom)
    mouth_width = dist(p_corner_r, p_corner_l)
    
    mouth_open_ratio = mouth_height / face_height
    mouth_width_ratio = mouth_width / face_width

    # Elevasi Sudut Bibir relatif terhadap garis tengah bibir
    lip_center_y = (p_lip_top[1] + p_lip_bottom[1]) / 2.0
    corner_lift_r = (lip_center_y - p_corner_r[1]) / face_height
    corner_lift_l = (lip_center_y - p_corner_l[1]) / face_height
    avg_corner_lift = (corner_lift_r + corner_lift_l) / 2.0
    max_corner_lift = max(corner_lift_r, corner_lift_l)

    # Mata (Eye Aspect Ratio)
    r_eye_h = dist(get_pt(159), get_pt(145))
    r_eye_w = max(1.0, dist(get_pt(133), get_pt(33)))
    right_ear = r_eye_h / r_eye_w

    l_eye_h = dist(get_pt(386), get_pt(374))
    l_eye_w = max(1.0, dist(get_pt(362), get_pt(263)))
    left_ear = l_eye_h / l_eye_w
    avg_ear = (right_ear + left_ear) / 2.0

    # 1. Kaget / Terbuka Lebar
    if mouth_open_ratio > 0.15 and avg_ear > 0.20:
        return "Kaget / Terbuka", (0, 215, 255)

    # 2. Tertawa (Mulut terbuka & melebar)
    if mouth_open_ratio > 0.06 and (mouth_width_ratio > 0.40 or avg_corner_lift > 0.008):
        return "Tertawa", (0, 255, 255)

    # 3. Tersenyum / Smirk (Sensitif terhadap senyum natural, tipis, maupun smirk satu sisi)
    if (
        (mouth_width_ratio > 0.385 and max_corner_lift > 0.004) or
        avg_corner_lift > 0.006 or
        max_corner_lift > 0.012 or
        mouth_width_ratio > 0.425
    ):
        return "Tersenyum", (0, 255, 0)

    # 4. Cemberut / Sedih
    if avg_corner_lift < -0.018:
        return "Cemberut / Sedih", (0, 140, 255)

    return "Netral", (255, 255, 255)


class FaceExpressionDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_rgb, w, h):
        mesh_results = self.face_mesh.process(frame_rgb)
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

        return face_expressions
