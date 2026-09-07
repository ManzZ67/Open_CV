import cv2
import os
import torch
import numpy as np
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from src.utils import resolve_path


class FaceIdentitySystem:
    def __init__(self, identities_dir=None):
        if identities_dir is None:
            identities_dir = resolve_path(os.path.join("data", "identitas"), "identitas")
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
        face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_LANCZOS4)
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
                    img_work = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4) if scale != 1.0 else img
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
                for person_name, emb_list in self.known_embeddings.items():
                    for k_emb in emb_list:
                        sim = float(np.dot(emb, k_emb))
                        if sim > best_similarity:
                            best_similarity = sim
                            if sim >= 0.65:
                                best_match_name = person_name

            detected_faces.append({
                "box": (x1, y1, x2, y2),
                "name": best_match_name,
                "similarity": best_similarity
            })

        return detected_faces
