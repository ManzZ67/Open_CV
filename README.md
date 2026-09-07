<div align="center">

# 🎭 VisionVerse: Interactive AI Computer Vision
**Real-Time Face Recognition, Expression Analysis, Hand Gesture Tracking & Dynamic Meme Overlays**

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-FaceNet-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-00C4B4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br />

<p align="center">
  A smart & fun Computer Vision project featuring deep-learning Face Recognition (FaceNet), Facial Expression Detection (MediaPipe Face Mesh), Multi-Hand Gesture Tracking, and interactive meme overlays (Absolute Cinema, Roblox Man Face, etc.).
</p>

[Fitur Utama](#-fitur-utama) •
[Struktur Proyek](#-struktur-folder--proyek) •
[Instalasi](#-instalasi--persiapan) •
[Cara Menjalankan](#-cara-menjalankan) •
[Gesture & Meme](#-interaksi-gesture--meme) •
[Hotkeys](#-kontrol-hotkeys)

---

</div>

## ✨ Fitur Utama

- 🧠 **Face Identity Recognition**: Mengenali wajah berdasarkan database lokal menggunakan Inception-ResNet-v1 (FaceNet) pretrained VGGFace2 dengan Cosine Similarity thresholding.
- 😊 **Ekspresi Wajah Real-Time**: Mendeteksi senyum, ekspresi cemberut/terkejut/netral melalui 468 landmark MediaPipe Face Mesh.
- 🖐️ **Dual Hand Tracking**: Melacak hingga 2 tangan sekaligus dan menghitung orientasi jari secara presisi.
- 🎬 **Meme Triggers & Overlays**:
  - **Dua Tangan Terangkat (10 Jari)** $\rightarrow$ Memunculkan efek watermark & logo **Absolute Cinema**!
  - **Tersenyum** $\rightarrow$ Otomatis menempelkan stiker **Roblox Man Face** ke layar.
  - **Pose Jari Tengah** $\rightarrow$ Respons teks interaktif.
- 📱 **Dukungan DroidCam**: Bisa menggunakan kamera HP baik melalui **Client Virtual Cam USB/WiFi** maupun **Direct IP Stream**.
- ⚡ **GPU Accelerated**: Mendukung CUDA / GPU NVIDIA bila tersedia untuk performa embedding instan.

---

## 📁 Struktur Folder & Proyek

Proyek ini telah dirancang dengan arsitektur folder yang rapi dan modular:

```text
Open_CV/
│
├── 🎨 assets/
│   └── images/                 # Aset gambar: logo watermark, stiker meme
│       ├── cinema_logo.png
│       └── meme_senyum.jpg
│
├── 📂 data/
│   └── identitas/              # Database identitas wajah
│       └── <Nama Orang>/       # Folder per orang (berisi 1 atau lebih foto)
│           ├── foto_1.jpg
│           └── foto_2.jpg
│
├── 🧠 models/
│   └── yolov8n.pt              # Model weights pre-trained
│
├── 💾 outputs/
│   └── output_test.png         # Penyimpanan hasil capture / export frame
│
├── 📁 src/                     # MODUL PROGRAM TERPISAH
│   ├── __init__.py
│   ├── face_recognition.py     # Modul FaceNet & MediaPipe Face Identity
│   ├── face_expression.py      # Modul MediaPipe Face Mesh Expressions
│   ├── hand_tracking.py        # Modul MediaPipe Hand Tracking & Gestures
│   ├── ar_math_drag.py         # Modul Game AR Math Drag & Drop Balok Angka
│   ├── math_game.py            # Modul logika kuis matematika & game
│   ├── ar_filters.py           # Modul filter AR (Air Canvas, Face Visor, 3D Cube)
│   └── utils.py                # Fungsi bantuan matematika, alpha overlay & HUD
│
├── 🚀 main.py                  # Skrip utama Face AI & Meme Overlays
├── 🧩 ar_math_game.py          # Skrip AR Math Game (Pinch & Drag Balok ke Slot)
├── 🎮 math_game.py             # Skrip Game Kuis Matematika berbasis Gerakan Jari
├── ✨ ar_experience.py         # Skrip Augmented Reality (AR) Studio Multi-Mode
├── 📦 requirements.txt         # Daftar dependensi Python
├── 📄 .gitignore               # Konfigurasi filter Git
└── 📖 README.md                # Dokumentasi proyek
```

---

## 🚀 Daftar Program & Cara Menjalankan

Berikut adalah seluruh program Python yang tersedia di repository ini dan cara menjalankannya:

| Program / Skrip | Fungsi & Deskripsi | Perintah Menjalankan |
| :--- | :--- | :--- |
| **🎯 `ar_math_game.py`** | **Game AR Matematika (TikTok Style)**: Cubit (*pinch*) balok angka melayang, geser (*drag*), dan taruh (*drop*) ke slot jawaban `?`. Dilengkapi fitur *Super-Sticky Drag* (tahan halangan wajah). | `python ar_math_game.py` |
| **🎭 `main.py`** | **Sistem Utama AI Computer Vision**: Face Identity Recognition (FaceNet), deteksi senyum & emosi, meme overlays (*Absolute Cinema*, *Roblox Man Face*), dan hand gestures. | `python main.py` |
| **🎮 `math_game.py`** | **Game Kuis Matematika Jari**: Menjawab soal matematika dengan mengangkat jari tangan (0 - 10 jari) dengan sistem nyawa & streak combo. | `python math_game.py` |
| **✨ `ar_experience.py`** | **Augmented Reality Studio**: 3 mode interaktif — *AR Air Canvas* (melukis di udara), *Face Cyber Visor & Crown*, dan *3D Hologram Cube*. | `python ar_experience.py` |

---

### 📷 Pilihan Sumber Kamera (Webcam vs DroidCam):

Setiap program di atas mendukung argumen kamera berikut:

1. **Webcam Laptop / Kamera Bawaan (Default):**
   ```bash
   python ar_math_game.py
   # atau
   python main.py
   ```

2. **Kamera HP melalui DroidCam WiFi (Direct IP):**
   ```bash
   python ar_math_game.py -i 192.168.1.15
   # atau
   python main.py -i 192.168.1.15
   ```
   *(Ganti `192.168.1.15` dengan IP yang tertera pada aplikasi DroidCam di HP Anda)*

3. **Kamera Eksternal / DroidCam Client (Index Device):**
   ```bash
   python ar_math_game.py -s 1
   # atau
   python main.py -s 1
   ```

---

## 🧩 Detail Game AR Matematika (`ar_math_game.py`)

Game matematika interaktif Augmented Reality persis seperti video viral di TikTok / Claude:

```bash
python ar_math_game.py
```

### 🎯 Cara Bermain:
1. Kotak persamaan matematika muncul di layar bawah: `[ 10 ] [ + ] [ 6 ] [ = ] [ ? ]`.
2. Balok-balok angka melayang di udara di bagian kanan atas.
3. **Cubit (*Pinch*)** jempol dan telunjuk pada balok angka untuk memegangnya.
4. **Geser (*Drag*)** tangan ke slot kosong yang bertanda `?`.
5. **Lepas cubitan (*Drop*)** untuk menaruh angka. Jika persamaan benar, Anda naik ke **Tahap Berikutnya** dan skor bertambah `+50`!
6. *Anti-Mantul:* Balok tetap menempel stabil di tangan meskipun tangan melewati area wajah (*face occlusion buffer* & *hysteresis lock*).

---

## 🎭 Interaksi Gesture & Meme

| Gesture / Aksi | Trigger | Hasil Tampilan |
| :--- | :--- | :--- |
| **Dua Tangan Terangkat (10 Jari)** | Angkat 2 telapak tangan terbuka ke kamera | 🎬 Mode **Absolute Cinema** & Logo Cinema muncul di bawah layar |
| **Wajah Tersenyum** | Senyum lebar terdeteksi oleh Face Mesh | 🗿 Overlay stiker **Roblox Man Face** muncul di pojok atas |
| **Pose Jari Tengah (Middle Finger)** | Angkat jari tengah saja | ⚠️ Notifikasi teks interaktif di layar |
| **Identitas Dikenal** | Wajah cocok dengan dataset di `data/identitas/` | 🏷️ Bounding box hijau + Nama & Similarity % |
| **Identitas Tidak Dikenal** | Wajah baru / belum terdaftar | ❓ Bounding box merah + *"Tidak Dikenal"* |

---

## ⌨️ Kontrol Hotkeys

Saat jendela kamera aktif, gunakan tombol keyboard berikut:

| Tombol | Fungsi |
| :---: | :--- |
| **`m`** | **Toggle Mirror**: Membalik tampilan kamera (Flip Horizontal On / Off) |
| **`r`** | **Reload Database**: Memuat ulang folder `data/identitas/` secara instan |
| **`s`** | **Screenshot**: Mengambil foto tangkapan layar HD (disimpan ke `outputs/`) |
| **`f`** | **Fullscreen**: Mengaktifkan / menonaktifkan tampilan layar penuh |
| **`q`** / **`ESC`** | **Keluar**: Menutup aplikasi |

---

## 🤝 Kontribusi

Kontribusi, *issues*, dan *pull requests* sangat dipersilakan! Silakan fork repository ini dan buat pull request terbaik Anda.

---

<div align="center">
  Dibuat dengan ❤️ oleh <a href="https://github.com/ManzZ67"><strong>ManzZ67</strong></a>
</div>
