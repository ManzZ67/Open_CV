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
  - **Tersenyum** $\rightarrow$ Otomatis menempelkan stiker **Roblox Man Face** ke wajah.
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
│       ├── absolute_cinema.jpg
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
├── 🚀 main.py                  # Skrip utama program
├── 📦 requirements.txt         # Daftar dependensi Python
├── 📄 .gitignore               # Konfigurasi filter Git
└── 📖 README.md                # Dokumentasi proyek
```

---

## 🛠️ Instalasi & Persiapan

### 1. Clone Repository
```bash
git clone https://github.com/ManzZ67/Open_CV.git
cd Open_CV
```

### 2. Buat Virtual Environment (Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

---

## 👥 Menambahkan Database Wajah

Untuk mendaftarkan wajah baru ke sistem Face Recognition:
1. Buka folder `data/identitas/`.
2. Buat folder baru dengan **Nama Orang** yang ingin didaftarkan:
   ```text
   data/identitas/Budi/
   ```
3. Masukkan 1 atau beberapa foto wajah yang jelas ke dalam folder tersebut (format `.jpg`, `.jpeg`, `.png`).
4. Saat program berjalan, Anda bisa menekan tombol **`r`** di keyboard untuk memuat ulang database tanpa perlu restart program!

---

## 🚀 Cara Menjalankan

### Opsi 1: Menggunakan Webcam Bawaan Laptop
```bash
python main.py
```

### Opsi 2: Menggunakan DroidCam PC Client (USB / WiFi)
1. Buka aplikasi DroidCam di HP & PC, lalu klik **Start**.
2. Jalankan:
   ```bash
   python main.py -s 1
   ```
   *(Ganti `1` dengan `2` jika DroidCam terdeteksi di index 2).*

### Opsi 3: Menggunakan DroidCam Direct IP (WiFi)
1. Pastikan HP dan Laptop terhubung ke WiFi / Hotspot yang sama.
2. Catat IP yang tertera di aplikasi DroidCam HP (misal: `192.168.1.15`).
3. Jalankan:
   ```bash
   python main.py -i 192.168.1.15
   ```

---

## 🎭 Interaksi Gesture & Meme

| Gesture / Aksi | Trigger | Hasil Tampilan |
| :--- | :--- | :--- |
| **Dua Tangan Terangkat (10 Jari)** | Angkat 2 telapak tangan terbuka ke kamera | 🎬 Mode **Absolute Cinema** & Logo Cinema muncul |
| **Wajah Tersenyum** | Senyum lebar terdeteksi oleh Face Mesh | 🗿 Overlay stiker **Roblox Man Face** menutupi wajah |
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
| **`f`** | **Fullscreen**: Mengaktifkan / menonaktifkan tampilan layar penuh |
| **`q`** / **`ESC`** | **Keluar**: Menutup aplikasi |

---

## 🤝 Kontribusi

Kontribusi, *issues*, dan *pull requests* sangat dipersilakan! Silakan fork repository ini dan buat pull request terbaik Anda.

---

<div align="center">
  Dibuat dengan ❤️ oleh <a href="https://github.com/ManzZ67"><strong>ManzZ67</strong></a>
</div>
