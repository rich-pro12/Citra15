"""
====================================================
 Deteksi Pejalan Kaki - Contoh 1: GAMBAR STATIS
 Metode: HOG (Histogram of Oriented Gradients) + Linear SVM
 Library: OpenCV, imutils
====================================================
"""

import cv2
import imutils

# ── 1. Inisialisasi HOG Descriptor + SVM Detector ──────────────────────────
# HOGDescriptor adalah deskriptor fitur yang mengekstrak gradien dari gambar
hog = cv2.HOGDescriptor()

# Menggunakan model SVM bawaan OpenCV yang sudah dilatih sebelumnya
# untuk mendeteksi pejalan kaki (manusia)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ── 2. Membaca Gambar Input ─────────────────────────────────────────────────
# Ganti 'img.png' dengan path gambar yang ingin dideteksi
image = cv2.imread('img.png')

if image is None:
    print("❌ Gambar tidak ditemukan! Pastikan file 'img.png' ada.")
    exit()

# ── 3. Resize Gambar ────────────────────────────────────────────────────────
# Mengubah ukuran gambar agar lebar maksimal 400px
# Tujuan: mempercepat proses deteksi tanpa kehilangan banyak detail
image = imutils.resize(image, width=min(400, image.shape[1]))

# ── 4. Deteksi Pejalan Kaki ─────────────────────────────────────────────────
# detectMultiScale mendeteksi objek pada berbagai skala
# Parameter:
#   winStride=(4,4)  → langkah geser jendela deteksi (piksel)
#   padding=(4,4)    → padding tambahan di sekitar jendela
#   scale=1.05       → faktor zoom setiap iterasi skala (1.05 = 5% lebih besar)
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(4, 4),
    scale=1.05
)

print(f"✅ Jumlah pejalan kaki terdeteksi: {len(regions)}")

# ── 5. Gambar Kotak Pembatas (Bounding Box) ─────────────────────────────────
# Setiap region adalah (x, y, w, h):
#   x, y = koordinat pojok kiri atas
#   w, h = lebar dan tinggi kotak
for (x, y, w, h) in regions:
    cv2.rectangle(
        image,
        (x, y),           # titik awal
        (x + w, y + h),   # titik akhir
        (0, 0, 255),       # warna merah (BGR)
        2                  # ketebalan garis
    )

# ── 6. Tampilkan Hasil ──────────────────────────────────────────────────────
cv2.imshow("Deteksi Pejalan Kaki - Gambar", image)
cv2.waitKey(0)            # tunggu tombol apapun ditekan
cv2.destroyAllWindows()
