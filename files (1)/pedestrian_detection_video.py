"""
====================================================
 Deteksi Pejalan Kaki - Contoh 2: VIDEO
 Metode: HOG (Histogram of Oriented Gradients) + Linear SVM
 Library: OpenCV, imutils
====================================================
"""

import cv2
import imutils

# ── 1. Inisialisasi HOG Descriptor + SVM Detector ──────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ── 2. Membuka File Video ────────────────────────────────────────────────────
# Ganti 'vid.mp4' dengan path video Anda
# Untuk webcam langsung, ganti dengan: cv2.VideoCapture(0)
cap = cv2.VideoCapture('vid.mp4')

if not cap.isOpened():
    print("❌ Video tidak dapat dibuka! Pastikan file 'vid.mp4' ada.")
    exit()

print("✅ Video berhasil dibuka. Tekan 'q' untuk keluar.")

# ── 3. Loop Frame per Frame ─────────────────────────────────────────────────
while cap.isOpened():
    ret, image = cap.read()   # baca satu frame

    if ret:
        # ── 3a. Resize Frame ────────────────────────────────────────────────
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # ── 3b. Deteksi Pejalan Kaki pada Frame ─────────────────────────────
        (regions, _) = hog.detectMultiScale(
            image,
            winStride=(4, 4),
            padding=(4, 4),
            scale=1.05
        )

        # ── 3c. Gambar Bounding Box ──────────────────────────────────────────
        for (x, y, w, h) in regions:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),   # merah
                2
            )

        # Tampilkan jumlah deteksi di pojok kiri atas
        cv2.putText(
            image,
            f"Terdeteksi: {len(regions)} orang",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ── 3d. Tampilkan Frame ──────────────────────────────────────────────
        cv2.imshow("Deteksi Pejalan Kaki - Video", image)

        # Tekan 'q' untuk berhenti
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        # Tidak ada frame lagi → video selesai
        break

# ── 4. Bersihkan Resource ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("✅ Selesai.")
