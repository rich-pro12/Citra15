# Tugas Chapter 15 – Deteksi Pejalan Kaki dengan OpenCV-Python

## Ringkasan

Program ini mengimplementasikan **Deteksi Pejalan Kaki** menggunakan pustaka OpenCV dengan metode **HOG (Histogram of Oriented Gradients) + Linear SVM** yang telah dilatih sebelumnya.

---

## Konsep Utama

### Apa itu OpenCV?
OpenCV (*Open Source Computer Vision*) adalah pustaka sumber terbuka untuk **visi komputer waktu nyata**. Dikembangkan oleh Intel, bersifat lintas platform (mendukung Python, C++, Java, dll). Digunakan untuk tugas seperti pengenalan wajah, deteksi gerakan, dan deteksi objek.

### Apa itu HOG (Histogram of Oriented Gradients)?
HOG adalah algoritma untuk **mengekstrak fitur** dari sebuah gambar dengan cara:
1. Memeriksa setiap piksel dibandingkan piksel di sekitarnya
2. Menentukan seberapa gelap piksel tersebut relatif terhadap tetangganya
3. Menggambar **panah (gradien)** yang menunjukkan arah dari terang ke gelap
4. Gradien-gradien ini kemudian dianalisis lebih lanjut oleh SVM

### Apa itu SVM dalam konteks ini?
**Linear SVM (Support Vector Machine)** digunakan sebagai classifier yang sudah dilatih dengan data pejalan kaki. Model ini menerima fitur HOG sebagai input dan memutuskan apakah suatu wilayah gambar mengandung pejalan kaki atau tidak.

---

## Persyaratan (Requirements)

| Library | Versi | Cara Install |
|---------|-------|--------------|
| `opencv-python` | 3.4.2 | `pip install opencv-python==3.4.2` |
| `imutils` | 0.5.3 | `pip install imutils==0.5.3` |

---

## Contoh 1: Deteksi pada Gambar Statis

**File:** `pedestrian_detection_image.py`

```python
import cv2
import imutils

# 1. Inisialisasi HOG Descriptor + default People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. Baca dan resize gambar
image = cv2.imread('img.png')
image = imutils.resize(image, width=min(400, image.shape[1]))

# 3. Deteksi pejalan kaki
(regions, _) = hog.detectMultiScale(image,
                                     winStride=(4, 4),
                                     padding=(4, 4),
                                     scale=1.05)

# 4. Gambar bounding box merah untuk setiap deteksi
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 5. Tampilkan hasil
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Penjelasan Parameter `detectMultiScale`

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `winStride` | `(4, 4)` | Langkah geser jendela deteksi dalam piksel (lebih kecil = lebih lambat tapi lebih akurat) |
| `padding` | `(4, 4)` | Padding di luar jendela deteksi (membantu mendeteksi objek di tepi) |
| `scale` | `1.05` | Faktor zoom antar iterasi – 1.05 artinya gambar diperkecil 5% setiap iterasi |

---

## Contoh 2: Deteksi pada Video

**File:** `pedestrian_detection_video.py`

```python
import cv2
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('vid.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image,
                                             winStride=(4, 4),
                                             padding=(4, 4),
                                             scale=1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```

### Perbedaan dengan Contoh 1

| Aspek | Gambar | Video |
|-------|--------|-------|
| Input | `cv2.imread()` | `cv2.VideoCapture()` |
| Loop | Tidak ada | `while cap.isOpened()` |
| Tampil | `cv2.waitKey(0)` – tunggu selamanya | `cv2.waitKey(25)` – tunggu 25ms per frame |
| Keluar | Tombol apapun | Tekan `'q'` |
| Tutup | `cv2.destroyAllWindows()` | `cap.release()` + `destroyAllWindows()` |

---

## Alur Kerja Program

```
Gambar/Video Input
        │
        ▼
   Resize Gambar (maks 400px lebar)
        │
        ▼
   HOG Descriptor (ekstrak fitur gradien)
        │
        ▼
   Linear SVM (klasifikasi: pejalan kaki / bukan)
        │
        ▼
   detectMultiScale (cari di berbagai skala)
        │
        ▼
   Gambar Bounding Box Merah
        │
        ▼
   Tampilkan Hasil
```

---

## Penerapan di Dunia Nyata

Deteksi pejalan kaki sangat penting untuk:
- 🚗 **Mobil Otonom** – sistem perlindungan pejalan kaki
- 📹 **CCTV & Pengawasan** – menghitung jumlah orang
- 🚦 **Smart City** – manajemen lalu lintas pejalan kaki

---

## Catatan Tambahan

- Model HOG + SVM sudah dilatih sebelumnya oleh OpenCV — tidak perlu melatih ulang
- Untuk akurasi lebih tinggi, dapat digunakan model deep learning seperti YOLO atau Faster R-CNN
- `imutils` digunakan untuk mempermudah operasi seperti resize yang mempertahankan aspek rasio
