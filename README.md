# Citra15
## Richie Pranata
## 312410451
# 🚶 Pedestrian Detection Using OpenCV (HOG + SVM)

## 📖 Overview

This project implements **Pedestrian Detection** using **OpenCV-Python** with the **Histogram of Oriented Gradients (HOG)** feature descriptor and a pre-trained **Linear Support Vector Machine (SVM)** classifier.

The system is capable of detecting pedestrians in both **static images** and **video streams** by drawing bounding boxes around detected people.

---

## 🎯 Objectives

The objectives of this project are:

- Understand the concept of object detection using OpenCV.
- Implement pedestrian detection using HOG and SVM.
- Detect pedestrians from images and videos.
- Visualize detection results using bounding boxes.
- Learn how traditional computer vision techniques work before using deep learning methods.

---

# 🛠 Technologies Used

| Technology | Description |
|------------|-------------|
| Python | Main programming language |
| OpenCV | Computer vision library |
| Imutils | Image processing utility library |
| HOG Descriptor | Feature extraction method |
| Linear SVM | Pedestrian classification model |

---

# 📦 Requirements

Install the required libraries:

```bash
pip install opencv-python==3.4.2
pip install imutils==0.5.3
```

---

# 📂 Project Structure

```text
├── pedestrian_detection_image.py
├── pedestrian_detection_video.py
├── img.png
├── vid.mp4
└── README.md
```

---

# 📚 Theory

## What is OpenCV?

OpenCV (Open Source Computer Vision Library) is an open-source computer vision library used for image processing, object detection, facial recognition, motion tracking, and many other computer vision applications.

Official Website:

https://opencv.org

---

## What is HOG (Histogram of Oriented Gradients)?

HOG is a feature extraction technique that analyzes image gradients and edge directions.

The algorithm works by:

1. Examining pixel intensity changes.
2. Computing gradient directions.
3. Grouping gradients into histograms.
4. Creating a feature vector representing the object shape.

These features are then used by the classifier to determine whether a pedestrian exists in the image.

---

## What is SVM (Support Vector Machine)?

Support Vector Machine (SVM) is a machine learning classifier used to separate objects into different categories.

In this project:

- Input = HOG Features
- Output = Pedestrian or Non-Pedestrian

OpenCV provides a pre-trained pedestrian detector, so no additional model training is required.

---

# 🖼 Example 1: Pedestrian Detection on Images

File:

```text
pedestrian_detection_image.py
```

## Workflow

```text
Input Image
      │
      ▼
Resize Image
      │
      ▼
Extract HOG Features
      │
      ▼
Linear SVM Classification
      │
      ▼
Detect Pedestrians
      │
      ▼
Draw Bounding Boxes
      │
      ▼
Display Result
```

### Key Code

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

Initialize the HOG descriptor and load OpenCV's pre-trained pedestrian detector.

### Detection Process

```python
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4,4),
    padding=(4,4),
    scale=1.05
)
```

### Parameters

| Parameter | Value | Description |
|------------|--------|------------|
| winStride | (4,4) | Sliding window step size |
| padding | (4,4) | Additional border around detection window |
| scale | 1.05 | Image scaling factor |

---

# 🎥 Example 2: Pedestrian Detection on Video

File:

```text
pedestrian_detection_video.py
```

Unlike image detection, video detection processes frames continuously until the video ends or the user presses the **Q** key.

## Workflow

```text
Video Input
      │
      ▼
Read Frame
      │
      ▼
Resize Frame
      │
      ▼
Extract HOG Features
      │
      ▼
Detect Pedestrians
      │
      ▼
Draw Bounding Boxes
      │
      ▼
Display Frame
      │
      ▼
Next Frame
```

### Open Video

```python
cap = cv2.VideoCapture('vid.mp4')
```

### Read Frames

```python
while cap.isOpened():
    ret, image = cap.read()
```

### Exit Program

```python
if cv2.waitKey(25) & 0xFF == ord('q'):
    break
```

---

# 📊 Results

The program detects pedestrians and displays red bounding boxes around each detected person.

Example output:

```text
Detected: 3 persons
```

Displayed on the video frame in real time.

---

# ⚙ Advantages

- Simple implementation
- No model training required
- Fast detection on low-resource systems
- Suitable for educational purposes

---

# ⚠ Limitations

- Less accurate than modern deep learning models
- Sensitive to occlusions
- Performance decreases in crowded environments
- Struggles with unusual body poses

---

# 🚀 Future Improvements

The pedestrian detection system can be improved by using:

- YOLOv8
- Faster R-CNN
- SSD (Single Shot Detector)
- MobileNet-SSD

These deep learning models generally provide higher accuracy and better real-time performance.

---

# 🌍 Real-World Applications

Pedestrian detection is widely used in:

- Autonomous Vehicles
- Traffic Monitoring Systems
- Smart City Applications
- CCTV Surveillance
- Crowd Counting Systems
- Security Monitoring

---

# 👨‍💻 Author

Chapter 15 Assignment – Pedestrian Detection using OpenCV-Python

Computer Vision Project using HOG + Linear SVM.
