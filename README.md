
---

## 📁 Project Structure

```
object_detection/
├── live_detection.py          # Real-time detection via webcam
├── video_detection.py         # Detection on pre-recorded video
├── people_counter.py          # Unique person counting with ID tracking
├── person_tracker.py          # Person tracking with trail visualization & video export
├── segmentation_tracker.py    # Instance segmentation + tracking (pixel-level masks)
├── image_detection.py         # Static image detection
└── vehicle_tracker.py         # Vehicle tracking with motion trails & video export
├── live_detection         # Real-time detection via webcam
├── video_detection       # Detection on pre-recorded video
├── people_counter         # Unique person counting with ID tracking
├── person_tracker          # Person tracking with trail visualization & video export
├── segmentation_tracker    # Instance segmentation + tracking (pixel-level masks)
├── image_detection       # Static image detection
└── vehicle_tracker.       # Vehicle tracking with motion trails & video export
```
<img width="164" height="236" alt="image" src="https://github.com/user-attachments/assets/765aa4c1-949d-4364-ae9a-56867ddbc881" />

<img width="1200" height="935" alt="image" src="https://github.com/user-attachments/assets/b109904b-7478-4a81-8549-f2bcd6feb18b" />
<img width="1000" height="735" alt="image" src="https://github.com/user-attachments/assets/b109904b-7478-4a81-8549-f2bcd6feb18b" />

<img width="602" height="469" alt="image" src="https://github.com/user-attachments/assets/671df0fb-4ae4-404f-9b6d-037893847c48" />

@@ -34,13 +31,13 @@ object_detection/

| Script | Capability |
|---|---|
| `live_detection.py` | Real-time object detection on live webcam feed |
| `video_detection.py` | Frame-by-frame detection on video files |
| `people_counter.py` | Tracks and counts unique individuals across frames |
| `person_tracker.py` | Assigns persistent IDs, draws motion trails, exports annotated video |
| `segmentation_tracker.py` | Pixel-level contour masks on tracked persons |
| `image_detection.py` | Single image inference with bounding box annotation |
| `vehicle_tracker.py` | Tracks cars, buses, trucks, motorcycles with motion history |
| `live_detection` | Real-time object detection on live webcam feed |
| `video_detection` | Frame-by-frame detection on video files |
| `people_counter` | Tracks and counts unique individuals across frames |
| `person_tracker` | Assigns persistent IDs, draws motion trails, exports annotated video |
| `segmentation_tracker` | Pixel-level contour masks on tracked persons |
| `image_detection` | Single image inference with bounding box annotation |
| `vehicle_tracker` | Tracks cars, buses, trucks, motorcycles with motion history |

---

@@ -118,86 +115,3 @@ object_detection/

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

```bash
# Live webcam detection
python live_detection.py

# Detection on a video file
python video_detection.py

# Count unique people in a scene
python people_counter.py

# Track persons and export annotated video
python person_tracker.py

# Instance segmentation with tracking
python segmentation_tracker.py

# Detect objects in a static image
python image_detection.py

# Track vehicles and export video
python vehicle_tracker.py
```

Press **`q`** to quit any running script.

---

## 📋 Requirements

- Python 3.8+
- Webcam (for `live_detection.py`)
- Input video files: `street.mp4`, `traffic.mp4`
- Input image: `image1.jpg`

---

## 🏗️ Architecture Overview

```
Video/Camera Input
       │
       ▼
  cv2.VideoCapture
       │
       ▼
 Frame Preprocessing
 (resize, scale)
       │
       ▼
  YOLOv8 Inference
  (detect / track / segment)
       │
       ├──► Bounding Boxes
       ├──► Track IDs
       └──► Segmentation Masks
                │
                ▼
        Annotation Layer
        (OpenCV drawing)
                │
                ▼
     Display / Export (mp4)
```

---

## 💡 Potential Extensions

- Add speed estimation using frame timestamps and real-world scale calibration
- Integrate a line-crossing counter for entry/exit analytics
- Add re-identification (Re-ID) features for cross-camera tracking
- Export detection logs to CSV or a database for downstream analytics
- Deploy as a web stream using FastAPI + WebSockets
