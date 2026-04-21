# 🎯 Object Detection & Tracking System

A computer vision project built with **YOLOv8** and **OpenCV**, covering real-time object detection, multi-object tracking, instance segmentation, and vehicle analytics — applied across live camera feeds, video files, and static images.

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
```
<img width="164" height="236" alt="image" src="https://github.com/user-attachments/assets/765aa4c1-949d-4364-ae9a-56867ddbc881" />

<img width="1000" height="735" alt="image" src="https://github.com/user-attachments/assets/b109904b-7478-4a81-8549-f2bcd6feb18b" />

<img width="602" height="469" alt="image" src="https://github.com/user-attachments/assets/671df0fb-4ae4-404f-9b6d-037893847c48" />

<img width="961" height="469" alt="image" src="https://github.com/user-attachments/assets/59c7ed7f-1207-45c6-a0ba-f11d19f1dcf5" />

<img width="1193" height="252" alt="image" src="https://github.com/user-attachments/assets/0b5f9de3-0d22-47f6-a6c0-9381faba81aa" />

---

## 🚀 Features

| Script | Capability |
|---|---|
| `live_detection.py` | Real-time object detection on live webcam feed |
| `video_detection.py` | Frame-by-frame detection on video files |
| `people_counter.py` | Tracks and counts unique individuals across frames |
| `person_tracker.py` | Assigns persistent IDs, draws motion trails, exports annotated video |
| `segmentation_tracker.py` | Pixel-level contour masks on tracked persons |
| `image_detection.py` | Single image inference with bounding box annotation |
| `vehicle_tracker.py` | Tracks cars, buses, trucks, motorcycles with motion history |

---

## 🧰 Libraries & Why I Used Them

### 🔷 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

**What it is:** State-of-the-art real-time object detection and tracking framework built on the YOLO (You Only Look Once) architecture.

**Why I used it:**
- YOLOv8 delivers an exceptional balance between speed and accuracy, making it ideal for real-time applications where every millisecond matters
- The `model.track()` API provides built-in multi-object tracking with persistent IDs across frames — no need to implement a separate tracker like SORT or DeepSORT from scratch
- The `yolov8n-seg.pt` segmentation variant enables pixel-level instance masks, going beyond simple bounding boxes to produce contour-level object boundaries
- Pre-trained on the COCO dataset (80 object classes), it works out-of-the-box for persons, vehicles, and common objects without any custom training

**Models used:**
- `yolov8n.pt` — Nano model (fastest, lightest) for detection tasks
- `yolov8m.pt` — Medium model (better accuracy) for tracking tasks requiring ID persistence
- `yolov8n-seg.pt` — Nano segmentation model for pixel-level masks

**Real-world use cases:** Retail footfall analytics, autonomous vehicles, security surveillance, sports performance analysis, industrial quality control

---

### 🔷 [OpenCV (`cv2`)](https://opencv.org/)

**What it is:** The most widely used open-source computer vision and image processing library, with over 2,500 algorithms.

**Why I used it:**
- Handles all video I/O: `cv2.VideoCapture` reads from webcams and video files with a single unified interface; `cv2.VideoWriter` exports processed frames to `.mp4`
- Provides essential drawing primitives (`cv2.rectangle`, `cv2.putText`, `cv2.circle`, `cv2.line`, `cv2.drawContours`) for annotating detection results
- `cv2.resize` with `INTER_AREA` interpolation gives high-quality downscaling of high-resolution video to manageable display sizes
- `cv2.findContours` extracts precise object boundaries from binary segmentation masks for contour-level rendering
- `cv2.imshow` provides a lightweight, real-time display window without needing a separate GUI framework

**Real-world use cases:** Medical imaging, robotics vision systems, augmented reality, satellite image processing, manufacturing defect detection

---

### 🔷 [NumPy](https://numpy.org/)

**What it is:** The foundational library for numerical computing in Python, enabling fast array and matrix operations.

**Why I used it:**
- YOLO returns bounding box coordinates, track IDs, and segmentation masks as NumPy arrays — converting with `.numpy()` is necessary to manipulate them with standard Python logic
- Used for efficient integer casting of bounding box coordinates (`map(int, box)`)
- Mask arrays from the segmentation model are NumPy arrays that must be scaled and converted to `uint8` before being passed to OpenCV's contour functions

**Real-world use cases:** Data science, scientific computing, image preprocessing pipelines, machine learning feature engineering

---

### 🔷 [collections.defaultdict & deque](https://docs.python.org/3/library/collections.html)

**What it is:** Built-in Python data structures from the `collections` module.

**Why I used them:**
- `defaultdict(int)` tracks how many times each object ID has appeared — used to filter out transient false positives by only assigning a display ID after an object appears for ≥5 consecutive frames (appearance gating)
- `defaultdict(lambda: deque(maxlen=30))` maintains a rolling motion trail of the last 30 center-point positions for each tracked object, automatically discarding older positions. This creates smooth motion path visualizations without unbounded memory growth
- Together they implement a lightweight tracklet management system without an external dependency

---

## 🧠 Key Technical Concepts Implemented

**Appearance Gating** — Objects are only assigned a stable display ID after being detected for 5+ consecutive frames, filtering noise and spurious detections.

**Motion Trail Visualization** — A sliding window of centroid positions is stored per track ID and rendered as a polyline, giving a real-time visual history of movement paths.

**Class Filtering** — The `classes` parameter in `model.track()` restricts inference to specific COCO classes (e.g., `[0]` for persons, `[2,3,5,7]` for vehicles), reducing computation and false positives.

**Scale-Aware Segmentation** — Inference runs on a downscaled frame for speed, then masks are upscaled back to the original resolution using nearest-neighbor interpolation to preserve mask boundaries.

**Unique ID Mapping** — Raw YOLO tracker IDs (which can be large or non-sequential) are remapped to compact sequential IDs for clean on-screen display.

---

