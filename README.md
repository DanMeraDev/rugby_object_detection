# 🏉 Rugby Object Detection - YOLOv8

This project uses **YOLOv8 (Ultralytics)** and **OpenCV** to detect and track objects in rugby match videos.  
It identifies players, referees, jersey numbers, and the ball — allowing color-based team distinction (white and black teams).

---

## 🚀 Features

- Detects **4 classes**:
  - 🏐 `ball`
  - 👕 `jersey number`
  - 🧍‍♂️ `player`
  - 🟢 `referee`
- Distinguishes players by **team color** (white or black jerseys).
- Processes any input video and outputs an annotated version.
- Real-time visualization during processing.
- Optionally saves the output as `result_rugby.mp4`.

---

## 🧠 Model Information

The model was trained using **YOLOv8n** with a custom dataset from [Roboflow](https://universe.roboflow.com/ian-muyala/rugby-kgrdw/dataset/3).

### Dataset structure (`dataset.yaml`)
```yaml
path: .
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4
names:
  0: ball
  1: jersey number
  2: player
  3: referee
