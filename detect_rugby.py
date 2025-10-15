from ultralytics import YOLO
import cv2
import numpy as np
import os

best_model_path = "runs/train/rugby_train/weights/best.pt"

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"No se encontró el modelo entrenado en: {best_model_path}")

model = YOLO(best_model_path)

video_path = "test_rugby.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"No se encontró el video {video_path}")

cap = cv2.VideoCapture(video_path)

output_path = "result_rugby.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Procesando video de Rugby...")

def get_team_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 60, 255])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])

    green_lower = np.array([40, 40, 40])
    green_upper = np.array([90, 255, 255])

    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    white_ratio = np.sum(mask_white) / (crop.shape[0]*crop.shape[1]*255)
    black_ratio = np.sum(mask_black) / (crop.shape[0]*crop.shape[1]*255)
    green_ratio = np.sum(mask_green) / (crop.shape[0]*crop.shape[1]*255)

    if white_ratio > 0.3:
        return 'white'
    elif black_ratio > 0.3:
        return 'black'
    elif green_ratio > 0.3:
        return 'green'
    else:
        return 'other'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = model.names

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[cls]
        color = (0, 255, 255)

        if label == "player":
            crop = frame[y1:y2, x1:x2]
            crop_top = crop[0:crop.shape[0]//2, :]
            team = get_team_color(crop_top)

            if team == 'white':
                color = (255, 255, 255)
                label = "Equipo Blanco"
            elif team == 'black':
                color = (0, 0, 0)
                label = "Equipo Negro"
            elif team == 'green':
                color = (0, 255, 0)
                label = "arbitro"

        elif label == "ball":
            color = (0, 165, 255)
            label = "Balón"
        elif label == "jersey number":
            color = (255, 0, 255)
            label = "Dorsal"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Rugby Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video procesado y guardado como: {output_path}")
