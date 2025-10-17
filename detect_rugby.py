# detect_rugby_analytics.py
from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
import json
from sort import Sort  # ensure sort.py is available in the same folder

# Optional OCR for jersey numbers (install easyocr if you want OCR)
try:
    import easyocr
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR loaded for jersey recognition.")
except Exception:
    ocr_reader = None
    print("EasyOCR not available — jersey numbers will be None unless you install easyocr.")

# ==========================
# CARGA DEL MODELO ENTRENADO
# ==========================
best_model_path = "runs/train/rugby_train/weights/best.pt"
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"No se encontró el modelo entrenado en: {best_model_path}")

model = YOLO(best_model_path)

# ==========================
# VIDEO A PROCESAR
# ==========================
video_path = "test_rugby.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"No se encontró el video {video_path}")

cap = cv2.VideoCapture(video_path)
output_path = "result_rugby_teams_tracking_with_analytics.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ==========================
# TRACKER Y ANALYTICS
# ==========================
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
analytics_rows = []

# ==========================
# FUNCIÓN DE DETECCIÓN DE EQUIPO
# ==========================
def get_team_color(crop):
    """Detecta el equipo según color de camiseta, optimizado para diferenciar negro y verde."""
    if crop is None or crop.size == 0:
        return 'other'
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return 'other'
    top = crop[0:int(h*0.7), int(w*0.1):int(w*0.9)]
    if top is None or top.size == 0:
        return 'other'
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    # Rangos HSV
    white_lower = np.array([0, 0, 180]); white_upper = np.array([180, 60, 255])
    black_lower = np.array([0, 0, 0]);   black_upper = np.array([180, 255, 50])
    dark_green_lower = np.array([40, 40, 20])   # saturación mínima para evitar confundir negro
    dark_green_upper = np.array([80, 120, 130])
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_green = cv2.inRange(hsv, dark_green_lower, dark_green_upper)
    mask_green[hsv[:, :, 2] > 100] = 0  # ignorar césped brillante
    total_pixels = top.shape[0] * top.shape[1]
    white_ratio = np.sum(mask_white > 0) / total_pixels
    black_ratio = np.sum(mask_black > 0) / total_pixels
    green_ratio = np.sum(mask_green > 0) / total_pixels
    if white_ratio > 0.25:
        return 'referee'
    elif black_ratio > 0.25:
        return 'black'
    elif green_ratio > 0.02:
        return 'green'
    else:
        return 'other'

# ==========================
# UTIL: OCR de dorsal (si available)
# ==========================
def read_jersey_number_from_crop(crop):
    """Try to read a number from the given crop using easyocr (if available).
       Returns a string number or None.
    """
    if ocr_reader is None:
        return None
    try:
        # Preprocess: convert to grayscale and resize to improve OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 10 or w < 10:
            return None
        scale = max(1, 200 // max(w, h))
        resized = cv2.resize(gray, (w * scale, h * scale))
        results = ocr_reader.readtext(resized)
        # pick the best result that's mostly digits
        best_num = None
        best_conf = -1
        for (_bbox, text, conf) in results:
            t = text.strip()
            # remove non-digit chars
            digits = ''.join(ch for ch in t if ch.isdigit())
            if digits != "" and conf > best_conf:
                best_num = digits
                best_conf = conf
        return best_num
    except Exception:
        return None

# ==========================
# MAIN LOOP
# ==========================
show_debug = False
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame)
    dets = []
    det_boxes = []     # keep boxes to map indexes to coordinates
    det_cls_map = {}   # detection index -> class index

    # Build detection list for SORT (without class info)
    for i, (box, cls, score) in enumerate(zip(results[0].boxes.xyxy.cpu().numpy(),
                                              results[0].boxes.cls.cpu().numpy(),
                                              results[0].boxes.conf.cpu().numpy())):
        x1, y1, x2, y2 = box
        dets.append([x1, y1, x2, y2, float(score)])
        det_boxes.append([x1, y1, x2, y2])
        det_cls_map[i] = int(cls)

    # run tracker
    tracks = tracker.update(np.array(dets) if len(dets) > 0 else np.empty((0,5)))

    # gather jersey detections separately (indices and boxes)
    jersey_detections = []
    jersey_class_index = None
    # try to find which class index is "jersey number" for this model, if exists
    for idx, name in model.names.items():
        if 'jersey' in name.lower() or 'number' in name.lower() or 'dorsal' in name.lower():
            jersey_class_index = idx
            break

    for i, cls in det_cls_map.items():
        if cls == jersey_class_index:
            # store jersey bbox and index to OCR later
            x1, y1, x2, y2 = det_boxes[i]
            jersey_detections.append((i, (int(x1), int(y1), int(x2), int(y2))))

    # Build track_info and map each detection to nearest track when needed
    ball_pos = None
    players = {}   # track_id -> dict with fields below
    # tracks returned shape: [x1,y1,x2,y2,track_id]
    for t in tracks:
        if len(t) < 5:  # safety
            continue
        x1, y1, x2, y2, track_id = t
        # center
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        # find detection index closest to this track center (to map class)
        best_i = None
        best_dist = float('inf')
        for i, box in enumerate(det_boxes):
            bx1, by1, bx2, by2 = box
            bcx = (bx1 + bx2) / 2.0
            bcy = (by1 + by2) / 2.0
            d = (bcx - cx)**2 + (bcy - cy)**2
            if d < best_dist:
                best_dist = d
                best_i = i
        cls = det_cls_map.get(best_i, None)
        label = model.names[cls] if cls is not None else "unknown"
        if label not in ["ball", "jersey number"]:
            label = "player"
        if label == "ball":
            ball_pos = (cx, cy)
        # store
        players[int(track_id)] = {
            "track_id": int(track_id),
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "center": (cx, cy),
            "label": label,
            "team": None,      # to fill below
            "jersey": None,    # to fill below if found
        }

    # Associate jersey detections to players by nearest center, and OCR them
    # For each jersey bbox detected by YOLO, attempt OCR and map to nearest player center
    for (det_idx, (jx1, jy1, jx2, jy2)) in jersey_detections:
        # crop jersey area
        jx1c = max(0, int(jx1)); jy1c = max(0, int(jy1))
        jx2c = min(frame.shape[1], int(jx2)); jy2c = min(frame.shape[0], int(jy2))
        if jx2c <= jx1c or jy2c <= jy1c:
            continue
        jcrop = frame[jy1c:jy2c, jx1c:jx2c]
        number = read_jersey_number_from_crop(jcrop)
        # find nearest player center
        best_pid = None; best_d = float('inf')
        for pid, p in players.items():
            pcx, pcy = p['center']
            d = (pcx - (jx1+jx2)/2.0)**2 + (pcy - (jy1+jy2)/2.0)**2
            if d < best_d:
                best_d = d; best_pid = pid
        if best_pid is not None:
            # store jersey if found
            if number is not None:
                players[best_pid]['jersey'] = number
            else:
                # if OCR not available, mark approx present (YOLO found a jersey box)
                players[best_pid]['jersey'] = players[best_pid].get('jersey', None)

    # Determine team for each player (use get_team_color on bbox crop)
    for pid, p in list(players.items()):
        x1, y1, x2, y2 = p['bbox']
        x1c, y1c = max(0, int(round(x1))), max(0, int(round(y1)))
        x2c, y2c = min(frame.shape[1], int(round(x2))), min(frame.shape[0], int(round(y2)))
        if x2c <= x1c or y2c <= y1c:
            players[pid]['team'] = 'other'
            continue
        crop = frame[y1c:y2c, x1c:x2c]
        team = get_team_color(crop)
        players[pid]['team'] = team

    # Now compute nearest 3 players for every player (based on current centers)
    player_ids = list(players.keys())
    for pid in player_ids:
        pcenter = players[pid]['center']
        # compute distances to others
        others = []
        for oid in player_ids:
            if oid == pid:
                continue
            ocenter = players[oid]['center']
            d = ((pcenter[0] - ocenter[0])**2 + (pcenter[1] - ocenter[1])**2)**0.5
            others.append((oid, d))
        others.sort(key=lambda x: x[1])
        nearest_3 = others[:3]
        nearest_ids = [int(x[0]) for x in nearest_3]
        nearest_dists = [float(round(x[1], 2)) for x in nearest_3]
        players[pid]['nearest_ids'] = nearest_ids
        players[pid]['nearest_dists'] = nearest_dists

    # Prepare CSV rows and draw on frame
    for pid, p in players.items():
        cx, cy = p['center']
        x1, y1, x2, y2 = p['bbox']
        team = p.get('team', 'other')
        jersey = p.get('jersey', None)
        label = p.get('label', 'player')

        # Si es un jersey number (detectado por YOLO directamente)
        if label == "jersey number":
            color = (180, 0, 180)  # morado
            display_label = "Jersey Number" if jersey in [None, ""] else f"#{jersey}"
            cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
            cv2.putText(frame, display_label, (int(round(x1)), max(10, int(round(y1)) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            continue  # no seguimos con distancia, equipos, etc. para estos

        # distance to ball
        dist_ball = None
        if ball_pos is not None:
            dist_ball = float(round(((cx - ball_pos[0]) ** 2 + (cy - ball_pos[1]) ** 2) ** 0.5, 2))
        else:
            dist_ball = None

        # nearest
        nearest_ids = p.get('nearest_ids', [])
        nearest_dists = p.get('nearest_dists', [])

        # append analytics row
        analytics_rows.append([
            frame_idx,
            int(pid),
            team,
            jersey if jersey is not None else "",
            float(round(cx, 2)),
            float(round(cy, 2)),
            dist_ball if dist_ball is not None else "",
            json.dumps(nearest_ids),
            json.dumps(nearest_dists)
        ])

        # Elegir color y texto por equipo
        if team == 'referee':
            color = (255, 255, 255)
            display_label = f"Ref {pid}" if jersey in [None, ""] else f"Ref {pid} #{jersey}"
        elif team == 'black':
            color = (0, 0, 0)
            display_label = f"{pid} (Black)" if jersey in [None, ""] else f"{pid} #{jersey}"
        elif team == 'green':
            color = (0, 255, 0)
            display_label = f"{pid} (Green)" if jersey in [None, ""] else f"{pid} #{jersey}"
        else:
            color = (0, 255, 255)
            display_label = f"{pid} (Other)" if jersey in [None, ""] else f"{pid} #{jersey}"

        # Dibujar
        cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
        cv2.putText(frame, display_label, (int(round(x1)), max(10, int(round(y1)) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dibujar línea al balón
        if ball_pos is not None:
            cv2.line(frame, (int(round(cx)), int(round(cy))),
                     (int(round(ball_pos[0])), int(round(ball_pos[1]))),
                     (200, 200, 0), 1)

        # draw bounding box + label
        if team == 'referee':
            color = (255, 255, 255)
            display_label = f"Ref {pid}" if jersey in [None, ""] else f"Ref {pid} #{jersey}"
        elif team == 'black':
            color = (0, 0, 0)
            display_label = f"{pid} (Black)" if jersey in [None, ""] else f"{pid} #{jersey}"
        elif team == 'green':
            color = (0, 255, 0)
            display_label = f"{pid} (Green)" if jersey in [None, ""] else f"{pid} #{jersey}"
        else:
            color = (0, 255, 255)
            display_label = f"{pid} (Other)" if jersey in [None, ""] else f"{pid} #{jersey}"
        # rectangle
        cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
        cv2.putText(frame, display_label, (int(round(x1)), max(10, int(round(y1))-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # draw line to ball (if exists)
        if ball_pos is not None:
            cv2.line(frame, (int(round(cx)), int(round(cy))), (int(round(ball_pos[0])), int(round(ball_pos[1]))),
                     (200, 200, 0), 1)

    # Draw ball (if detected as a track or detection)
    # If ball was detected as a track we already used its center; draw a small circle at ball_pos:
    if ball_pos is not None:
        cv2.circle(frame, (int(round(ball_pos[0])), int(round(ball_pos[1]))), 6, (0, 165, 255), -1)
        cv2.putText(frame, "Ball", (int(round(ball_pos[0]))+6, int(round(ball_pos[1]))+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

    # show debug/hot overlay for green if user wants
    if show_debug:
        hsv_debug = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark_green_lower = np.array([40, 40, 20])
        dark_green_upper = np.array([80, 120, 130])
        mask_green = cv2.inRange(hsv_debug, dark_green_lower, dark_green_upper)
        mask_green[hsv_debug[:, :, 2] > 100] = 0
        green_overlay = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
        debug_frame = cv2.addWeighted(frame, 0.7, green_overlay, 0.3, 0)
        cv2.imshow("Rugby Detection - Teams Tracking (debug)", debug_frame)
        out.write(debug_frame)
    else:
        cv2.imshow("Rugby Detection - Teams Tracking", frame)
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_debug = not show_debug

# End main loop --------------------------------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

# ==========================
# SAVE CSV (final)
# ==========================
df = pd.DataFrame(analytics_rows, columns=[
    "frame", "id", "team", "jersey", "x_center", "y_center", "dist_to_ball", "nearest_ids", "nearest_dists"
])
df.to_csv("rugby_analytics_full.csv", index=False)
print("✅ Done — CSV saved as: rugby_analytics_full.csv")
        