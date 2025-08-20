import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Configuration
# -----------------------------
VIDEO_PATH = r"E:\vehicle_speed_detection\test_1.mp4"  # local video
MODEL_PATH = "yolo11n.pt"  # YOLO11 model
FPS = 30  # video FPS (adjust to actual video)
METERS_PER_PIXEL = 0.05  # scale (adjust for real scene)

# COCO Class Names (only first 15, extend if needed)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorbike",
    5: "bus", 7: "truck"
}

# -----------------------------
# Initialize model and tracker
# -----------------------------
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30, n_init=3)
track_history = {}

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model.predict(frame, verbose=False)[0]
    detections = []

    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        cls = int(cls)
        if cls in COCO_CLASSES:  # filter only vehicles/persons
            x1, y1, x2, y2 = map(int, box)
            conf = float(score)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls)))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        w, h = r - l, b - t
        x_center = int(l + w/2)
        y_center = int(t + h/2)

        # Speed calculation
        if track_id in track_history:
            prev_x, prev_y = track_history[track_id]
            dx, dy = x_center - prev_x, y_center - prev_y
            distance_pixels = np.sqrt(dx**2 + dy**2)
            speed_mps = distance_pixels * METERS_PER_PIXEL * FPS
            speed_kmh = speed_mps * 3.6
        else:
            speed_kmh = 0

        track_history[track_id] = (x_center, y_center)

        # Vehicle class
        cls_id = int(track.get_det_class()) if track.get_det_class() else -1
        label = COCO_CLASSES.get(cls_id, "unknown")

        # Draw results
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id} | {label} | {speed_kmh:.1f} km/h",
                    (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Classification & Speed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()