import cv2
from collections import deque
import os
from ultralytics import YOLO

VIDEO_PATH = r"C:\Users\machi\Downloads\12121263_3840_2160_30fps.mp4"
MODEL_WEIGHTS = "yolov10x.pt"
CONF_THRESHOLD = 0.05
IMG_SIZE = 1280
FRAME_HISTORY = 10  # How long to persist boxes after missed detection

OUTPUT_FOLDER = r"C:\Users\machi\OneDrive\Desktop\OIL_RIG_FALL_DETECTOR"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "annotated_output.mp4")

def bbox_iou(box1, box2):
    # Returns intersection-over-union of two boxes
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

class SimpleBoxTracker:
    def __init__(self, iou_threshold=0.3, max_lost=FRAME_HISTORY):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_track_id = 1
        self.tracks = {}  # track_id: {'bbox': (x1,y1,x2,y2), 'lost': int}

    def update(self, detections):
        updated_tracks = {}
        unmatched_detections = []
        unmatched_track_ids = list(self.tracks.keys())

        for det in detections:
            best_iou = 0
            best_track_id = None
            for track_id in unmatched_track_ids:
                iou = bbox_iou(det[:4], self.tracks[track_id]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_iou >= self.iou_threshold and best_track_id is not None:
                updated_tracks[best_track_id] = {'bbox': det[:4], 'lost': 0}
                unmatched_track_ids.remove(best_track_id)
            else:
                unmatched_detections.append(det)

        for det in unmatched_detections:
            updated_tracks[self.next_track_id] = {'bbox': det[:4], 'lost': 0}
            self.next_track_id += 1

        for track_id in unmatched_track_ids:
            lost_count = self.tracks[track_id]['lost'] + 1
            if lost_count <= self.max_lost:
                updated_tracks[track_id] = {'bbox': self.tracks[track_id]['bbox'], 'lost': lost_count}
            # Else drop track

        self.tracks = updated_tracks
        # Only output tracks still present
        active_tracks = [
            self.tracks[track_id]['bbox']
            for track_id in self.tracks if self.tracks[track_id]['lost'] <= self.max_lost
        ]
        return active_tracks

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

model = YOLO(MODEL_WEIGHTS)
tracker = SimpleBoxTracker(iou_threshold=0.3, max_lost=FRAME_HISTORY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, augment=True)
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if class_id == 0:  # 'person' in COCO
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 100:
                detections.append((x1, y1, x2, y2, confidence))

    tracked_boxes = tracker.update(detections)
    if tracked_boxes:
        # Draw bounding boxes and the white/purple "MAN OVERBOARD" label
        for x1, y1, x2, y2 in tracked_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            label = "MAN OVERBOARD"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 10), (x1 + tw, y1), (128, 0, 128), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # Persistent warning at top
        cv2.putText(frame, "!!! MAN OVERBOARD WARNING !!!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    out.write(frame)
    cv2.imshow("YOLOv10 Man Overboard Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Annotated video saved at: {OUTPUT_VIDEO_PATH}")
