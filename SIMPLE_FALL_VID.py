import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLOv9 (humans only)
model = YOLO('yolov9c.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video input
video_path = r"C:\Users\machi\Downloads\MOB_1.mp4"
cap = cv2.VideoCapture(video_path)

# Prepare output video writer
output_path = r"C:\Users\machi\OneDrive\Desktop\fall_detection_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get properties from input video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Fall detection parameters
FALL_ANGLE_THRESHOLD = 40           # Degrees from vertical considered "upright"
FALL_VELOCITY_THRESHOLD = 0.1      # Sensitive to downward movement

# Tracking variables
prev_hip_y = {}           # Previous hip y-coord per person
person_fall_state = {}    # True if currently in a fall
person_fall_ever = {}     # True if a fall was ever detected for a person

def get_shoulder_hip_angle(landmarks, roi_shape):
    h, w = roi_shape[:2]
    ls = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h])
    rs = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h])
    lh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h])
    rh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h])
    shoulder_center = (ls + rs) / 2
    hip_center = (lh + rh) / 2
    body_vector = shoulder_center - hip_center
    angle_rad = np.arctan2(body_vector[1], body_vector[0])
    angle_deg = np.degrees(angle_rad) % 360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg, hip_center[1]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect humans only
    results = model(frame, classes=[0], verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        person_id = (x1, y1, x2, y2)  # Simple tracking by bounding box

        # Pose estimation
        image_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)

        if pose_results.pose_landmarks:
            angle, hip_y = get_shoulder_hip_angle(pose_results.pose_landmarks.landmark, person_roi.shape)

            # Initialize tracking variables for person
            if person_id not in prev_hip_y:
                prev_hip_y[person_id] = hip_y
                person_fall_state[person_id] = False
                person_fall_ever[person_id] = False

            # Calculate normalized velocity: (previous y - current y) / person height
            velocity = (prev_hip_y[person_id] - hip_y) / max(1, (y2 - y1))
            prev_hip_y[person_id] = hip_y

            # Is the person upright?
            is_upright = abs(angle - 90) <= FALL_ANGLE_THRESHOLD

            # Fall detection logic
            if (not person_fall_ever[person_id]) and (velocity > FALL_VELOCITY_THRESHOLD):
                person_fall_ever[person_id] = True

            if person_fall_ever[person_id]:
                if not is_upright:
                    person_fall_state[person_id] = True
                else:
                    person_fall_state[person_id] = False

            # Display posture and velocity (for debugging)
            posture = "Standing/Walking/Running" if is_upright else "Lying Down (FALL)"
            cv2.putText(annotated_frame, f"Posture: {posture}", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"Velocity: {velocity:.2f}", (x1, y1 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw FALL DETECTED with red background and white text ONLY when posture is FALL
            if not is_upright:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 6
                text = "FALL DETECTED!"
                # Get text size for bg rectangle
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x1
                text_y = max(text_size[1] + 10, y1 - 10)
                pad = 10
                # Draw red rectangle
                cv2.rectangle(
                    annotated_frame,
                    (text_x - pad, text_y - text_size[1] - pad),
                    (text_x + text_size[0] + pad, text_y + pad),
                    (0, 0, 255),  # Red in BGR
                    -1  # Filled
                )
                # Draw white text above
                cv2.putText(
                    annotated_frame, text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
                )

    # Show (optional)
    cv2.imshow('Fall Detection (RedBG/WhiteText Alert)', annotated_frame)
    # Write frame to output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
