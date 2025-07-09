import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
pose = mp_pose.Pose()

# Webcam
cap = cv2.VideoCapture(0)

# CSV Logging
log_file = open("iris_posture_log.csv", "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow([
    "timestamp", "gaze", "iris_score", "norm_iris_movement",
    "posture_score", "head_tilt", "shoulder_angle"
])

# Movement tracking
prev_center = None
movement_history = []

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ih, iw, _ = frame.shape

    # Process Pose & Face
    pose_result = pose.process(rgb)
    face_result = face_mesh.process(rgb)

    gaze = "No face"
    head_tilt_angle = 0
    shoulder_angle = 0
    posture_score = 100
    score = 100
    norm_movement = 0

    # Posture: pose landmarks
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark

        # Get landmarks
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        # Convert to pixel coords
        l_px = np.array([int(l_shoulder.x * iw), int(l_shoulder.y * ih)])
        r_px = np.array([int(r_shoulder.x * iw), int(r_shoulder.y * ih)])
        nose_px = np.array([int(nose.x * iw), int(nose.y * ih)])

        # Shoulder tilt
        shoulder_angle = get_angle(r_px, l_px)
        if abs(shoulder_angle) > 10:
            posture_score -= int(abs(shoulder_angle))

        # Head tilt
        mid_shoulder = (r_px + l_px) / 2
        head_tilt_angle = get_angle(mid_shoulder, nose_px)
        if abs(head_tilt_angle) > 15:
            posture_score -= int(abs(head_tilt_angle) / 2)

        posture_score = max(0, min(100, posture_score))

        # Draw full pose mesh
        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Iris Tracking
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            right_iris = face_landmarks.landmark[468]
            left_iris = face_landmarks.landmark[473]

            r_px = np.array([int(right_iris.x * iw), int(right_iris.y * ih)])
            l_px = np.array([int(left_iris.x * iw), int(left_iris.y * ih)])
            iris_center = (r_px + l_px) / 2

            # Eye size normalization
            re_outer = face_landmarks.landmark[33]
            re_inner = face_landmarks.landmark[133]
            le_outer = face_landmarks.landmark[362]
            le_inner = face_landmarks.landmark[263]

            re_width = np.linalg.norm(np.array([re_inner.x, re_inner.y]) - np.array([re_outer.x, re_outer.y])) * iw
            le_width = np.linalg.norm(np.array([le_inner.x, le_inner.y]) - np.array([le_outer.x, le_outer.y])) * iw
            avg_eye_width = (re_width + le_width) / 2

            if prev_center is not None and avg_eye_width > 0:
                pixel_movement = np.linalg.norm(iris_center - prev_center)
                norm_movement = pixel_movement / avg_eye_width
                movement_history.append(norm_movement)
                if len(movement_history) > 30:
                    movement_history.pop(0)
            else:
                norm_movement = 0
            prev_center = iris_center

            # Gaze estimation
            def eye_ratio(outer, inner, iris):
                return (iris.x - outer.x) / ((inner.x - outer.x) + 1e-6)

            right_ratio = eye_ratio(re_outer, re_inner, right_iris)
            left_ratio = eye_ratio(le_outer, le_inner, left_iris)
            avg_ratio = (right_ratio + left_ratio) / 2

            if avg_ratio < 0.35:
                gaze = "Looking Left"
            elif avg_ratio > 0.65:
                gaze = "Looking Right"
            else:
                gaze = "Looking Center"

            # Draw iris
            for idx in [468, 473]:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    # Iris Score
    if movement_history:
        smoothed_movement = np.mean(movement_history)
        score = int(max(0, min(100, 100 - (smoothed_movement * 100))))
    else:
        smoothed_movement = 0
        score = 100

    # Display
    cv2.putText(frame, f"Gaze: {gaze}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Iris Stability: {score}/100", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 50), 2)
    cv2.putText(frame, f"Posture Score: {posture_score}/100", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 200), 2)

    # Log to CSV
    timestamp = datetime.now().strftime("%H:%M:%S")
    csv_writer.writerow([
        timestamp, gaze, score,
        round(smoothed_movement, 4), posture_score,
        round(head_tilt_angle, 2), round(shoulder_angle, 2)
    ])

    cv2.imshow("Iris + Full Body Posture Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
