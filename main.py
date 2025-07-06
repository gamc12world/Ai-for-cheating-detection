import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    ih, iw, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Right eye
            re_left = face_landmarks.landmark[33]   # outer
            re_right = face_landmarks.landmark[133] # inner
            re_iris = face_landmarks.landmark[468]  # iris center

            # Left eye
            le_left = face_landmarks.landmark[362]  # outer
            le_right = face_landmarks.landmark[263] # inner
            le_iris = face_landmarks.landmark[473]  # iris center

            # Normalize both eye ratios
            def eye_ratio(outer, inner, iris):
                eye_width = (inner.x - outer.x)
                iris_offset = (iris.x - outer.x)
                return iris_offset / (eye_width + 1e-6)

            right_ratio = eye_ratio(re_left, re_right, re_iris)
            left_ratio  = eye_ratio(le_left, le_right, le_iris)

            # Average ratio from both eyes
            avg_ratio = (right_ratio + left_ratio) / 2

            # Determine gaze direction
            if avg_ratio < 0.35:
                gaze = "Looking Left"
            elif avg_ratio > 0.65:
                gaze = "Looking Right"
            else:
                gaze = "Looking Center"

            # Show gaze text
            cv2.putText(frame, gaze, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Optional: draw iris centers
            for idx in [468, 473]:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Gaze Tracking - Both Eyes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
