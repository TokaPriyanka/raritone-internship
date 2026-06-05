import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = pose.process(rgb)
    openpose = np.zeros_like(frame)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            openpose,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(
                color=(0,255,0),
                thickness=3,
                circle_radius=4
            ),
            mp_draw.DrawingSpec(
                color=(255,255,255),
                thickness=2
            )
        )
    out = np.hstack(
        (
            frame,
            openpose
        )
    )
    cv2.imshow(
        "Original | Open Pose",
        out
    )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()