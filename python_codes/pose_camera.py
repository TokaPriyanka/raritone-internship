import cv2
import mediapipe as mp

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=2,
                    circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(255,0,0),
                    thickness=2
                )
            )

        cv2.imshow(
            "MediaPipe Pose Detection",
            frame
        )

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()