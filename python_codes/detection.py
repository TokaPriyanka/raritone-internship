import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils
image_path = "person_d.jpg"
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)
if results.pose_landmarks:
    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
cv2.imshow("Landmark Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()