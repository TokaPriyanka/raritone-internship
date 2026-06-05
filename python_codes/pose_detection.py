import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
) as pose:
    results = pose.process(image_rgb)
    output = image_rgb.copy()
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            output,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(output)
plt.title("Pose Detection Output")
plt.axis("off")
plt.show()
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite("pose_detection_output.png", output_bgr)
print("Saved: pose_detection_output.png")