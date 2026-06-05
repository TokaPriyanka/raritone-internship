import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
person = cv2.imread("person.jpg")
shirt = cv2.imread("shirt.png")
if person is None:
    print("Error: person.jpg not found")
    exit()
if shirt is None:
    print("Error: shirt.png not found")
    exit()
shirt_rgb = cv2.cvtColor(shirt, cv2.COLOR_BGR2RGB)
lower_white = np.array([220, 220, 220])
upper_white = np.array([255, 255, 255])
mask = cv2.inRange(
    shirt_rgb,
    lower_white,
    upper_white
)
shirt_rgba = cv2.cvtColor(
    shirt_rgb,
    cv2.COLOR_RGB2RGBA
)
shirt_rgba[:, :, 3] = 255 - mask
shirt = shirt_rgba
def crop_transparent(img):
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]
shirt = crop_transparent(shirt)
person_rgb = cv2.cvtColor(
    person,
    cv2.COLOR_BGR2RGB
)
h, w = person_rgb.shape[:2]
mp_pose = mp.solutions.pose
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
) as pose:
    results = pose.process(person_rgb)
if not results.pose_landmarks:
    print("No pose detected")
    exit()
landmarks = results.pose_landmarks.landmark
left_shoulder = landmarks[
    mp_pose.PoseLandmark.LEFT_SHOULDER
]
right_shoulder = landmarks[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]
left_hip = landmarks[
    mp_pose.PoseLandmark.LEFT_HIP
]
right_hip = landmarks[
    mp_pose.PoseLandmark.RIGHT_HIP
]
left_x = int(left_shoulder.x * w)
right_x = int(right_shoulder.x * w)
left_y = int(left_shoulder.y * h)
right_y = int(right_shoulder.y * h)
hip_y = int(
    ((left_hip.y + right_hip.y) / 2) * h
)
shoulder_width = abs(
    right_x - left_x
)
shirt_width = int(
    shoulder_width * 1.8
)
aspect_ratio = shirt.shape[0] / shirt.shape[1]
shirt_height = int(
    shirt_width * aspect_ratio
)
shirt = cv2.resize(
    shirt,
    (shirt_width, shirt_height)
)
center_x = (
    left_x + right_x
) // 2
x = center_x - shirt_width // 2
neck_y = min(
    left_y,
    right_y
)
y = neck_y - int(
    shirt_height * 0.05
)
x = max(0, x)
y = max(0, y)
if x + shirt_width > w:
    shirt_width = w - x
if y + shirt_height > h:
    shirt_height = h - y
shirt = cv2.resize(
    shirt,
    (shirt_width, shirt_height)
)
output = person_rgb.copy()
alpha = shirt[:, :, 3] / 255.0
for c in range(3):
    output[
        y:y+shirt_height,
        x:x+shirt_width,
        c
    ] = (
        alpha * shirt[:, :, c]
        +
        (1 - alpha) *
        output[
            y:y+shirt_height,
            x:x+shirt_width,
            c
        ]
    )
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(person_rgb)
plt.title("Original Person")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(shirt[:, :, :3])
plt.title("Garment")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(output)
plt.title("Dynamic Virtual Try-On")
plt.axis("off")
plt.tight_layout()
plt.show()
cv2.imwrite(
    "dynamic_tryon_output.png",
    cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )
)
print("\n===== TRY-ON DETAILS =====")
print("Shoulder Width :", shoulder_width)
print("Shirt Width    :", shirt_width)
print("Shirt Height   :", shirt_height)
print("Position (x,y) :", x, y)
print("Saved : dynamic_tryon_output.png")