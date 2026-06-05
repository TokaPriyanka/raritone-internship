import cv2
import numpy as np
import mediapipe as mp
import os

# ==========================
# CREATE OUTPUT
# ==========================

os.makedirs(
    "output",
    exist_ok=True
)

# ==========================
# LOAD IMAGES
# ==========================

person = cv2.imread(
    "input/person.jpg"
)

cloth = cv2.imread(
    "input/cloth.png",
    cv2.IMREAD_UNCHANGED
)

if person is None:
    raise Exception(
        "person.jpg missing"
    )

if cloth is None:
    raise Exception(
        "cloth.png missing"
    )

H,W,_ = person.shape

# ==========================
# EXTRACT SHIRT ONLY
# ==========================

if cloth.shape[2] == 4:

    cloth_rgb = cloth[:,:,:3]

    alpha = cloth[:,:,3]

else:

    cloth_rgb = cloth

    gray = cv2.cvtColor(
        cloth_rgb,
        cv2.COLOR_BGR2GRAY
    )

    _,alpha = cv2.threshold(
        gray,
        240,
        255,
        cv2.THRESH_BINARY_INV
    )

# remove noise

kernel = np.ones(
    (3,3),
    np.uint8
)

alpha = cv2.morphologyEx(
    alpha,
    cv2.MORPH_OPEN,
    kernel
)

# ==========================
# POSE DETECTION
# ==========================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True
)

rgb = cv2.cvtColor(
    person,
    cv2.COLOR_BGR2RGB
)

res = pose.process(
    rgb
)

if res.pose_landmarks is None:

    raise Exception(
        "pose not found"
    )

lm = res.pose_landmarks.landmark

LS = lm[
mp_pose.PoseLandmark.LEFT_SHOULDER
]

RS = lm[
mp_pose.PoseLandmark.RIGHT_SHOULDER
]

LH = lm[
mp_pose.PoseLandmark.LEFT_HIP
]

RH = lm[
mp_pose.PoseLandmark.RIGHT_HIP
]

# ==========================
# TORSO REGION
# ==========================

x1 = int(
min(LS.x,RS.x)*W
)-40

x2 = int(
max(LS.x,RS.x)*W
)+40

y1 = int(
min(
LS.y,
RS.y
)*H
)

y2 = int(
max(
LH.y,
RH.y
)*H
)

x1=max(0,x1)
y1=max(0,y1)

x2=min(W,x2)
y2=min(H,y2)

tw = x2-x1
th = y2-y1

# ==========================
# RESIZE SHIRT
# ==========================

cloth_res = cv2.resize(
    cloth_rgb,
    (tw,th)
)

alpha_res = cv2.resize(
    alpha,
    (tw,th)
)

# ==========================
# TPS STYLE WARP
# ==========================

src = np.float32([

[0,0],

[tw,0],

[0,th],

[tw,th]

])

dst = np.float32([

[20,0],

[tw-20,10],

[0,th],

[tw,th]

])

M = cv2.getPerspectiveTransform(
    src,
    dst
)

warp = cv2.warpPerspective(

    cloth_res,

    M,

    (tw,th)

)

warp_alpha = cv2.warpPerspective(

    alpha_res,

    M,

    (tw,th)

)

# ==========================
# OVERLAY SHIRT ONLY
# ==========================

overlay = person.copy()

region = overlay[
    y1:y2,
    x1:x2
]

rh,rw = region.shape[:2]

warp = cv2.resize(
    warp,
    (rw,rh)
)

warp_alpha = cv2.resize(
    warp_alpha,
    (rw,rh)
)

mask = (
warp_alpha.astype(
np.float32
)/255.0
)

mask = np.expand_dims(
mask,
axis=2
)

result = (

mask*warp

+

(1-mask)

*region

)

result = result.astype(
np.uint8
)

overlay[
y1:y2,
x1:x2
]=result

# ==========================
# SAVE
# ==========================

cv2.imwrite(

"output/vton.png",

overlay

)

print(
"VTON OUTPUT SAVED"
)