import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = pose.process(rgb)
    mask = np.zeros(
        (h,w),
        dtype=np.uint8
    )
    cloth = np.zeros_like(frame)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ls = lm[11]
        rs = lm[12]
        lh = lm[23]
        rh = lm[24]
        cloth_region = np.array([
            [
                int(ls.x*w)-30,
                int(ls.y*h)-20
            ],
            [
                int(rs.x*w)+30,
                int(rs.y*h)-20
            ],
            [
                int(rh.x*w)+20,
                int(rh.y*h)
            ],
            [
                int(lh.x*w)-20,
                int(lh.y*h)
            ]
        ])
        cv2.fillPoly(
            mask,
            [cloth_region],
            255
        )
        cloth = cv2.bitwise_and(
            frame,
            frame,
            mask=mask
        )
    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )
    output = np.hstack(
        (
            frame,
            cloth,
            mask_bgr
        )
    )
    cv2.imshow(
        "Original | Cloth | Cloth Mask",
        output
    )
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()