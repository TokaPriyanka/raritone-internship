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
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = pose.process(rgb)
    agnostic = frame.copy()
    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark
        ls = lm[11]
        rs = lm[12]
        le = lm[13]
        re = lm[14]
        lw = lm[15]
        rw = lm[16]
        lh = lm[23]
        rh = lm[24]
        body = np.array([
            [
                int(ls.x*w)-50,
                int(ls.y*h)-30
            ],
            [
                int(rs.x*w)+50,
                int(rs.y*h)-30
            ],
            [
                int(rw.x*w)+40,
                int(rw.y*h)
            ],
            [
                int(rh.x*w)+30,
                int(rh.y*h)+20
            ],
            [
                int(lh.x*w)-30,
                int(lh.y*h)+20
            ],
            [
                int(lw.x*w)-40,
                int(lw.y*h)
            ]
        ])
        cv2.fillPoly(
            agnostic,
            [body],
            (120,120,120)
        )
        cv2.line(
            agnostic,
            (
                int(ls.x*w),
                int(ls.y*h)
            ),
            (
                int(le.x*w),
                int(le.y*h)
            ),
            (120,120,120),
            50
        )
        cv2.line(
            agnostic,
            (
                int(le.x*w),
                int(le.y*h)
            ),
            (
                int(lw.x*w),
                int(lw.y*h)
            ),
            (120,120,120),
            45
        )
        cv2.line(
            agnostic,

            (
                int(rs.x*w),
                int(rs.y*h)
            ),
            (
                int(re.x*w),
                int(re.y*h)
            ),
            (120,120,120),
            50
        )
        cv2.line(
            agnostic,
            (
                int(re.x*w),
                int(re.y*h)
            ),
            (
                int(rw.x*w),
                int(rw.y*h)
            ),
            (120,120,120),
            45
        )
        neck_x = int(
            (
                ls.x +
                rs.x
            )/2 * w
        )
        neck_y = int(
            (
                ls.y +
                rs.y
            )/2 * h
        )
        cv2.circle(
            agnostic,
            (
                neck_x,
                neck_y
            ),
            35,
            (120,120,120),
            -1
        )
    output = np.hstack(
        (
            frame,
            agnostic
        )
    )
    cv2.imshow(
        "Original | Agnostic-v3.2",
        output
    )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()