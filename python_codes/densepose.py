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
    dense = np.zeros(
        (h,w,3),
        dtype=np.uint8
    )
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        nose = lm[0]
        ls = lm[11]
        rs = lm[12]
        le = lm[13]
        re = lm[14]
        lw = lm[15]
        rw = lm[16]
        lh = lm[23]
        rh = lm[24]
        lk = lm[25]
        rk = lm[26]
        la = lm[27]
        ra = lm[28]
        cv2.circle(
            dense,
            (
                int(nose.x*w),
                int(nose.y*h)
            ),
            55,
            (0,255,255),
            -1
        )
        torso = np.array([
            [int(ls.x*w),int(ls.y*h)],
            [int(rs.x*w),int(rs.y*h)],
            [int(rh.x*w),int(rh.y*h)],
            [int(lh.x*w),int(lh.y*h)]
        ])
        cv2.fillPoly(
            dense,
            [torso],
            (255,0,0)
        )
        cv2.line(
            dense,
            (
                int(ls.x*w),
                int(ls.y*h)
            ),
            (
                int(lw.x*w),
                int(lw.y*h)
            ),
            (0,255,255),
            30
        )
        cv2.line(
            dense,
            (
                int(rs.x*w),
                int(rs.y*h)
            ),
            (
                int(rw.x*w),
                int(rw.y*h)
            ),
            (0,255,255),
            30
        )
        left_leg = np.array([

            [int(lh.x*w),int(lh.y*h)],
            [int(lk.x*w),int(lk.y*h)],
            [int(la.x*w),int(la.y*h)]

        ])
        right_leg = np.array([

            [int(rh.x*w),int(rh.y*h)],
            [int(rk.x*w),int(rk.y*h)],
            [int(ra.x*w),int(ra.y*h)]
        ])
        cv2.polylines(
            dense,
            [left_leg],
            False,
            (255,255,0),
            25
        )
        cv2.polylines(
            dense,
            [right_leg],
            False,
            (255,255,0),
            25
        )
    out = np.hstack(
        (
            frame,
            dense
        )
    )
    cv2.imshow(
        "Original | Dense Pose",
        out
    )
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()