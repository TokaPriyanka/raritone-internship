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
    parse = np.zeros(
        (h,w,3),
        dtype=np.uint8
    )
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        nose = lm[0]
        cv2.circle(
            parse,
            (
                int(nose.x*w),
                int(nose.y*h)
            ),
            50,
            (255,0,0),
            -1
        )
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
        torso = np.array([
            [int(ls.x*w),int(ls.y*h)],
            [int(rs.x*w),int(rs.y*h)],
            [int(rh.x*w),int(rh.y*h)],
            [int(lh.x*w),int(lh.y*h)]
        ])
        cv2.fillPoly(
            parse,
            [torso],
            (0,165,255)
        )
        cv2.line(
            parse,
            (
                int(ls.x*w),
                int(ls.y*h)
            ),
            (
                int(lw.x*w),
                int(lw.y*h)
            ),
            (255,255,0),
            35
        )
        cv2.line(
            parse,
            (
                int(rs.x*w),
                int(rs.y*h)
            ),
            (
                int(rw.x*w),
                int(rw.y*h)
            ),
            (255,255,0),
            35
        )
        lower = np.array([
            [int(lh.x*w),int(lh.y*h)],
            [int(rh.x*w),int(rh.y*h)],
            [int(rk.x*w),int(rk.y*h)],
            [int(lk.x*w),int(lk.y*h)]
        ])
        cv2.fillPoly(
            parse,
            [lower],
            (0,255,255)
        )
    output = np.hstack(
        (
            frame,
            parse
        )
    )
    cv2.imshow(
        "Original | Parse-v3",
        output
    )
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()