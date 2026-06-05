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
        ls = lm[11]
        rs = lm[12]
        le = lm[13]
        re = lm[14]
        lw = lm[15]
        rw = lm[16]
        lh = lm[23]
        rh = lm[24]
        upper = np.array([
            [int(ls.x*w)-40,
             int(ls.y*h)-30],
            [int(rs.x*w)+40,
             int(rs.y*h)-30],
            [int(rh.x*w)+20,
             int(rh.y*h)],
            [int(lh.x*w)-20,
             int(lh.y*h)]
        ])
        cv2.fillPoly(
            parse,
            [upper],
            (255,255,255)
        )
        cv2.line(
            parse,
            (int(ls.x*w),
             int(ls.y*h)),
            (int(lw.x*w),
             int(lw.y*h)),
            (255,255,255),
            35
        )
        cv2.line(
            parse,
            (int(rs.x*w),
             int(rs.y*h)),
            (int(rw.x*w),
             int(rw.y*h)),
            (255,255,255),
            35
        )
        neckx = int(
            ((ls.x+rs.x)/2)*w
        )
        necky = int(
            ((ls.y+rs.y)/2)*h
        )
        cv2.circle(
            parse,
            (neckx,necky),
            25,
            (255,255,255),
            -1
        )
    output = np.hstack(
        (
            frame,
            parse
        )
    )
    cv2.imshow(
        "Original | Parse-Agnostic-v3.2",
        output
    )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()