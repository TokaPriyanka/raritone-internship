import cv2
import mediapipe as mp
import numpy as np
mp_selfie_segmentation = mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        mask = results.segmentation_mask
        condition = mask > 0.5
        bg_color = np.zeros(image.shape, dtype=np.uint8)
        output_image = np.where(
            condition[:, :, np.newaxis],
            image,
            bg_color
        )
        mask_display = (mask * 255).astype(np.uint8)
        cv2.imshow("Original Image", image)
        cv2.imshow("Segmented Human", output_image)
        cv2.imshow("Segmentation Mask", mask_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()