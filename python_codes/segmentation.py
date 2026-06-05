import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_segmentation = mp.solutions.selfie_segmentation
with mp_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    results = segmenter.process(image_rgb)
mask = results.segmentation_mask > 0.5
background = np.ones_like(image_rgb) * 255
output = np.where(mask[:, :, None], image_rgb, background)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(output)
plt.title("Body Segmentation Output")
plt.axis("off")
plt.show()
cv2.imwrite(
    "segmentation_output.png",
    cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
)
print("Segmentation output saved as segmentation_output.png")