import os
import cv2

IMAGE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

os.makedirs(LABEL_DIR, exist_ok=True)

count = 0

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # YOLO format: class x_center y_center width height
    label = "0 0.5 0.5 0.9 0.9\n"

    txt_name = img_name.rsplit(".", 1)[0] + ".txt"
    with open(os.path.join(LABEL_DIR, txt_name), "w") as f:
        f.write(label)

    count += 1

print(f"✅ Labels created for {count} images")
