import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================== PATH ==================
image_dir = Path(r"C:\Users\pc\OneDrive\Desktop\Engineering English\star tracker\star_image")

# Folder lưu ảnh đã xử lý
adjust_root = image_dir / "adjust"
adjust_root.mkdir(exist_ok=True)

# ================== LOOP THROUGH BMP FILES ==================
bmp_files = list(image_dir.glob("*.bmp"))

if not bmp_files:
    raise ValueError("Không tìm thấy file .bmp nào trong folder!")

for bmp_path in bmp_files:
    print(f"🔹 Processing: {bmp_path.name}")

    # Subfolder cho từng ảnh
    img_adjust_dir = adjust_root / bmp_path.stem
    img_adjust_dir.mkdir(exist_ok=True)

    # ================== LOAD IMAGE ==================
    image = cv2.imread(str(bmp_path))
    if image is None:
        print("Không load được ảnh:", bmp_path.name)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ================== PREPROCESS ==================
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.GaussianBlur(blur, (51, 51), 0)
    clean = cv2.subtract(blur, background)

    # ================== THRESHOLD ==================
    mean = np.mean(clean)
    std = np.std(clean)
    k = 4

    _, binary = cv2.threshold(
        clean,
        mean + k * std,
        255,
        cv2.THRESH_BINARY
    )

    # ================== SAVE IMAGES ==================
    cv2.imwrite(str(img_adjust_dir / "1_blur.png"), blur)
    cv2.imwrite(str(img_adjust_dir / "2_clean.png"), clean)
    cv2.imwrite(str(img_adjust_dir / "3_binary.png"), binary)

    # ================== OPTIONAL DISPLAY ==================
    #plt.figure(figsize=(10, 6))
    #plt.imshow(binary, cmap='gray')
    # plt.title(f"Threshold – {bmp_path.name}")
    # plt.axis("off")
    # plt.show()

print("✅ Xử lý xong tất cả ảnh")
