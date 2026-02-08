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
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    #để đọc thêm tài liệu
    background = cv2.GaussianBlur(blur, (51, 51), 0)
    clean = cv2.subtract(blur, background)

    # ================== THRESHOLD ================== (đọc thêm tài liệu)
    mean = np.mean(clean)
    std = np.std(clean)
    k = 4

    _, binary = cv2.threshold(
        clean,
        mean + k * std,
        255,
        cv2.THRESH_BINARY
    )

        # 1. Tìm tất cả các đường viền (contours) từ ảnh nhị phân
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo ảnh mặt nạ mới (sẽ chỉ chứa các ngôi sao đạt chuẩn)
    filtered_binary = np.zeros_like(binary)
    
    # Tạo ảnh màu để visualize (Dùng để debug: Xanh=Lấy, Đỏ=Bỏ)
    debug_vis = image.copy() 
    if len(debug_vis.shape) == 2: debug_vis = cv2.cvtColor(debug_vis, cv2.COLOR_GRAY2BGR)

    # --- CÁC THAM SỐ LỌC (TÙY CHỈNH TẠI ĐÂY) ---
    MIN_AREA = 2       # Nhỏ hơn 2px -> Nhiễu hạt (Salt noise)
    MAX_AREA = 150     # Lớn hơn 150px -> Đèn đường / Mặt trăng / Cửa sổ
    MIN_CIRCULARITY = 0.6 # Dưới 0.6 -> Lá cây / Vệt máy bay (1.0 là tròn vo)

    filtered_count = 0
    
    for cnt in contours:
        # A. Lọc theo Diện tích (Area)
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            cv2.drawContours(debug_vis, [cnt], -1, (255, 0, 0), 1) # Vẽ màu ĐỎ (Bị loại do size)
            continue

        # B. Lọc theo Độ tròn (Circularity)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            cv2.drawContours(debug_vis, [cnt], -1, (255, 0, 255), 1) # Vẽ màu TÍM (Bị loại do méo)
            continue
        
        # C. Nếu vượt qua cả 2 vòng -> Là Sao (hoặc vật thể rất giống sao)
        cv2.drawContours(filtered_binary, [cnt], -1, 255, -1)     # Vẽ đốm trắng lên mask sạch
        cv2.drawContours(debug_vis, [cnt], -1, (0, 255, 0), 1)    # Vẽ màu XANH LÁ (Được chọn)
        filtered_count += 1

    print(f"   -> Filtered: {filtered_count} / {len(contours)}")


  # SAVE IMAGES cv2.imwrite(str(img_adjust_dir / "4_filtered_binary.png"), filtered_binary)

    # ================== SAVE IMAGES ==================
    cv2.imwrite(str(img_adjust_dir / "1_blur.png"), blur)
    cv2.imwrite(str(img_adjust_dir / "2_clean.png"), clean)
    cv2.imwrite(str(img_adjust_dir / "3_binary.png"), binary)

    cv2.imwrite(str(img_adjust_dir / "54_debug_visual.png"), cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR))
    # ================== OPTIONAL DISPLAY ==================
    #plt.figure(figsize=(10, 6))
    #plt.imshow(binary, cmap='gray')
    # plt.title(f"Threshold – {bmp_path.name}")
    # plt.axis("off")
    # plt.show()

print("✅ Xử lý xong tất cả ảnh")
