import cv2
import numpy as np
import os
import mss
import time
import pyautogui

# --- Thiết lập thư mục template và các thông số chung ---
template_dir  = "templates"
threshold     = 0.8

# Tên các template sẽ được resize
resize_names  = ["banana", "watermelon", "banana_1", "watermelon_1"]
# Các kích thước vuông (pixel) dùng để thử resize
sizes_to_try  = [65, 70, 75]

# Tên template cần loại trừ (không auto-click)
no_click_name  = "boom"

# --- Bước 1: Đọc tất cả template (chỉ PNG có alpha) ---
templates = []
for filename in os.listdir(template_dir):
    if not filename.lower().endswith(".png"):
        continue

    template_name = os.path.splitext(filename)[0]
    template_path = os.path.join(template_dir, filename)
    template_orig = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template_orig is None:
        print(f"Không thể đọc template '{filename}'")
        continue

    # Chỉ xử lý PNG có 4 kênh (BGR + alpha)
    if template_orig.shape[2] == 4:
        bgr_part      = template_orig[:, :, :3]
        alpha_channel = template_orig[:, :, 3]
        gray_part     = cv2.cvtColor(bgr_part, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = gray_part.shape[:2]

        templates.append({
            "name"      : template_name,
            "orig_gray" : gray_part,
            "orig_mask" : alpha_channel,
            "h_orig"    : h_orig,
            "w_orig"    : w_orig
        })
        print(f"Đã đọc template '{template_name}' (kích thước gốc: {w_orig}×{h_orig})")
    else:
        print(f"Template '{filename}' không có kênh alpha, bỏ qua.")

if not templates:
    print("Không tìm thấy template hợp lệ trong thư mục. Thoát.")
    exit()

# --- Bước 2: Khởi tạo MSS để chụp toàn màn hình ---
sct = mss.mss()
monitor = sct.monitors[1]  # Sử dụng màn hình chính (full screen)

cv2.namedWindow("Screen Matching", cv2.WINDOW_NORMAL)
print("Bắt đầu capture màn hình, nhấn 'q' để thoát.")

# Tạm giữ thời điểm click cuối cùng (để tránh click quá dày)
last_click_time = 0
click_cooldown = 0.1  # cho phép click tối đa 10 lần/giây

# --- Bước 3: Vòng lặp chụp & match liên tục ---
while True:
    # 3.1. Chụp màn hình
    img = np.array(sct.grab(monitor))           # Kết quả là BGRA
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3.2. Duyệt từng template
    for tpl in templates:
        name      = tpl["name"]
        orig_gray = tpl["orig_gray"]
        orig_mask = tpl["orig_mask"]
        h_orig    = tpl["h_orig"]
        w_orig    = tpl["w_orig"]

        best_match = None

        # Nếu template cần resize, thử tất cả các kích thước trong sizes_to_try
        if name in resize_names:
            candidates = []
            for sz in sizes_to_try:
                gray_resized = cv2.resize(orig_gray, (sz, sz))
                mask_resized = cv2.resize(orig_mask, (sz, sz))
                tH, tW = sz, sz

                if gray_frame.shape[0] >= tH and gray_frame.shape[1] >= tW:
                    result = cv2.matchTemplate(
                        gray_frame, gray_resized,
                        cv2.TM_CCOEFF_NORMED,
                        mask=mask_resized
                    )
                    _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
                    if maxVal >= threshold:
                        candidates.append({
                            "size"  : sz,
                            "maxVal": maxVal,
                            "maxLoc": maxLoc,
                            "w"     : tW,
                            "h"     : tH
                        })

            if candidates:
                best_match = max(candidates, key=lambda x: x["maxVal"])

        else:
            # Với template khác, chỉ match ở kích thước gốc
            tH, tW = h_orig, w_orig
            if gray_frame.shape[0] >= tH and gray_frame.shape[1] >= tW:
                result = cv2.matchTemplate(
                    gray_frame, orig_gray,
                    cv2.TM_CCOEFF_NORMED,
                    mask=orig_mask
                )
                _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
                if maxVal >= threshold:
                    best_match = {
                        "size"  : None,
                        "maxVal": maxVal,
                        "maxLoc": maxLoc,
                        "w"     : tW,
                        "h"     : tH
                    }

        # 3.3. Nếu có match ≥ threshold, vẽ bounding box và (nếu không phải 'boom') thực hiện click
        if best_match:
            startX, startY = best_match["maxLoc"]
            endX = startX + best_match["w"]
            endY = startY + best_match["h"]

            if best_match["size"] is not None:
                label = f"{name} ({best_match['size']}px, {best_match['maxVal']:.2f})"
            else:
                label = f"{name} (orig, {best_match['maxVal']:.2f})"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(
                frame, label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Nếu template không phải 'boom', tự động click vào giữa vùng khung
            if name.lower() != no_click_name:
                now = time.time()
                # Kiểm tra cooldown để không click quá nhanh
                if now - last_click_time >= click_cooldown:
                    midX = startX + best_match["w"] // 2
                    midY = startY + best_match["h"] // 2
                    # Thực hiện click
                    pyautogui.click(midX, midY)
                    last_click_time = now

    # 3.4. Hiển thị kết quả lên cửa sổ
    cv2.imshow("Screen Matching", frame)

    # 3.5. Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
