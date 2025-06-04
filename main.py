import cv2
import numpy as np
import os
import mss
import time
import pyautogui

template_dir = "templates"
threshold    = 0.75

# Tên các template sẽ được resize
resize_names = ["banana", "watermelon", "banana_1", "watermelon_1"]
# Các kích thước vuông (pixel) dùng để thử resize
sizes_to_try = [65, 70, 75]

# Tên template cần loại trừ
no_click_name = "boom"

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

sct = mss.mss()
mon = sct.monitors[1]

# Lấy kích thước màn hình
screen_left = mon["left"]
screen_top  = mon["top"]
screen_w    = mon["width"]
screen_h    = mon["height"]

# Kích thước quanh chuột

# roi_w = screen_w // 7.5
# roi_h = screen_h // 7.5

roi_w = 300
roi_h = 300

cv2.namedWindow("Screen Matching", cv2.WINDOW_NORMAL)
print("Bắt đầu capture màn hình quanh con trỏ, nhấn 'q' để thoát.")

# Thời gian cooldown giữa 2 swipe
last_slice_time = 0
slice_cooldown  = 0.01 

# Chụp & match liên tục quanh vị trí chuột
while True:
    # Lấy vị trí hiện tại của con trỏ chuột
    mouse_x, mouse_y = pyautogui.position()

    # Tính sao cho căn giữa quanh (mouse_x, mouse_y)
    roi_left = mouse_x - roi_w // 2
    roi_top  = mouse_y - roi_h // 2

    # Clamp để không vượt ra ngoài màn hình
    if roi_left < screen_left:
        roi_left = screen_left
    elif roi_left + roi_w > screen_left + screen_w:
        roi_left = screen_left + screen_w - roi_w

    if roi_top < screen_top:
        roi_top = screen_top
    elif roi_top + roi_h > screen_top + screen_h:
        roi_top = screen_top + screen_h - roi_h

    monitorROI = {
        "left"  : int(roi_left),
        "top"   : int(roi_top),
        "width" : int(roi_w),
        "height": int(roi_h)
    }

    # Gray
    sct_img    = sct.grab(monitorROI)
    img        = np.array(sct_img)
    frame_roi  = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Duyệt từng template
    for tpl in templates:
        name      = tpl["name"]
        orig_gray = tpl["orig_gray"]
        orig_mask = tpl["orig_mask"]
        h_orig    = tpl["h_orig"]
        w_orig    = tpl["w_orig"]

        best_match = None

        # Nếu template cần resize, thử nhiều kích thước
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

        # Nếu có match ≥ threshold
        if best_match:
            startX, startY = best_match["maxLoc"]
            endX = startX + best_match["w"]
            endY = startY + best_match["h"]

            # if best_match["size"] is not None:
            #     label = f"{name} ({best_match['size']}px, {best_match['maxVal']:.2f})"
            # else:
            #     label = f"{name} (orig, {best_match['maxVal']:.2f})"

            # cv2.rectangle(frame_roi, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.putText(
            #     frame_roi, label,
            #     (startX, startY - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            # )

            # Nếu template không phải 'boom'
            if name.lower() != no_click_name:
                now = time.time()
                if now - last_slice_time >= slice_cooldown:
                    # Tọa độ góc trên trái và góc dưới phải của bounding box
                    midX_roi = startX + best_match["w"] // 2
                    midY_roi = startY + best_match["h"] // 2
                    # Chuyển sang toạ độ màn hình
                    midX_screen = int(roi_left + midX_roi)
                    midY_screen = int(roi_top  + midY_roi)
                    # Di chuột đến giữa bounding box rồi click
                    pyautogui.moveTo(midX_screen, midY_screen)
                    # pyautogui.click()
                    last_click_time = now

    cv2.imshow("Screen Matching", frame_roi)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()