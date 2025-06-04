import cv2
import numpy as np
import os

input_path  = 'test.jpg'
output_path = 'processed.jpg'
threshold   = 0.75
template_dir = 'templates'

resize_names = ['banana', 'watermelon','banana_1', 'watermelon_1']

sizes_to_try = [60, 65, 70, 75, 80]

templates = []
for filename in os.listdir(template_dir):
    if not filename.lower().endswith('.png'):
        continue

    template_name = os.path.splitext(filename)[0]
    template_path = os.path.join(template_dir, filename)
    template_orig = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template_orig is None:
        print(f"Không thể đọc template '{filename}'")
        continue
    else:
        print(f"Đã đọc template '{template_name}'")

    if template_orig.shape[2] == 4:
        bgr_part = template_orig[:, :, :3]
        alpha_channel = template_orig[:, :, 3]

        # Lưu nguyên bản để dùng match (nếu không resize)
        templates.append({
            'name': template_name,
            'orig_bgr': bgr_part,
            'orig_mask': alpha_channel
        })
    else:
        print(f"Template '{filename}' không có kênh alpha để làm mask.")

if not templates:
    print("Không tìm thấy template hợp lệ trong thư mục.")
    exit()

# Đọc ảnh gốc (màu)
image = cv2.imread(input_path)
if image is None:
    print("Không thể mở ảnh:", input_path)
    exit()

for tpl in templates:
    name = tpl['name']
    orig_bgr  = tpl['orig_bgr']
    orig_mask = tpl['orig_mask']
    h_orig, w_orig = orig_bgr.shape[:2]

    best_match = None

    # Nếu tên template nằm trong resize_names -> thử nhiều kích thước
    if name in resize_names:
        best_matches = []
        for square_size in sizes_to_try:
            # Resize về square_size x square_size
            template_bgr_resized = cv2.resize(orig_bgr, (square_size, square_size))
            mask_resized         = cv2.resize(orig_mask, (square_size, square_size))
            tH, tW = square_size, square_size

            if image.shape[0] >= tH and image.shape[1] >= tW:
                result = cv2.matchTemplate(
                    image, template_bgr_resized,
                    cv2.TM_CCOEFF_NORMED, mask=mask_resized
                )
                _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
                if maxVal >= threshold:
                    best_matches.append({
                        'size': square_size,
                        'maxVal': maxVal,
                        'maxLoc': maxLoc,
                        'w': tW,
                        'h': tH
                    })

        if best_matches:
            # Chọn match có confidence cao nhất
            best_match = max(best_matches, key=lambda x: x['maxVal'])
    else:
        # Với template khác, chỉ match ở kích thước gốc
        tH, tW = h_orig, w_orig
        if image.shape[0] >= tH and image.shape[1] >= tW:
            result = cv2.matchTemplate(
                image, orig_bgr,
                cv2.TM_CCOEFF_NORMED, mask=orig_mask
            )
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
            if maxVal >= threshold:
                best_match = {
                    'size': None,      # không phải scale tùy chọn
                    'maxVal': maxVal,
                    'maxLoc': maxLoc,
                    'w': tW,
                    'h': tH
                }

    # Nếu tìm được kết quả thỏa threshold, vẽ bounding box
    if best_match:
        startX, startY = best_match['maxLoc']
        endX = startX + best_match['w']
        endY = startY + best_match['h']

        # Ghi chú: nếu có size, hiển thị size đó, ngược lại bỏ qua
        if best_match['size'] is not None:
            label = f"{name} ({best_match['size']}px, {best_match['maxVal']:.2f})"
            print(f"Tìm thấy '{name}' (resize={best_match['size']}) "
                  f"— confidence={best_match['maxVal']:.3f} tại ({startX}, {startY})")
        else:
            label = f"{name} (orig, {best_match['maxVal']:.2f})"
            print(f"Tìm thấy '{name}' (kích thước gốc) — confidence={best_match['maxVal']:.3f} "
                  f"tại ({startX}, {startY})")

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Lưu ảnh kết quả
cv2.imwrite(output_path, image)
print("Kết quả đã lưu tại:", output_path)

# Hiển thị (nhấn phím bất kỳ để đóng)
cv2.imshow("Kết quả match có điều kiện resize", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
