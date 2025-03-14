import cv2
import os

input_folder = "dataset"  # Thư mục chứa ảnh gốc
output_folder = "dataset_resized"  # Thư mục chứa ảnh sau khi resize

# Tạo thư mục đầu ra nếu chưa có
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Kích thước chuẩn
target_size = (256, 256)

# Duyệt qua từng thư mục con (loại đồ vật)
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    output_category_path = os.path.join(output_folder, category)

    # Tạo thư mục cho từng loại đồ vật trong thư mục đầu ra
    if not os.path.exists(output_category_path):
        os.makedirs(output_category_path)

    # Duyệt qua từng ảnh trong thư mục con
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)  # Đọc ảnh

        if img is not None:
            img_resized = cv2.resize(img, target_size)  # Resize ảnh
            cv2.imwrite(os.path.join(output_category_path, img_name), img_resized)  # Lưu ảnh mới

print("Resize hoàn tất!")
