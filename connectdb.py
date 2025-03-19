import os
from pymongo import MongoClient

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]  # Thay bằng tên database
collection = db["images"]  # Thay bằng tên collection

# Thư mục chứa ảnh
DATASET_PATH = "dataset_resized"

# Duyệt qua tất cả thư mục con trong dataset_resized
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    # Kiểm tra nếu là thư mục
    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            # Chỉ lưu file .jpg
            if img_name.lower().endswith(".jpg"):
                document = {
                    "category": category,  # Lưu tên thư mục làm category
                    "image_path": img_path  # Lưu đường dẫn ảnh
                }
                collection.insert_one(document)  # Lưu vào MongoDB
                print(f"✅ Đã lưu: {img_path}")

print("🎉 Hoàn thành lưu danh sách ảnh vào MongoDB!")
