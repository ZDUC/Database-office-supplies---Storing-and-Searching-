import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from pymongo import MongoClient

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
images_collection = db["images"]
features_collection = db["image_features"]

# Đường dẫn dataset
DATASET_PATH = "dataset_resized"

# Load mô hình ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp Fully Connected
model.eval()

# Chuyển đổi ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize ảnh về 256x256 trước
    transforms.CenterCrop(224),      # Cắt chính giữa ảnh để đảm bảo kích thước 224x224
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa giống ImageNet
])

# Duyệt dataset
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            if img_name.lower().endswith(".jpg"):
                # Chuẩn hóa đường dẫn
                img_path = os.path.normpath(img_path)

                # Kiểm tra xem ảnh đã tồn tại trong DB chưa
                existing_image = features_collection.find_one({"image_path": img_path})
                if existing_image:
                    print(f"⚠ Ảnh đã tồn tại, bỏ qua: {img_path}")
                    continue  # Bỏ qua nếu ảnh đã tồn tại

                # Tiền xử lý ảnh
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0)

                with torch.no_grad():
                    feature = model(image).squeeze().numpy()

                # Đảm bảo đặc trưng là danh sách 1D
                feature_list = feature.flatten().tolist()

                # Lưu vào database
                features_collection.insert_one({
                    "image_path": img_path,
                    "category": category,
                    "feature": feature_list
                })

                print(f"✅ Đã lưu đặc trưng của: {img_path}")

print("🎉 Hoàn thành trích xuất & lưu vào MongoDB!")
