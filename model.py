import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
features_collection = db["image_features"]

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

def extract_feature(image_path):
    """Trích xuất đặc trưng từ ảnh đầu vào."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(image).squeeze().numpy()
    
    return feature

def find_top_3_similar(image_path):
    """Tìm 3 ảnh giống nhất với ảnh đầu vào."""
    input_feature = extract_feature(image_path)
    
    # Lấy toàn bộ đặc trưng từ database
    stored_images = list(features_collection.find({}, {"_id": 0, "image_path": 1, "feature": 1}))

    similarities = {}

    for img in stored_images:
        img_path = img["image_path"]
        stored_feature = np.array(img["feature"])
        similarity = cosine_similarity([input_feature], [stored_feature])[0][0]

        # Tránh trùng lặp bằng cách lưu vào dictionary
        similarities[img_path] = similarity

    # Sắp xếp theo độ tương đồng giảm dần
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Lọc bỏ ảnh đầu vào nếu có
    top_3 = [(path, score) for path, score in sorted_results if path != image_path][:3]

    return top_3

# 🖼 Nhập đường dẫn ảnh cần tìm
input_image_path = "g.jpg"  # Đổi thành đường dẫn ảnh thực tế
top_3_results = find_top_3_similar(input_image_path)

# 🎯 Hiển thị kết quả
print("🔥 3 ảnh giống nhất:")
for idx, (img_path, score) in enumerate(top_3_results, 1):
    print(f"{idx}. {img_path} - Độ tương đồng: {score:.4f}")
