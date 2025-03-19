from pymongo import MongoClient
import os
import json
from extract_features import extract_cnn_features, extract_sift_features, extract_hog_features

# ✅ Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["images"]

# ✅ Thư mục chứa ảnh
DATASET_PATH = "dataset_resized"

for document in collection.find():
    img_path = document["image_path"]
    
    # Trích xuất đặc trưng
    cnn_features = extract_cnn_features(img_path).tolist()
    sift_features = extract_sift_features(img_path).tolist()
    hog_features = extract_hog_features(img_path).tolist()
    
    # Cập nhật vào MongoDB
    collection.update_one(
        {"_id": document["_id"]},
        {"$set": {
            "cnn_features": cnn_features,
            "sift_features": sift_features,
            "hog_features": hog_features
        }}
    )
    print(f"✅ Cập nhật xong đặc trưng cho: {img_path}")

print("🎉 Hoàn thành lưu đặc trưng vào MongoDB!")
