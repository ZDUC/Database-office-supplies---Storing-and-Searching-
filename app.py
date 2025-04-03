import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from pymongo import MongoClient
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
features_collection = db["image_features"]

# Đường dẫn dataset
DATASET_PATH = "dataset_resized"

# Load mô hình ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp Fully Connected
model.eval()

# Chuyển đổi ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Route phục vụ ảnh từ dataset_resized
@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(DATASET_PATH, filename)

# Route chính để hiển thị giao diện tìm kiếm
@app.route("/", methods=["GET", "POST"])
def index():
    top_3_results = []
    uploaded_image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            uploaded_image_path = os.path.join("static", "uploads", file.filename)
            file.save(uploaded_image_path)

            # Xử lý ảnh truy vấn
            query_image = Image.open(uploaded_image_path).convert("RGB")
            query_image = transform(query_image).unsqueeze(0)

            with torch.no_grad():
                query_feature = model(query_image).squeeze().numpy()

            # Tìm ảnh giống nhất trong MongoDB
            all_images = list(features_collection.find({}))
            similarities = []
            for img in all_images:
                stored_feature = np.array(img["feature"])
                similarity = 1 - cosine(query_feature, stored_feature)
                similarities.append((img["image_path"], similarity))

            # Sắp xếp theo độ tương đồng và lấy 3 ảnh giống nhất
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_3_results = similarities[:3]

    return render_template("index.html", uploaded_image=uploaded_image_path, top_3_results=top_3_results)

if __name__ == "__main__":
    app.run(debug=True)
