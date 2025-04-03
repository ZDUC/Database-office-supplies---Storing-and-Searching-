from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔗 Kết nối MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
features_collection = db["image_features"]

# 📷 Thư mục chứa ảnh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset_resized")

# 🚀 Load mô hình ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp Fully Connected
model.eval()

# 🎨 Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(image_path):
    """Trích xuất đặc trưng từ ảnh."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            feature = model(image).squeeze().numpy()
        return feature
    except Exception as e:
        print(f"❌ Lỗi đọc ảnh {image_path}: {e}")
        return None

def ensure_features_in_db():
    """Kiểm tra và lưu đặc trưng của ảnh vào MongoDB nếu chưa có."""
    if features_collection.count_documents({}) > 0:
        print("✅ Đã có dữ liệu trong MongoDB.")
        return

    print("🔍 Đang trích xuất và lưu đặc trưng ảnh vào MongoDB...")
    for filename in os.listdir(IMAGE_FOLDER):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        feature = extract_feature(image_path)
        if feature is not None:
            features_collection.insert_one({
                "image_path": filename,  # Chỉ lưu tên file để tránh lỗi đường dẫn
                "feature": feature.tolist()
            })
    print("✅ Hoàn thành việc lưu đặc trưng ảnh.")

def find_top_3_similar(image_path):
    """Tìm 3 ảnh giống nhất mà không bị trùng."""
    print("📥 Đang tìm kiếm ảnh tương tự...")
    input_feature = extract_feature(image_path)
    if input_feature is None:
        return []

    # Lấy tất cả ảnh từ MongoDB
    stored_images = list(features_collection.find({}, {"image_path": 1, "feature": 1, "_id": 0}))
    if not stored_images:
        print("⚠ Không tìm thấy dữ liệu trong MongoDB!")
        return []

    similarities = []
    seen_images = set()  # Dùng để tránh ảnh trùng

    for img in stored_images:
        img_path = img["image_path"].replace("\\", "/")  # Chuẩn hóa đường dẫn
        if img_path in seen_images:  # Nếu ảnh đã xét trước đó, bỏ qua
            continue

        stored_feature = np.array(img["feature"])
        similarity = cosine_similarity([input_feature], [stored_feature])[0][0]

        similarities.append((img_path, similarity))
        seen_images.add(img_path)  # Đánh dấu ảnh đã xử lý

    # Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Lấy top 3 ảnh không trùng
    top_3_images = [{"image_url": f"/{img[0]}", "score": round(img[1], 4)} for img in similarities[:3]]

    return top_3_images


@app.route("/dataset_resized/<path:filename>")
def get_image(filename):
    """Phục vụ ảnh từ dataset_resized."""
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/search", methods=["POST"])
def search():
    # Kiểm tra file gửi đến
    if "file" not in request.files:
        return jsonify({"error": "Không có file được tải lên"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Không có file nào được chọn"}), 400

    # 🔍 Tìm ảnh tương tự
    top_3_results = find_top_3_similar(file)

    # 🛠 Chuyển np.float64 → float và thêm domain vào image_url
    for item in top_3_results:
        item["score"] = float(item["score"])  # Chuyển numpy float64 thành float
        item["image_url"] = f"http://127.0.0.1:5000{item['image_url']}"  # Thêm domain đầy đủ

    print(f"📤 Kết quả gửi về frontend: {top_3_results}")

    return jsonify(top_3_results)

if __name__ == "__main__":
    ensure_features_in_db()
    app.run(debug=True)
