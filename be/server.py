import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.segmentation import chan_vese
from sklearn.metrics.pairwise import cosine_similarity

# ===== Khởi tạo Flask và CORS =====
app = Flask(__name__)
CORS(app)

# ===== Cấu hình MongoDB =====
MONGO_URI = (
    "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Database0"
)
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_four"]

# ===== Thư mục lưu ảnh =====
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (224, 224)

# ===== Trọng số đặc trưng (đã đồng bộ với code mới) =====
WEIGHT_COLOR = 0.4
WEIGHT_TEXTURE = 0.4
WEIGHT_SHAPE = 0.2

def extract_color_features(image):
    # Kiểm tra và chuyển đổi ảnh đầu vào
    if len(image.shape) == 2:  # Nếu ảnh grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Histogram của từng kênh HSV
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    # Thống kê màu sắc
    (b, g, r) = cv2.split(image)
    color_stats = [
        np.mean(b), np.std(b), 
        np.mean(g), np.std(g), 
        np.mean(r), np.std(r),
        np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
        np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
    ]
    
    features = np.concatenate([h_hist, s_hist, v_hist, color_stats])
    return features

def extract_texture_features(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    # LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm] + lbp_hist.tolist())

def extract_shape_features(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Phân đoạn ảnh với xử lý lỗi
    try:
        segmented = chan_vese(gray, max_iter=100, dt=0.5, extended_output=False)
        segmented = (segmented * 255).astype(np.uint8)
    except:
        segmented = gray
    
    # Tìm contour
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(14)  # Số lượng features shape
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Hu Moments
    moments = cv2.moments(largest_contour)
    hu = cv2.HuMoments(moments).flatten()
    
    # Hình dạng
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-7) if perimeter > 0 else 0
    
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-7) if hull_area > 0 else 0
    
    # Biên Canny
    edges = cv2.Canny(segmented, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    return np.concatenate([hu, [area, perimeter, circularity, solidity, edge_density]])

def normalize_features(features):
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / (std + 1e-7)

# ===== ROUTE PHỤC VỤ ẢNH =====
@app.route('/dataset_resized/<path:filename>')
def serve_image(filename):
    path = os.path.join(DATASET_PATH, filename)
    if os.path.exists(path):
        return send_file(path)
    return "File not found", 404

# ===== ROUTE TÌM KIẾM ẢNH =====
@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được gửi'}), 400

    file = request.files['file']
    img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, TARGET_SIZE)
    
    # Trích xuất đặc trưng
    f_color = extract_color_features(img)
    f_texture = extract_texture_features(img)
    f_shape = extract_shape_features(img)
    
    # Gán trọng số và chuẩn hóa
    features = np.concatenate([
        WEIGHT_COLOR * normalize_features(f_color),
        WEIGHT_TEXTURE * normalize_features(f_texture),
        WEIGHT_SHAPE * normalize_features(f_shape)
    ])

    # So sánh với database
    results = []
    for doc in collection.find():
        db_feat = np.array(doc['features'])
        score = cosine_similarity([features], [db_feat])[0][0]
        results.append({'image_url': doc['image_path'].replace('\\', '/'), 'score': float(score)})

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(results[:3])

if __name__ == '__main__':
    app.run(debug=True)