import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# Cấu hình MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_3"]

DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

WEIGHT_COLOR = 0.2
WEIGHT_TEXTURE = 0.5
WEIGHT_SHAPE = 0.3

# Sử dụng lại các hàm extract_features từ file extract_features.py
def extract_color_features(image):
    """Trích xuất đặc trưng màu sắc"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    features = []
    for channel in range(3):
        # BGR
        hist = cv2.calcHist([image], [channel], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
        
        # HSV
        bins = 8 if channel == 0 else 4
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
        
        # LAB
        hist = cv2.calcHist([lab], [channel], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    return np.array(features)

def extract_texture_features(image):
    """Trích xuất đặc trưng kết cấu"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    features = []
    features.append(np.var(gray))
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    hist, _ = np.histogram(angle, bins=8, range=(-180, 180))
    hist = hist / (hist.sum() + 1e-7)
    features.extend(hist)
    
    return np.array(features)

def extract_shape_features(image):
    """Trích xuất đặc trưng hình dạng"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7)
    
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-7)
    
    return hu

def extract_features(image):
    """Hàm trích xuất đặc trưng giống hệt trong file extract_features.py"""
    f_color = extract_color_features(image)
    f_texture = extract_texture_features(image)
    f_shape = extract_shape_features(image)
    
    features = np.concatenate([
        WEIGHT_COLOR * f_color,
        WEIGHT_TEXTURE * f_texture,
        WEIGHT_SHAPE * f_shape
    ])
    
    features = features / (np.linalg.norm(features) + 1e-7)
    return features

def calculate_similarity(feat1, feat2):
    """Tính cosine similarity đơn giản"""
    return np.dot(feat1, feat2)

@app.route('/dataset_resized/<path:filename>')
def serve_image(filename):
    path = os.path.join(DATASET_PATH, filename)
    if os.path.exists(path):
        return send_file(path)
    return "File not found", 404

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, TARGET_SIZE)
    
    # Trích xuất đặc trưng giống hệt cách lưu trong DB
    features = extract_features(img)
    
    # Tìm kiếm trong DB
    results = []
    for doc in collection.find():
        db_feat = np.array(doc['features'])
        score = calculate_similarity(features, db_feat)
        results.append({
            'image_url': doc['image_path'].replace('\\', '/'),
            'score': float(score),
            'category': doc.get('category', 'unknown')
        })
    
    # Sắp xếp và trả về top 3
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(results[:3])

if __name__ == '__main__':
    app.run(debug=True)