import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient
import mahotas as mh
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import chan_vese
from sklearn.preprocessing import MinMaxScaler
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
collection = db["image_features_5"]

# ===== Thư mục lưu ảnh =====
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

# ===== Trọng số đặc trưng =====
WEIGHT_COLOR = 0.15
WEIGHT_TEXTURE = 0.45
WEIGHT_SHAPE = 0.4

# ===== Các hàm trích xuất đặc trưng =====
def calculate_moments(channel):
    mean = np.mean(channel)
    std = np.std(channel)
    skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-10) if std > 0 else 0
    return mean, std, skew


def extract_color_features(image, hsv):
    # Moments HSV
    h_mean, h_std, h_skew = calculate_moments(hsv[:, :, 0])
    s_mean, s_std, s_skew = calculate_moments(hsv[:, :, 1])
    v_mean, v_std, v_skew = calculate_moments(hsv[:, :, 2])
    # Histogram HSV
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-10)
    s_hist /= (s_hist.sum() + 1e-10)
    v_hist /= (v_hist.sum() + 1e-10)
    # Mean BGR
    b_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    r_mean = np.mean(image[:, :, 2])
    return np.concatenate([
        [h_mean, h_std, h_skew, s_mean, s_std, s_skew, v_mean, v_std, v_skew],
        [b_mean, g_mean, r_mean],
        h_hist, s_hist, v_hist
    ])


def extract_texture_features(gray):
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        vals = graycoprops(glcm, prop)
        features.append(np.mean(vals))
    haralick = mh.features.haralick(gray).mean(axis=0)
    return np.concatenate([features, haralick])


def chan_vese_segmentation(gray):
    norm = gray / 255.0
    seg = chan_vese(norm, mu=0.25, lambda1=1, lambda2=1,
                    tol=1e-3, max_num_iter=100, dt=0.5,
                    init_level_set="checkerboard", extended_output=False)
    return (seg.astype(np.uint8) * 255)


def extract_shape_features(segmented):
    m = cv2.moments(segmented)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    # Contour features
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = solidity = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-10)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-10)
    zernike = mh.features.zernike_moments(segmented, radius=21, degree=8)
    return np.concatenate([hu, [area, perimeter, circularity, solidity], zernike])

# ===== Chuẩn hóa đặc trưng =====
def normalize_features(feat):
    scaler = MinMaxScaler()
    return scaler.fit_transform(feat.reshape(-1, 1)).flatten()

# ===== Route phục vụ ảnh tĩnh =====
@app.route('/dataset_resized/<path:filename>')
def serve_image(filename):
    path = os.path.join(DATASET_PATH, filename)
    if os.path.exists(path):
        return send_file(path)
    return "File not found", 404

# ===== Route tìm kiếm ảnh =====
@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được gửi'}), 400
    # Đọc file ảnh upload
    file = request.files['file']
    img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    seg = chan_vese_segmentation(gray)
    # Trích đặc trưng
    f_color = extract_color_features(img, hsv)
    f_texture = extract_texture_features(gray)
    f_shape = extract_shape_features(seg)
    # Nhân trọng số
    f_color *= WEIGHT_COLOR
    f_texture *= WEIGHT_TEXTURE
    f_shape *= WEIGHT_SHAPE
    # Chuẩn hóa
    f_color = normalize_features(f_color)
    f_texture = normalize_features(f_texture)
    f_shape = normalize_features(f_shape)
    feat = np.concatenate([f_color, f_texture, f_shape])
    # So sánh với DB
    results = []
    for doc in collection.find():
        db_feat = np.array(doc['features'])
        score = cosine_similarity([feat], [db_feat])[0][0]
        results.append({'image_url': doc['image_path'].replace('\\','/'), 'score': float(score)})
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(results[:3])

if __name__ == '__main__':
    app.run(debug=True) 