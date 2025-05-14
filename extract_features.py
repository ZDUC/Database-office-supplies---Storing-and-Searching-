import os
import cv2
import numpy as np
from pymongo import MongoClient
import mahotas as mh
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import chan_vese
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_5"]

# Thư mục ảnh
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

# Trọng số của các đặc trưng
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
    h_mean, h_std, h_skew = calculate_moments(hsv[:,:,0])
    s_mean, s_std, s_skew = calculate_moments(hsv[:,:,1])
    v_mean, v_std, v_skew = calculate_moments(hsv[:,:,2])

    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    h_hist /= np.sum(h_hist) + 1e-10
    s_hist /= np.sum(s_hist) + 1e-10
    v_hist /= np.sum(v_hist) + 1e-10

    b_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    r_mean = np.mean(image[:,:,2])

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
        features.append(np.mean([graycoprops(glcm, prop)[d, a] 
                                 for d in range(len(distances)) 
                                 for a in range(len(angles))]))

    haralick_features = mh.features.haralick(gray).mean(axis=0)
    return np.concatenate([features, haralick_features])

def chan_vese_segmentation(gray):
    norm = gray / 255.0
    segmented = chan_vese(norm, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                          max_num_iter=100, dt=0.5, init_level_set="checkerboard",
                          extended_output=False)
    return (segmented.astype(np.uint8) * 255)

def extract_shape_features(segmented):
    moments = cv2.moments(segmented)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu + 1e-10))

    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area, perimeter, circularity, solidity = 0, 0, 0, 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-10)
        if len(c) > 2:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-10)

    zernike = mh.features.zernike_moments(segmented, radius=21, degree=8)
    return np.concatenate([hu, [area, perimeter, circularity, solidity], zernike])

# ===== Chuẩn hóa đặc trưng =====
def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features.reshape(-1, 1)).flatten()

# ===== Quét và xử lý ảnh =====
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path): continue

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")): continue

        img_path = os.path.join(folder_path, filename)
        if collection.find_one({"image_path": img_path}):
            print("⚠ Ảnh đã tồn tại:", img_path)
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                print("❌ Không đọc được ảnh:", img_path)
                continue

            img = cv2.resize(img, TARGET_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            segmented = chan_vese_segmentation(gray)

            f_color = extract_color_features(img, hsv)
            f_texture = extract_texture_features(gray)
            f_shape = extract_shape_features(segmented)

            # Áp dụng trọng số vào các đặc trưng
            f_color_weighted = WEIGHT_COLOR * f_color
            f_texture_weighted = WEIGHT_TEXTURE * f_texture
            f_shape_weighted = WEIGHT_SHAPE * f_shape

            # Chuẩn hóa các đặc trưng
            f_color_normalized = normalize_features(f_color_weighted)
            f_texture_normalized = normalize_features(f_texture_weighted)
            f_shape_normalized = normalize_features(f_shape_weighted)

            # Kết hợp các đặc trưng đã chuẩn hóa
            features_normalized = np.concatenate([f_color_normalized, f_texture_normalized, f_shape_normalized])

            # Lưu vào MongoDB
            collection.insert_one({
                "image_path": img_path,
                "category": folder,
                "features": features_normalized.tolist()
            })
            print("✅ Xử lý:", img_path)
        except Exception as e:
            print("❌ Lỗi ảnh:", img_path, "|", e)

print("🎉 Đã hoàn tất trích xuất và lưu đặc trưng!")