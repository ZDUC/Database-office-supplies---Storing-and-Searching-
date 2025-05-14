import os
import cv2
import numpy as np
from pymongo import MongoClient

# ====== CẤU HÌNH MONGODB ======
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_3"]

# ====== CẤU HÌNH DỮ LIỆU ======
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

# ====== TRỌNG SỐ ĐẶC TRƯNG ======
WEIGHT_COLOR = 0.2
WEIGHT_TEXTURE = 0.5
WEIGHT_SHAPE = 0.3

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
    """Hàm trích xuất đặc trưng chính"""
    # Trích xuất các loại đặc trưng
    f_color = extract_color_features(image)
    f_texture = extract_texture_features(image)
    f_shape = extract_shape_features(image)
    
    # Kết hợp có trọng số
    features = np.concatenate([
        WEIGHT_COLOR * f_color,
        WEIGHT_TEXTURE * f_texture,
        WEIGHT_SHAPE * f_shape
    ])
    
    # Chuẩn hóa toàn bộ vector
    features = features / (np.linalg.norm(features) + 1e-7)
    
    return features.tolist()

def process_dataset():
    """Xử lý dataset và lưu vào MongoDB"""
    processed_images = set()
    existing_images = {doc['image_path'] for doc in collection.find({}, {'image_path': 1})}
    
    for root, _, files in os.walk(DATASET_PATH):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(root, filename)
            if img_path in processed_images or img_path in existing_images:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, TARGET_SIZE)
            features = extract_features(img)
            
            if features is not None:
                category = os.path.basename(root)
                collection.insert_one({
                    'image_path': img_path,
                    'category': category,
                    'features': features
                })
                processed_images.add(img_path)
                print(f"Đã xử lý: {img_path}")

if __name__ == '__main__':
    process_dataset()