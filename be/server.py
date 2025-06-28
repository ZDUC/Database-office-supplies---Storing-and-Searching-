import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient
import math
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ===== Khởi tạo Flask và CORS =====
app = Flask(__name__)
CORS(app)

# ===== Cấu hình MongoDB =====
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["6"]

# ===== Thư mục lưu ảnh =====
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

# ===== Các hàm trích xuất đặc trưng =====

# ====== Hàm chuyển RGB sang HSV ======
def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360 if g >= b else (60 * ((g-b)/df) + 360 + 360))
        h %= 360
    elif mx == g:
        h = 60 * ((b-r)/df) + 120
    elif mx == b:
        h = 60 * ((r-g)/df) + 240
    s = 0 if mx == 0 else df/mx
    v = mx
    return h, s, v

# ====== Hàm tính moment (trung bình, độ lệch chuẩn, độ lệch tâm) ======
def calculate_moments(channel):
    mean = np.mean(channel)
    std = np.std(channel)
    skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-10) if std > 0 else 0
    return mean, std, skew

# ====== Hàm trích xuất đặc trưng màu ======
def extract_color_features(image):
    # Chuyển ảnh sang HSV thủ công
    hsv = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i,j]
            h, s, v = rgb_to_hsv(r, g, b)
            hsv[i,j] = [h, s, v]
    
    # Tính các thống kê màu
    h_mean, h_std, h_skew = calculate_moments(hsv[:,:,0])
    s_mean, s_std, s_skew = calculate_moments(hsv[:,:,1])
    v_mean, v_std, v_skew = calculate_moments(hsv[:,:,2])

    # Tính histogram màu thủ công
    def manual_hist(channel, bins=8, range=(0, 256)):
        hist = np.zeros(bins)
        bin_size = (range[1] - range[0]) / bins
        for val in channel.flatten():
            idx = min(int((val - range[0]) / bin_size), bins-1)
            hist[idx] += 1
        hist /= hist.sum() + 1e-10
        return hist
    
    h_hist = manual_hist(hsv[:,:,0], 8, (0, 360))
    s_hist = manual_hist(hsv[:,:,1], 8, (0, 1))
    v_hist = manual_hist(hsv[:,:,2], 8, (0, 1))

    # Thống kê màu RGB
    b_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    r_mean = np.mean(image[:,:,2])

    return np.concatenate([
        [h_mean, h_std, h_skew, s_mean, s_std, s_skew, v_mean, v_std, v_skew],
        [b_mean, g_mean, r_mean],
        h_hist, s_hist, v_hist
    ])

# ====== Hàm trích xuất đặc trưng kết cấu ======
def extract_texture_features(gray):
    # Đảm bảo giá trị pixel nằm trong khoảng 0-255
    gray = (gray - gray.min()) * (255.0 / (gray.max() - gray.min() + 1e-10))
    gray = gray.astype(np.uint8)
    
    # Tính toán GLCM thủ công (đơn giản hóa)
    glcm = np.zeros((256, 256), dtype=int)
    for i in range(gray.shape[0]-1):
        for j in range(gray.shape[1]-1):
            glcm[gray[i,j], gray[i,j+1]] += 1
            glcm[gray[i,j], gray[i+1,j]] += 1
    
    glcm = glcm / (glcm.sum() + 1e-10)
    
    # Tính các đặc trưng texture từ GLCM
    contrast = 0
    homogeneity = 0
    energy = 0
    for i in range(256):
        for j in range(256):
            contrast += glcm[i,j] * (i-j)**2
            homogeneity += glcm[i,j] / (1 + abs(i-j))
            energy += glcm[i,j]**2
    
    # LBP đơn giản
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i,j]
            code = 0
            code |= (gray[i-1,j-1] > center) << 7
            code |= (gray[i-1,j] > center) << 6
            code |= (gray[i-1,j+1] > center) << 5
            code |= (gray[i,j+1] > center) << 4
            code |= (gray[i+1,j+1] > center) << 3
            code |= (gray[i+1,j] > center) << 2
            code |= (gray[i+1,j-1] > center) << 1
            code |= (gray[i,j-1] > center) << 0
            lbp[i,j] = code
    
    lbp_hist = np.histogram(lbp, bins=8, range=(0, 256))[0]
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
    
    return np.concatenate([
        [contrast, homogeneity, energy],
        lbp_hist
    ])

# ====== Hàm trích xuất đặc trưng hình dạng ======
def extract_shape_features(binary):
    # Đảm bảo binary là 0 hoặc 1
    binary = (binary > 0).astype(np.uint8)
    
    # Tính moments thủ công
    m00 = binary.sum()
    if m00 == 0:
        return np.zeros(14)  # Trả về vector 0 nếu không có hình dạng
    
    # Tính tọa độ trọng tâm
    y, x = np.indices(binary.shape)
    m10 = (x * binary).sum()
    m01 = (y * binary).sum()
    cx = m10 / m00
    cy = m01 / m00
    
     # Tính central moments (7 Hu moments)
    mu20 = ((x - cx)**2 * binary).sum()
    mu02 = ((y - cy)**2 * binary).sum()
    mu11 = ((x - cx) * (y - cy) * binary).sum()
    mu30 = ((x - cx)**3 * binary).sum()
    mu03 = ((y - cy)**3 * binary).sum()
    mu12 = ((x - cx) * (y - cy)**2 * binary).sum()
    mu21 = ((x - cx)**2 * (y - cy) * binary).sum()
    
    # Tính 7 Hu moments
    hu1 = mu20 + mu02
    hu2 = (mu20 - mu02)**2 + 4*mu11**2
    hu3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2
    hu4 = (mu30 + mu12)**2 + (mu21 + mu03)**2
    hu5 = (mu30 - 3*mu12)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) + \
          (3*mu21 - mu03)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
    hu6 = (mu20 - mu02)*((mu30 + mu12)**2 - (mu21 + mu03)**2) + \
          4*mu11*(mu30 + mu12)*(mu21 + mu03)
    hu7 = (3*mu21 - mu03)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) - \
          (mu30 - 3*mu12)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
    
    # Log scale để giảm độ lớn
    hu = np.array([hu1, hu2, hu3, hu4, hu5, hu6, hu7])
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Tính diện tích và chu vi
    area = m00
    perimeter = 0
    
    # Tìm contour đơn giản
    contours = []
    visited = np.zeros_like(binary, dtype=bool)
    
    for i in range(1, binary.shape[0]-1):
        for j in range(1, binary.shape[1]-1):
            if binary[i,j] == 1 and not visited[i,j]:
                contour = []
                stack = [(i,j)]
                visited[i,j] = True
                
                while stack:
                    x, y = stack.pop()
                    contour.append((x,y))
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < binary.shape[0] and 0 <= ny < binary.shape[1]:
                                if binary[nx,ny] == 1 and not visited[nx,ny]:
                                    visited[nx,ny] = True
                                    stack.append((nx,ny))
                
                if len(contour) > 0:
                    contours.append(np.array(contour))
    
    if contours:
        main_contour = max(contours, key=len)
        perimeter = len(main_contour)
        
        # Tính circularity
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-10)
        
    else:
        perimeter = circularity = 0
    
    return np.concatenate([
        hu,
        [area, perimeter, circularity],
    ])


def zscore_normalize(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1e-8 
    normalized = (features - mean) / std
    return normalized

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
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        with open("zscore_params.pkl", "rb") as f:
            params = pickle.load(f)

        color_mean = params["mean"]["color"]
        color_std = params["std"]["color"]
        texture_mean = params["mean"]["texture"]
        texture_std = params["std"]["texture"]
        shape_mean = params["mean"]["shape"]
        shape_std = params["std"]["shape"]
        # Xử lý ảnh giống hệt như trong file extract
        img = np.array(Image.open(file.stream).convert('RGB'))
        img_pil = Image.fromarray(img).resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img = np.array(img_pil)
        
        # Chuyển sang grayscale
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        gray = gray.astype(np.float32)
        
        # Tạo binary mask
        binary = (gray > gray.mean()).astype(np.uint8)

        # Trích xuất đặc trưng
        f_color = extract_color_features(img)
        f_texture = extract_texture_features(gray)
        f_shape = extract_shape_features(binary)

        f_color_normalized = (f_color - np.array(color_mean)) / np.array(color_std)
        f_texture_normalized = (f_texture - np.array(texture_mean)) / np.array(texture_std)
        f_shape_normalized = (f_shape - np.array(shape_mean)) / np.array(shape_std)

        # Kết hợp các đặc trưng
        combined_query = np.concatenate([
            f_color_normalized, 
            f_texture_normalized , 
            f_shape_normalized
        ]).reshape(1, -1)


        # Tìm kiếm trong database
        results = []
        for doc in collection.find():
            db_features = np.array(doc["features"]).reshape(1, -1)
            sim = cosine_similarity(combined_query, db_features)[0][0]
            # Tạo URL đầy đủ cho ảnh
            image_filename = os.path.basename(doc["image_path"])
            category = os.path.basename(os.path.dirname(doc["image_path"]))
            image_url = f"http://127.0.0.1:5000/dataset_resized/{category}/{image_filename}"
            
            results.append({
                "image_url": image_url,
                "score": round(float((sim + 1) * 50), 4)  # Làm tròn 4 chữ số thập phân
            })

        # Sắp xếp và chỉ lấy 3 kết quả giống nhất
        results.sort(key=lambda x: x["score"], reverse=True)
        top_3_results = results[:3]
        
        return jsonify(top_3_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== Khởi chạy server =====
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)