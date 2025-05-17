import os
import numpy as np
from pymongo import MongoClient
from PIL import Image
import math

# Cấu hình MongoDB
MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_0"]

# Thư mục ảnh
DATASET_PATH = "dataset_resized"
TARGET_SIZE = (256, 256)

# Trọng số của các đặc trưng
WEIGHT_COLOR = 0.2
WEIGHT_TEXTURE = 0.35
WEIGHT_SHAPE = 0.45

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

# ====== Hàm phát hiện biên bằng Sobel ======
def sobel_edge_detection(gray):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Padding ảnh
    padded = np.pad(gray, ((1, 1), (1, 1)), mode='constant')
    edges_x = np.zeros_like(gray)
    edges_y = np.zeros_like(gray)
    
    # Áp dụng kernel Sobel (chỉ xử lý các pixel bên trong)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            patch = padded[i:i+3, j:j+3]
            edges_x[i, j] = np.sum(patch * kernel_x)
            edges_y[i, j] = np.sum(patch * kernel_y)
    
    edges = np.sqrt(edges_x**2 + edges_y**2)
    return edges

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
    
    # LBP đơn giản (chỉ xử lý các pixel bên trong)
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
    
    # Tính central moments
    mu20 = ((x - cx)**2 * binary).sum() / m00
    mu02 = ((y - cy)**2 * binary).sum() / m00
    mu11 = ((x - cx) * (y - cy) * binary).sum() / m00
    
    # Tính Hu moments (đơn giản hóa)
    hu1 = mu20 + mu02
    hu2 = (mu20 - mu02)**2 + 4*mu11**2
    hu = [hu1, hu2] + [0]*5  # Giả lập 7 Hu moments
    
    # Tính diện tích và chu vi
    area = m00
    perimeter = 0
    
    # Tìm contour đơn giản (chỉ xử lý các pixel bên trong)
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
        
        # Tính solidity (đơn giản hóa)
        hull_area = area  # Giả lập convex hull area
        solidity = area / (hull_area + 1e-10)
    else:
        perimeter = circularity = solidity = 0
    
    return np.concatenate([
        hu,
        [area, perimeter, circularity, solidity],
        [0]*7  # Giả lập Zernike moments
    ])

# ====== Hàm chuẩn hóa đặc trưng về khoảng [0,1] ======
def normalize_features(features):
    min_val = features.min()
    max_val = features.max()
    if max_val - min_val > 1e-10:
        return (features - min_val) / (max_val - min_val)
    return features

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
            # Đọc ảnh bằng PIL
            img = np.array(Image.open(img_path).convert('RGB'))
            
            # Resize ảnh (đảm bảo đúng kích thước)
            img_pil = Image.fromarray(img).resize(TARGET_SIZE, Image.Resampling.LANCZOS) 
            img = np.array(img_pil)
            
            # Chuyển sang grayscale và đảm bảo kiểu dữ liệu
            gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            gray = gray.astype(np.float32)
            
            # Tạo binary mask đơn giản (thay cho Chan-Vese)
            binary = (gray > gray.mean()).astype(np.uint8)

            # Trích xuất đặc trưng
            f_color = extract_color_features(img)
            f_texture = extract_texture_features(gray)
            f_shape = extract_shape_features(binary)

            # Áp dụng trọng số
            f_color_weighted = WEIGHT_COLOR * f_color
            f_texture_weighted = WEIGHT_TEXTURE * f_texture
            f_shape_weighted = WEIGHT_SHAPE * f_shape

            # Chuẩn hóa các đặc trưng
            f_color_normalized = normalize_features(f_color_weighted)
            f_texture_normalized = normalize_features(f_texture_weighted)
            f_shape_normalized = normalize_features(f_shape_weighted)

            # Kết hợp các đặc trưng
            features_normalized = np.concatenate([
                f_color_normalized, 
                f_texture_normalized, 
                f_shape_normalized
            ])

            # Lưu vào MongoDB
            collection.insert_one({
                "image_path": img_path,
                "category": folder,
                "features": features_normalized.tolist()
            })
            print("✅ Xử lý:", img_path)
        except Exception as e:
            print("❌ Lỗi ảnh:", img_path, "|", str(e))

print("🎉 Đã hoàn tất trích xuất và lưu đặc trưng!")