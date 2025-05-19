import os
import numpy as np
from pymongo import MongoClient
from PIL import Image
import pickle

MONGO_URI = "mongodb+srv://zeros0000:d21httt06@database0.d6lmc.mongodb.net/?retryWrites=true&w=majority&appName=Database0"
client = MongoClient(MONGO_URI)
db = client["Database0"]
collection = db["image_features_6"]

WEIGHT_COLOR = 0.2
WEIGHT_TEXTURE = 0.45
WEIGHT_SHAPE = 0.35

def load_features(collection):
    documents = list(collection.find({}))
    features = np.array([doc["features"] for doc in documents])
    return documents, features

def zscore_normalize(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1e-8 
    normalized = (features - mean) / std
    return normalized, mean, std

def save_normalized_features(documents, normalized_features, target_collection):
    new_documents = []
    for original_doc, norm_feat in zip(documents, normalized_features):
        new_doc = {
            "image_path": original_doc["image_path"],
            "category": original_doc.get("category", None),
            "features": norm_feat.tolist()
        }
        new_documents.append(new_doc)

    if new_documents:
        target_collection.insert_many(new_documents)
        print(f"[✓] Đã lưu {len(new_documents)} bản ghi vào collection mới.")

def save_zscore_params(mean, std, filepath="zscore_params.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    print(f"[✓] Đã lưu tham số Z-score vào: {filepath}")

def main():
    source_collection_name = "image_features_6"
    target_collection_name = "6"
    zscore_param_file = "zscore_params.pkl"

    print("[•] Kết nối MongoDB...")
    source_collection = db[source_collection_name]
    target_collection = db[target_collection_name]

    print("[•] Đọc dữ liệu features gốc...")
    documents = list(source_collection.find({}))
    color_features = np.array([doc["color_features"] for doc in documents])
    texture_features = np.array([doc["texture_features"] for doc in documents])
    shape_features = np.array([doc["shape_features"] for doc in documents])

    print("[•] Chuẩn hoá Z-score...")
    normalized_color, color_mean, color_std = zscore_normalize(color_features)
    normalized_texture, texture_mean, texture_std = zscore_normalize(texture_features)
    normalized_shape, shape_mean, shape_std = zscore_normalize(shape_features)
    normalized_features = np.concatenate((normalized_color ,
                           normalized_texture ,
                           normalized_shape ) , axis=1)

    print("[•] Lưu features chuẩn hoá vào collection mới...")
    save_normalized_features(documents, normalized_features, target_collection)
    print("[•] Lưu mean và std vào file...")
    mean = {
        "color": color_mean.tolist(),
        "texture": texture_mean.tolist(),
        "shape": shape_mean.tolist()
    }
    std = {
        "color": color_std.tolist(),
        "texture": texture_std.tolist(),
        "shape": shape_std.tolist()
    }
    save_zscore_params(mean, std, zscore_param_file)

    print("[✓] Hoàn tất!")

if __name__ == "__main__":
    main()