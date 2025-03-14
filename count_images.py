import os

def count_images(directory):
    total = 0
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        num_files = len(os.listdir(category_path))
        print(f"{category}: {num_files} ảnh")
        total += num_files
    print(f"📌 Tổng số ảnh sau resize: {total}")

count_images("dataset_resized")
