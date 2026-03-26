import os

DATASET_PATH = r"../data/Driver Drowsiness Dataset (DDD)"

def count_images(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                count += 1
    return count

if __name__ == "__main__":
    drowsy_path = os.path.join(DATASET_PATH, "Drowsy")
    non_drowsy_path = os.path.join(DATASET_PATH, "Non Drowsy")

    print("=== KIEM TRA DATASET ===")
    print("Drowsy folder exists:", os.path.exists(drowsy_path))
    print("Non Drowsy folder exists:", os.path.exists(non_drowsy_path))

    drowsy_count = count_images(drowsy_path)
    non_drowsy_count = count_images(non_drowsy_path)

    print(f"So anh Drowsy: {drowsy_count}")
    print(f"So anh Non Drowsy: {non_drowsy_count}")
    print(f"Tong so anh: {drowsy_count + non_drowsy_count}")