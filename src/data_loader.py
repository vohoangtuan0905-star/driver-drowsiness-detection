import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = r"../data/Driver Drowsiness Dataset (DDD)"

CLASS_MAP = {
    "Drowsy": 0,
    "Non Drowsy": 1
}

def build_dataframe():
    rows = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Khong tim thay thu muc: {class_dir}")

        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(class_dir, file_name)
                rows.append({
                    "filepath": file_path,
                    "label": label,
                    "class_name": class_name
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Khong tim thay anh nao trong dataset.")

    return df

def split_dataframe(df, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

if __name__ == "__main__":
    df = build_dataframe()

    print("=== PHAN BO TOAN BO DATASET ===")
    print(df["class_name"].value_counts())

    train_df, val_df = split_dataframe(df)

    print("\n=== PHAN BO TRAIN ===")
    print(train_df["class_name"].value_counts())

    print("\n=== PHAN BO VALIDATION ===")
    print(val_df["class_name"].value_counts())