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
                rows.append({
                    "filepath": os.path.join(class_dir, file_name),
                    "label": label,
                    "class_name": class_name
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Khong tim thay anh nao trong dataset.")

    return df

if __name__ == "__main__":
    df = build_dataframe()

    print("=== TOAN BO DATASET ===")
    print(df["class_name"].value_counts())

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    os.makedirs("../data", exist_ok=True)
    train_df.to_csv("../data/train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv("../data/val_split.csv", index=False, encoding="utf-8-sig")

    print("\n=== TRAIN SPLIT ===")
    print(train_df["class_name"].value_counts())

    print("\n=== VALIDATION SPLIT ===")
    print(val_df["class_name"].value_counts())

    print("\nDa luu:")
    print("../data/train_split.csv")
    print("../data/val_split.csv")