import numpy as np
import tensorflow as tf
from dataset_utils import load_split_csv, make_dataset

TRAIN_CSV = r"../data/train_split.csv"
VAL_CSV = r"../data/val_split.csv"

if __name__ == "__main__":
    train_df = load_split_csv(TRAIN_CSV)
    val_df = load_split_csv(VAL_CSV)

    print("=== TRAIN DISTRIBUTION ===")
    print(train_df["class_name"].value_counts())
    print(train_df["label"].value_counts().sort_index())

    print("\n=== VAL DISTRIBUTION ===")
    print(val_df["class_name"].value_counts())
    print(val_df["label"].value_counts().sort_index())

    print("\n=== FIRST 10 VAL ROWS ===")
    print(val_df[["filepath", "label", "class_name"]].head(10))

    val_ds = make_dataset(val_df, batch_size=8, shuffle=False)

    for images, labels in val_ds.take(1):
        print("\nBatch image shape:", images.shape)
        print("Batch labels:", labels.numpy())