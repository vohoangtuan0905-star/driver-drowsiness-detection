import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset_utils import load_split_csv, make_dataset

VAL_CSV = r"../data/val_split.csv"
MODEL_PATH = r"../models/drowsiness_cnn.h5"

if __name__ == "__main__":
    val_df = load_split_csv(VAL_CSV)

    print("=== VALIDATION DISTRIBUTION FROM CSV ===")
    print(val_df["class_name"].value_counts())
    print(val_df["label"].value_counts().sort_index())

    val_ds = make_dataset(val_df, batch_size=32, shuffle=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = val_df["label"].astype(int).values
    y_pred_probs = model.predict(val_ds, verbose=1).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n=== FIRST 20 TRUE LABELS ===")
    print(y_true[:20])

    print("\n=== FIRST 20 PRED PROBS ===")
    print(np.round(y_pred_probs[:20], 4))

    print("\n=== FIRST 20 PRED LABELS ===")
    print(y_pred[:20])

    class_names = ["Drowsy", "Non Drowsy"]

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    os.makedirs("../models", exist_ok=True)
    with open("../models/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../models/confusion_matrix.png")
    plt.show()

    print("Saved classification report to ../models/classification_report.txt")
    print("Saved confusion matrix to ../models/confusion_matrix.png")