import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset_utils import load_split_csv, make_dataset

TRAIN_CSV = r"../data/train_split.csv"
VAL_CSV = r"../data/val_split.csv"

BATCH_SIZE = 32
EPOCHS = 10

if __name__ == "__main__":
    train_df = load_split_csv(TRAIN_CSV)
    val_df = load_split_csv(VAL_CSV)

    print("=== TRAIN DISTRIBUTION ===")
    print(train_df["class_name"].value_counts())
    print(train_df["label"].value_counts().sort_index())

    print("\n=== VALIDATION DISTRIBUTION ===")
    print(val_df["class_name"].value_counts())
    print(val_df["label"].value_counts().sort_index())

    train_ds = make_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False)

    model = Sequential([
        Rescaling(1./255, input_shape=(64, 64, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    os.makedirs("../models", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="../models/drowsiness_cnn.h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../models/training_history.png")
    plt.show()

    print("Train xong.")
    print("Model da luu tai: ../models/drowsiness_cnn.h5")
    print("Bieu do da luu tai: ../models/training_history.png")