import tensorflow as tf

DATASET_PATH = r"../data/Driver Drowsiness Dataset (DDD)"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("=== KIEM TRA TENSORFLOW LOADER ===")
print("Class names:", dataset.class_names)

for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First 10 labels:", labels[:10].numpy())