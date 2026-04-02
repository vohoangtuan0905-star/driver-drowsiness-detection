import pandas as pd
import tensorflow as tf

IMG_SIZE = (64, 64)

def load_split_csv(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"filepath", "label", "class_name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} thieu cot can thiet: {required_cols}")

    return df

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label = tf.cast(label, tf.float32)
    return image, label

def make_dataset(df, batch_size=32, shuffle=False):
    filepaths = df["filepath"].astype(str).values
    labels = df["label"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds