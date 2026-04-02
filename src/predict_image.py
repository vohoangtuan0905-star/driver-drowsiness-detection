import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = r"../models/drowsiness_cnn.h5"
IMG_PATH = r"../data/Driver Drowsiness Dataset (DDD)/Drowsy/A0010.png"
IMG_SIZE = (64, 64)

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)

    img = load_img(IMG_PATH, target_size=IMG_SIZE)
    img_array = img_to_array(img)

    # KHONG chia 255 o day vi model da co Rescaling(1./255)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Non Drowsy" if prediction > 0.5 else "Drowsy"

    print(f"Image path: {IMG_PATH}")
    print(f"Prediction value: {prediction:.6f}")
    print(f"Predicted label: {label}")