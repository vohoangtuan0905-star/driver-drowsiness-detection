import cv2
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = r"../models/drowsiness_cnn.h5"
IMG_SIZE = (64, 64)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Haar Cascade de phat hien khuon mat
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Queue de lam muot du doan
prediction_queue = deque(maxlen=15)

# Mo webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc frame tu webcam")
        break

    # Chuyen sang anh xam de detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    status_text = "NO FACE"
    status_color = (255, 255, 0)

    for (x, y, w, h) in faces:
        # Cat vung khuon mat
        face = frame[y:y+h, x:x+w]

        # Resize dung kich thuoc model
        face_resized = cv2.resize(face, IMG_SIZE)

        # KHONG chia 255 vi model da co Rescaling(1./255)
        face_input = np.expand_dims(face_resized, axis=0)

        # Predict
        pred_value = model.predict(face_input, verbose=0)[0][0]

        # Quy uoc:
        # pred_value > 0.5 => Non Drowsy
        # pred_value <= 0.5 => Drowsy
        pred_label = 1 if pred_value > 0.5 else 0

        # Dua vao queue de voting
        prediction_queue.append(pred_label)

        # Tinh trung binh queue
        avg_pred = sum(prediction_queue) / len(prediction_queue)

        if avg_pred > 0.5:
            final_label = "Non Drowsy"
            status_color = (0, 255, 0)
        else:
            final_label = "Drowsy"
            status_color = (0, 0, 255)

        status_text = final_label

        # Ve khung mat
        cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)

        # Hien ket qua thô
        cv2.putText(
            frame,
            f"Raw: {pred_value:.4f}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        # Hien ket qua sau voting
        cv2.putText(
            frame,
            f"Final: {final_label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        # Chi xu ly 1 khuon mat dau tien
        break

    # Hien trang thai tong quat
    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2
    )

    cv2.imshow("Driver Drowsiness Detection - Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()