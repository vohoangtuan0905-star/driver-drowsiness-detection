import cv2
import numpy as np
import tensorflow as tf
import pygame
from collections import deque

MODEL_PATH = r"../models/drowsiness_cnn.h5"
ALARM_PATH = r"../assets/alarm1.wav"
IMG_SIZE = (64, 64)

# =========================
# CẤU HÌNH NHẠY / ỔN ĐỊNH
# =========================
QUEUE_SIZE = 7
DROWSY_THRESHOLD = 4

# threshold cho model
# pred_value gan 1 -> Non Drowsy
# pred_value gan 0 -> Drowsy
HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.35

ENABLE_SOUND = True

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# SOUND
# =========================
alarm_sound = None
alarm_playing = False

if ENABLE_SOUND:
    try:
        pygame.mixer.init()
        alarm_sound = pygame.mixer.Sound(ALARM_PATH)
        print("Loaded alarm sound successfully.")
    except Exception as e:
        print(f"Khong khoi tao duoc am thanh: {e}")
        alarm_sound = None

# =========================
# BIẾN TRẠNG THÁI
# =========================
prediction_queue = deque(maxlen=QUEUE_SIZE)
drowsy_counter = 0
last_stable_label = 1   # mac dinh Non Drowsy

# =========================
# HÀM HỖ TRỢ
# =========================
def get_largest_face(faces):
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]

def crop_face_with_padding(frame, x, y, w, h, pad_ratio=0.15):
    h_img, w_img, _ = frame.shape
    pad = int(max(w, h) * pad_ratio)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def predict_face(face_bgr):
    global last_stable_label

    face_resized = cv2.resize(face_bgr, IMG_SIZE)

    # QUAN TRỌNG: đổi BGR -> RGB vì model train theo RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # KHONG chia 255 vi model da co Rescaling(1./255)
    face_input = np.expand_dims(face_rgb.astype("float32"), axis=0)

    pred_value = model.predict(face_input, verbose=0)[0][0]

    # Hysteresis threshold
    if pred_value >= HIGH_THRESHOLD:
        pred_label = 1   # Non Drowsy
    elif pred_value <= LOW_THRESHOLD:
        pred_label = 0   # Drowsy
    else:
        pred_label = last_stable_label

    last_stable_label = pred_label
    return float(pred_value), int(pred_label)

def update_alarm(is_drowsy):
    global alarm_playing

    if alarm_sound is None:
        return

    if is_drowsy and not alarm_playing:
        alarm_sound.play(-1)
        alarm_playing = True
    elif not is_drowsy and alarm_playing:
        alarm_sound.stop()
        alarm_playing = False

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc frame tu webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    status_text = "NO FACE"
    status_color = (255, 255, 0)
    raw_text = ""
    avg_text = ""
    final_text = ""

    largest_face = get_largest_face(faces)

    if largest_face is None:
        prediction_queue.clear()
        drowsy_counter = 0
        update_alarm(False)
    else:
        x, y, w, h = largest_face
        face_crop, (x1, y1, x2, y2) = crop_face_with_padding(frame, x, y, w, h)

        pred_value, pred_label = predict_face(face_crop)

        prediction_queue.append(pred_label)
        avg_pred = sum(prediction_queue) / len(prediction_queue)

        if avg_pred >= 0.5:
            final_label = "Non Drowsy"
            status_color = (0, 255, 0)
            drowsy_counter = 0
            update_alarm(False)
        else:
            final_label = "Drowsy"
            status_color = (0, 0, 255)
            drowsy_counter += 1
            update_alarm(drowsy_counter >= DROWSY_THRESHOLD)

        status_text = final_label
        raw_text = f"Raw: {pred_value:.4f}"
        avg_text = f"AvgPred: {avg_pred:.4f}"
        final_text = f"Final: {final_label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)

        cv2.putText(frame, avg_text, (x1, max(25, y1 - 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, raw_text, (x1, max(50, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, final_text, (x1, max(75, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.putText(frame, f"STATUS: {status_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    cv2.putText(frame, f"Drowsy Count: {drowsy_counter}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Driver Drowsiness Detection - Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()

if alarm_sound is not None and alarm_playing:
    alarm_sound.stop()

if ENABLE_SOUND:
    try:
        pygame.mixer.quit()
    except:
        pass