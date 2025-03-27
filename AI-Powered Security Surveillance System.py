import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import smtplib
import requests
import time
import threading
from flask import Flask, render_template, Response
import firebase_admin
from firebase_admin import credentials, db
import redis
import json
import datetime
from kafka import KafkaProducer
from blockchain import Blockchain
from twilio.rest import Client
import dlib

# Initialize Firebase for Cloud Storage & Alert Logging
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-database-url.firebaseio.com'
})
alerts_ref = db.reference("alerts")

# Initialize Redis for real-time threat intelligence
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Kafka for distributed alert processing
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Initialize Blockchain for tamper-proof logging
security_blockchain = Blockchain()

# Load YOLOv8 model for object detection
yolo_model = YOLO('yolov8x.pt')  # High-accuracy YOLO model

# Load DeepSORT for object tracking
deep_sort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")

# Load Anomaly Detection Model
anomaly_model = load_model("advanced_anomaly_detection.h5")
scaler = StandardScaler()

# Load Facial Recognition Model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Flask App for Live Video Feed
app = Flask(__name__)

def send_alert(message):
    print("ALERT:", message)
    alert_data = {
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    # Log alert to Firebase
    alerts_ref.push(alert_data)
    # Store alert in Redis for real-time monitoring
    redis_client.publish("security_alerts", json.dumps(alert_data))
    # Publish alert to Kafka for further processing
    producer.send("security_alerts", alert_data)
    # Add alert to Blockchain for security logging
    security_blockchain.add_new_transaction(alert_data)
    security_blockchain.mine()
    # Send SMS alert using Twilio
    client = Client("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")
    client.messages.create(body=message, from_="+1234567890", to="+0987654321")
    # Send Email Alert
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("your_email@gmail.com", "your_password")
    server.sendmail("your_email@gmail.com", "security_team@gmail.com", message)
    server.quit()

def process_video():
    cap = cv2.VideoCapture("security_footage.mp4")  # Replace with 0 for webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Object Detection
        results = yolo_model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo_model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append([x1, y1, x2, y2, confidence, label])
        # Track Objects
        tracker_outputs = deep_sort.update(np.array(detections), frame)
        for track in tracker_outputs:
            x1, y1, x2, y2, track_id, label = track
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Facial Recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
        # Anomaly Detection
        frame_resized = cv2.resize(frame, (64, 64)) / 255.0
        frame_reshaped = np.expand_dims(frame_resized, axis=0)
        anomaly_score = anomaly_model.predict(frame_reshaped)
        if anomaly_score > 0.8:
            send_alert("Suspicious Activity Detected! Check Security Feed.")
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=process_video).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
