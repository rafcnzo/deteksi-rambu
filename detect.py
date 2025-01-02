import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from plyer import notification
import datetime

# Initialize YOLO model
model = YOLO("runs/detect/train12/weights/best.pt")

# Class names for predictions
classNames = ["larangan berhenti", "larangan masuk bagi kendaraan bermotor dan tidak bermotor", 
              "larangan parkir", "lampu hijau", "lampu kuning", "lampu merah", "larangan belok kanan", "larangan belok kiri", 
              "larangan berjalan terus wajib berhenti sesaat", "larangan memutar balik", "peringatan alat pemberi isyarat lalu lintas", 
              "peringatan banyak pejalan kaki menggunakan zebra cross", "peringatan pintu perlintasan kereta api", "peringatan simpang tiga sisi kiri", 
              "peringatan penegasan rambu tambahan", "perintah masuk jalur kiri", "perintah pilihan memasuki salah satu jalur", 
              "petunjuk area parkir", "petunjuk lokasi pemberhentian bus", "petunjuk lokasi putar balik", "petunjuk-penyeberangan-pejalan-kaki"]

# Notification-related variables
listnotify = ["lampu hijau", "lampu merah", "lampu kuning", "larangan berhenti", "larangan belok kanan", "larangan belok kiri"]
t_lastnotify = datetime.datetime(year=2024, month=1, day=1)
interval = datetime.timedelta(seconds=1)
max_interval = 7

def notifyme(class_name, confidence):
    """Function to send desktop notifications."""
    notification.notify(
        title="Detect Rambu",
        message=f"{class_name} terdeteksi dengan confidence {confidence:.2f}",
        app_name="Detect Rambu",
        timeout=5
    )

def process_frame(frame):
    """Run YOLO detection on a single frame."""
    global t_lastnotify, max_interval
    results = model(frame, stream=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get confidence and class name
            confidence = box.conf[0].item()
            class_index = int(box.cls[0])
            class_name = classNames[class_index]

            # Display class and confidence on frame
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send notifications for listed classes
            if class_name in listnotify:
                t_currtime = datetime.datetime.now()
                if (t_currtime - t_lastnotify) > interval:
                    notifyme(class_name, confidence)
                    t_lastnotify = t_currtime
                    max_interval -= 1
                    if max_interval == 0:
                        max_interval = 7

    return frame

# Streamlit app setup
st.title("Traffic Sign Detection with YOLOv8")
st.markdown("Detect traffic signs from a video feed using a YOLOv8 model.")

# Video input source
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Convert uploaded video into a format OpenCV can read
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Open video file with OpenCV
    video_cap = cv2.VideoCapture(video_path)

    # Display video frames with predictions
    stframe = st.empty()

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            st.write("End of video or error reading frame.")
            break

        # Process frame for YOLO detection
        processed_frame = process_frame(frame)

        # Convert processed frame (BGR to RGB for Streamlit display)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        stframe.image(processed_frame, channels="RGB")

    video_cap.release()
    st.success("Video processing complete.")