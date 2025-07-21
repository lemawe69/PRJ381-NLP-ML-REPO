from ultralytics import YOLO
import cv2
import os
from Utils.Validation import load_image, validate_video

# Load YOLOv8 model
model = YOLO("model/yolov8n.pt")

# Image Object Detection
def detect_image(img_path, output_folder="Outputs/Image"):
    os.makedirs(output_folder, exist_ok=True)
    try:
        image = load_image(img_path)
        results = model.predict(source=img_path, save=False)
        annotated = results[0].plot()
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, annotated)
        print(f"Image saved to → {output_path}")
    except Exception as e:
        print(str(e))


# Video Object Detection
def detect_video(video_path, output_folder="Outputs/Video", output_name="Test.mp4"):
    os.makedirs(output_folder, exist_ok=True)
    try:
        validate_video(video_path)

        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(output_folder, output_name)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, save=False)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()
        print(f"Video saved to → {output_path}")

    except Exception as e:
        print(str(e))

# Run Object Detection
detect_image("inputs/images/Test.jpeg")
detect_video("inputs/videos/Test.mp4")
