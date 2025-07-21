import os
import cv2

# Validate if image exists
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.imread(path)

# Validate if video exists
def validate_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    return True
