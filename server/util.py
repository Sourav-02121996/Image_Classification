"""Author: Sourav Das

Utility helpers for loading artifacts, preprocessing images, and running predictions.
"""

import joblib
import json
import numpy as np
import base64
import cv2
from pathlib import Path
from wavelet import w2d

# Project paths used for model artifacts and OpenCV cascades.
_BASE_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _BASE_DIR / "artifacts"
_OPENCV_DIR = _BASE_DIR / "opencv"
_TEST_IMAGES_DIR = _ARTIFACTS_DIR / "test_images"

# Label mappings loaded from artifacts.
__class_name_to_number = {}
__class_number_to_name = {}

# Lazily-loaded model instance.
__model = None

def classify_image(image_base64_data, file_path=None):

    # Crop faces with >=2 eyes for robustness before feature extraction.
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        # Resize the raw image to a fixed size for consistent feature shape.
        scalled_raw_img = cv2.resize(img, (32, 32))
        # Wavelet transform captures edge/texture details.
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        # Stack raw + wavelet features into one column vector.
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        # Predict class label and probabilities.
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Load label mappings for consistent class ids across training/inference.
    with open(_ARTIFACTS_DIR / "class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        model_path = _ARTIFACTS_DIR / "saved_model.pkl"
        if not model_path.exists():
            # Backward-compat for earlier filename.
            fallback_path = _ARTIFACTS_DIR / "save_model.pkl"
            if fallback_path.exists():
                model_path = fallback_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found. Expected 'saved_model.pkl' or "
                f"'save_model.pkl' in {_ARTIFACTS_DIR}"
            )
        # Load the trained classifier once.
        with open(model_path, "rb") as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    # Decode base64 -> bytes -> NumPy array, then OpenCV image.
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    # Load Haar cascades for face/eye detection.
    face_cascade = cv2.CascadeClassifier(
        str(_OPENCV_DIR / "haarcascades/haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        str(_OPENCV_DIR / "haarcascades/haarcascade_eye.xml")
    )

    if image_path:
        # Resolve relative paths against project and artifacts directories.
        path = Path(image_path)
        if not path.is_absolute() and not path.exists():
            candidate = (_BASE_DIR / path).resolve()
            if candidate.exists():
                path = candidate
            else:
                candidate = (_ARTIFACTS_DIR / path).resolve()
                if candidate.exists():
                    path = candidate
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {path}")
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    # Detect faces on grayscale for better cascade performance.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                # Keep only faces with at least two detected eyes.
                cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    with open(_ARTIFACTS_DIR / "b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    # Simple smoke test for local images.
    load_saved_artifacts()

    print(classify_image(get_b64_test_image_for_virat(), None))

    # print(classify_image(None, str(_TEST_IMAGES_DIR / "federer1.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "federer2.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "virat1.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "virat2.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "virat3.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "serena1.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "serena2.jpg")))
    # print(classify_image(None, str(_TEST_IMAGES_DIR / "sharapova1.jpg")))
