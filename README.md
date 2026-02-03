Author: Sourav Das

# Sports Celebrity Image Classifier

This project classifies a face image into one of five sports celebrities using a
computer-vision preprocessing pipeline and a trained scikit-learn model. It also
includes a simple web UI that uploads an image and displays the predicted class
with per-class probabilities.

## Features
- Face + eye detection with OpenCV Haar cascades
- Crops faces only when two eyes are detected (better reliability)
- Wavelet-transform features combined with raw pixels
- Flask API for inference
- Static web UI for image upload and result display

## Supported Classes
- lionel_messi
- maria_sharapova
- roger_federer
- serena_williams
- virat_kohli

## Tech Stack
- Python, Flask
- OpenCV, NumPy, PyWavelets
- scikit-learn + joblib (model training/serialization)
- HTML/CSS/JS (Bootstrap + Dropzone for UI)

## Repository Structure
- `server/` - Flask API and inference utilities
- `server/artifacts/` - Trained model and class dictionary
- `server/opencv/` - Haar cascade XML files
- `UI/` - Static frontend (open `app.html` in a browser)
- `model/` - Jupyter notebooks for data cleaning and model training
- `model/dataset/` - Local dataset folders (if present)

## Setup
### 1) Create and activate a virtual environment
Example (macOS/Linux):
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
Server/runtime dependencies are listed in the root `requirements.txt`. Notebook
dependencies are listed in `model/requirements.txt`.

```
pip install -r requirements.txt
pip install -r model/requirements.txt
```

## Run the API Server
From the project root:
```
python server/server.py
```
The API runs at `http://127.0.0.1:5000`.

## Run the Web UI
Open `UI/app.html` directly in a browser, or serve the folder:
```
cd UI
python -m http.server 8000
```
Then visit `http://127.0.0.1:8000/app.html`.

## API Usage
### Endpoint
`POST /classify_image`

### Form Data
- `image_data`: base64 data URL (e.g., `data:image/jpeg;base64,...`)

### Response (example)
```
[
  {
    "class": "virat_kohli",
    "class_probability": [1.05, 12.67, 22.0, 4.5, 91.56],
    "class_dictionary": {
      "lionel_messi": 0,
      "maria_sharapova": 1,
      "roger_federer": 2,
      "serena_williams": 3,
      "virat_kohli": 4
    }
  }
]
```

## Model Training Notes
- `model/data_cleaning.ipynb`: explores face/eye detection and crops valid faces.
- `model/sports_celebrity_classification.ipynb`: builds features, trains models,
  evaluates accuracy, and saves artifacts.
- Trained model is saved in `server/artifacts/save_model.pkl` (or
  `saved_model.pkl`), and class mapping in `server/artifacts/class_dictionary.json`.

## Troubleshooting
- Empty results usually mean no face (or fewer than two eyes) detected. Use a
  clear, frontal face image with good lighting.
- If you see a model load error, ensure the `.pkl` file exists in
  `server/artifacts/`.
- If the UI cannot reach the API, confirm the server is running on port 5000.

## Notes
- The inference pipeline requires detectable eyes; heavily obstructed faces may
  not classify.
- This repository currently uses a local `.venv` (Python 3.14) for development,
  but any compatible Python version should work.

## License
MIT License. See `LICENSE`.
