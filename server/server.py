"""Author: Sourav Das

Flask API for sports celebrity image classification.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from . import util

app = Flask(__name__)
# Enable CORS for browser-based clients (static frontend on a different origin).
CORS(app)
# Load model artifacts at import time for gunicorn workers.
util.load_saved_artifacts()


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    # Expect base64-encoded image data sent from the UI.
    image_data = request.form['image_data']

    # Run inference and serialize model output as JSON.
    response = jsonify(util.classify_image(image_data))

    # Allow browser clients to call this endpoint (simple CORS support).
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    # Start the development server.
    app.run(port=5000)
