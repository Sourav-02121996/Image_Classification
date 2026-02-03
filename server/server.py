"""Author: Sourav Das

Flask API for sports celebrity image classification.
"""

from flask import Flask, request, jsonify
from . import util

app = Flask(__name__)


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
    # Load model artifacts once at startup.
    util.load_saved_artifacts()
    # Start the development server.
    app.run(port=5000)
