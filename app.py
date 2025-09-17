from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load your model
model = load_model("brain_tumor_model.h5")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Expect image as bytes in request
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess (resize, normalize as per your training)
    img = cv2.resize(img, (128, 128))  # change size to match training
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Run prediction
    prediction = model.predict(img)
    result = prediction.tolist()

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
