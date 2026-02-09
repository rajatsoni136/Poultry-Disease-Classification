from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from pathlib import Path
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = tf.keras.models.load_model("artifacts/training/model.h5")

# Initialize the Model Logic
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # This button triggers the training pipeline we built earlier
    os.system("python stage_03_training.py")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.files['image']
    filename = "inputImage.jpg"
    image.save(filename)

    # 1. Load the image
    img = tf.keras.utils.load_img(filename, target_size=(224, 224))
    
    # 2. Convert to array & Normalize (Just like we did in training)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Vital step! VGG16 expects 0-1 range

    # 3. Predict
    prediction = clApp.classifier.predict(img_array)
    
    # 4. Decode the result (0 = Coccidiosis, 1 = Healthy)
    # The argmax finds which slot has the highest number
    result_index = np.argmax(prediction, axis=1)

    if result_index[0] == 0:
        prediction_label = "Coccidiosis"
    else:
        prediction_label = "Healthy"

    return jsonify([{"image": prediction_label}])

if __name__ == "__main__":
    # Use the PORT environment variable if available, otherwise default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)