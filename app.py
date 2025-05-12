import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load Model & Classes
MODEL_PATH = "model/seed_classifier.h5"
LABELS_PATH = "model/label_classes.npy"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Label classes file not found!")

model = load_model(MODEL_PATH)
class_labels = np.load(LABELS_PATH, allow_pickle=True)

# ✅ Ensure Upload Folder Exists
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # ✅ Load & Preprocess Image
        img = load_img(file_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return render_template("index.html", prediction=predicted_class, confidence=confidence, image_path=file_path)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
