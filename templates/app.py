import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image

from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  


MODEL_PATH = os.path.join(os.path.dirname(__file__), "flowers_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Pink Primrose",
    "Hard-Leaved Pocket Orchid",
    "Canterbury Bells",
    "Sweet Pea",
    "English Marigold",
    # дописать
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
    
        if "file" not in request.files:
            return "Файл не загружен", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "Пустое имя файла", 400

        image_bytes = file.read()
      
        img = Image.open(io.BytesIO(image_bytes))

        
        img = img.resize((224, 224)) 
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)


        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        if pred_idx < len(class_names):
            pred_label = class_names[pred_idx]
        else:
            pred_label = "Unknown"

        
        return render_template("index.html", prediction=pred_label)

    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run()
