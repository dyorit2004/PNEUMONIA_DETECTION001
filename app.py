import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("pneumonia2.h5")
UPLOAD_FOLDER = os.path.join('static','uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prepare_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((120, 120))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 120, 120, 3)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            image = prepare_image(file_path)
            prediction = float(model.predict(image)[0][0])
            label = "Pneumonia" if prediction > 0.5 else "Normal"
            image_url=url_for('static',filename='uploads/' + filename)
            return render_template("index.html", prediction=label, image_url=image_url)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

