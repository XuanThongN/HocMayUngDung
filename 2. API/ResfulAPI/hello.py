import os
from PIL import Image

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'nxtsecuritykey'
jwt = JWTManager(app)

flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
model = joblib.load('mhnb-3.joblib')


@app.route('/login', methods=['POST'])
def login():
    # Nhận dữ liệu từ request
    data = request.get_json(force=True)
    username = data['username']
    password = data['password']
    if username != 'test' or password != 'test':
        return jsonify({'message': 'Bad username or password'}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/predict', methods=['POST'])
@jwt_required
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json(force=True)
        features = data['features']
        prediction = model.predict([features])
        result = {"prediction": int(prediction[0]), "flower_name": flower_names[int(prediction[0])]}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request.', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400
    if file:
        # Save file into folder upload
        file.save('upload/' + file.filename)
        # Open the image file and convert it to a numpy array
        img = Image.open(os.path.join('upload/', file.filename))
        img_vector = np.array(img)
        return 'File uploaded and converted to vector successfully.', 200


if __name__ == '__main__':
    app.run()
