import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
model = joblib.load('mhnb-3.joblib')


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/predict', methods=['POST'])
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


if __name__ == '__main__':
    app.run()
