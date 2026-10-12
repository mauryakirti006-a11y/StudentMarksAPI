
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Sample training data
X = [[1], [2], [3], [4], [5], [6]]
y = [35, 40, 50, 60, 70, 80]

model = LinearRegression()
model.fit(X, y)


@app.route("/")
def home():
    return "Student Marks Prediction API is running!"


@app.route("/predict")
def predict():
    try:
        hours = float(request.args.get("hours", ""))

        if not 0 <= hours <= 24:
            return jsonify({
                "error": "Hours must be between 0 and 24"
            }), 400

        marks = model.predict([[hours]])[0]

        return jsonify({
            "predicted_marks": round(float(marks), 2)
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Provide a valid number of study hours"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
