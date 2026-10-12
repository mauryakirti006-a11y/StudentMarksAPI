
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set MLflow experiment
mlflow.set_experiment("Student Marks Prediction")

# Sample student dataset
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "marks": [35, 40, 50, 60, 70, 80, 85, 90, 95, 98]
})

X = data[["hours"]]
y = data["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("training_rows", len(X_train))

    # Log evaluation metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log and register model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="student_marks_model",
        registered_model_name="StudentMarksModel"
    )

    print("Model training completed successfully!")
    print("MSE:", mse)
    print("R2 Score:", r2)
