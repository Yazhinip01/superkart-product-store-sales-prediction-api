
# Import necessary libraries
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
product_store_sales_api = Flask(__name__)

# Load the trained model pipeline
model = joblib.load("product_store_sales_prediction_model_v1_0.joblib")


# -------------------- Home Route -------------------- #
@product_store_sales_api.route("/", methods=["GET"])
def home():
    return "Welcome to the Product Store Sales Prediction API!"


# -------------------- Single Prediction -------------------- #
@product_store_sales_api.route("/v1/predict", methods=["POST"])
def predict_product_store_sales():

    # Get JSON input
    product_data = request.get_json()

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Product_Weight": product_data["Product_Weight"],
        "Product_Sugar_Content": product_data["Product_Sugar_Content"],
        "Product_Allocated_Area": product_data["Product_Allocated_Area"],
        "Product_Type": product_data["Product_Type"],
        "Product_MRP": product_data["Product_MRP"],
        "Store_Id": product_data["Store_Id"],
        "Store_Establishment_Year": product_data["Store_Establishment_Year"],
        "Store_Size": product_data["Store_Size"],
        "Store_Location_City_Type": product_data["Store_Location_City_Type"],
        "Store_Type": product_data["Store_Type"]
    }])

    # Predict
    predicted_sales = model.predict(input_data)[0]

    # Convert NumPy datatype to Python float
    predicted_sales = round(float(predicted_sales), 2)

    # Return prediction
    return jsonify(
        {
            "Predicted_Product_Store_Sales_Total": predicted_sales
        }
    )


# -------------------- Batch Prediction -------------------- #
@product_store_sales_api.route("/v1/predict_batch", methods=["POST"])
def predict_batch():

    # Read uploaded CSV file
    file = request.files["file"]

    input_data = pd.read_csv(file)

    # Make predictions
    predictions = model.predict(input_data)

    # Convert predictions to Python float
    predictions = [round(float(x), 2) for x in predictions]

    # Return predictions
    return jsonify(
        {
            "Predictions": predictions
        }
    )


# -------------------- Run API -------------------- #
if __name__ == "__main__":
    product_store_sales_api.run(debug=True)
