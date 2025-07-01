import pandas as pd
import joblib
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def main():
    model_path = "best_random_forest_model.pkl"
    data_path = "preprocess.csv"  # Make sure this matches your actual data file name

    print("Loading model...")
    model = load_model(model_path)

    print("Loading data...")
    new_data = load_data(data_path)

    # Drop CustomerId if it exists — typically not needed for prediction
    if 'CustomerId' in new_data.columns:
        new_data = new_data.drop(columns=['CustomerId'])

    print("Making predictions...")
    predictions = model.predict(new_data)

    # Add predictions to the dataframe
    new_data["predicted_is_high_risk"] = predictions

    # Save or show results
    output_path = "predictions_output.csv"
    new_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
