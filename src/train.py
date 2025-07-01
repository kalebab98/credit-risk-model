import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
import joblib

warnings.filterwarnings("ignore")
import os
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath)  # Use the filepath parameter here
    return df


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["CustomerId"])  # Drop non-numeric column
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf


def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    return metrics


def main():
    mlflow.set_experiment("fraud_detection")

    print("Loading data...")
    X, y = load_data("df_final_task4.csv")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    print(f"Logistic Regression Metrics: {lr_metrics}")

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print(f"Random Forest Metrics (default params): {rf_metrics}")

    print("Tuning Random Forest with RandomizedSearchCV...")
    best_rf, best_params = tune_random_forest(X_train, y_train)
    best_rf_metrics = evaluate_model(best_rf, X_test, y_test)
    print(f"Tuned Random Forest Best Params: {best_params}")
    print(f"Tuned Random Forest Metrics: {best_rf_metrics}")

     # Save the best tuned Random Forest model locally
    joblib.dump(best_rf, "best_random_forest_model.pkl")
    print("Saved best model to best_random_forest_model.pkl")

    print("Logging best model to MLflow...")
    with mlflow.start_run(run_name="RandomForest_best"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_rf_metrics)
        mlflow.sklearn.log_model(best_rf, "model")

    print("Training and tracking complete.")

    


if __name__ == "__main__":
    main()


