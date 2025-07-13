import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.preprocessing.preprocess import (
    process_dataset, 
    feature_engineering, 
    split_train_test_data,
    get_features_labels
)
from src.model_training.training import get_best_lightgbm_model, export_model
from src.evaluation.evaluate_model import calculate_metrics, save_metrics
from datetime import date

def main():
    print("Reading and processing the dataset\n")
    df = process_dataset("data/raw/")

    print("Engineering new features\n")
    df = feature_engineering(df)

    print(f"Feature engineered dataset: {df.shape}")

    train_df, test_df = split_train_test_data(df, '2024-05-20')
    print(f"\nTrain dataset: {train_df.shape}")
    print(f"Test dataset: {test_df.shape}\n")

    # Get the features and target variable
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)

    # Model training with optuna
    print("Model training using LightGBM with hyperparameter tuning Optuna.")
    today_str = date.today().strftime("%Y-%m-%d") # Get today's date in YYYY-MM-DD format
    model_name = f"LightGBM_{today_str}"
    model = get_best_lightgbm_model(X_train, X_test, y_train, y_test, n_trials=50)

    # Export the model as a joblib file
    print(f"Exporting model\n")
    export_model(model, f"models/{model_name}.joblib")

    # Evaluate the model
    metrics_df = calculate_metrics(model, X_train, y_train, X_test, y_test, model_name=model_name)
    
    # Save model metrics
    save_metrics(metrics_df, model_name=model_name)

if __name__ == "__main__":
    main()
