from loguru import logger
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

class CustomMLModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow PyFunc model wrapper for Random Forest Regression."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])
        
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values
        
        return self.model.predict(processed_input)


def model_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    n_trials: int = 25,
    mlflow_uri: str = "http://localhost:5000"
) -> tuple[RandomForestRegressor, str]:
    """
    ML training pipeline with MLflow tracking:
    - Hyperparameter tuning with Optuna
    - Model training and logging
    
    Returns:
        tuple: (trained_model, run_id)
    """
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("NYC Citi Bike Demand Forecasting")
    
    # Start MLflow run
    run = mlflow.start_run()
    run_id = run.info.run_id
    
    try:
        logger.info(f"Starting MLflow tracked training and evaluation with run_id: {run_id}")
        
        # Hyperparameter optimization
        def objective(trial):
            params = {
                "n_estimators": 300,
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": -1,
                "random_state": 42,
            }
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, y_pred))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Best hyperparameters: {study.best_params}")

        # Train final model with best parameters
        best_model = RandomForestRegressor(
            **study.best_params,
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        )
        best_model.fit(X_train, y_train)
        
        # Log hyperparameters only
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("n_trials", n_trials)
        
        # Save and log model artifacts
        Path("mlflow/artifacts").mkdir(parents=True, exist_ok=True)
        
        model_path = "mlflow/artifacts/model.pkl"
        joblib.dump(best_model, model_path)
        
        feature_names_path = "mlflow/artifacts/feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")

        # Log the model with MLflow
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CustomMLModel(),
            artifacts={
                "model": model_path,
                "feature_names": feature_names_path
            }
        )
        
        logger.info(f"Model training completed and logged to MLflow run: {run_id}")
        
        return best_model, run_id
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        mlflow.end_run(status="FAILED")
        raise