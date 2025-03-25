import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

# Set tracking URI and experiment
mlflow.set_tracking_uri('sqlite:///mlflow.db')
experiment_name = 'wine-classification'
experiment = mlflow.get_experiment_by_name(experiment_name)

# Initialize MLflow client
client = MlflowClient()

# Get all runs for the experiment
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Sort runs by validation accuracy
sorted_runs = runs.sort_values(by=['metrics.cv_accuracy'], ascending=False)

# Get the top 3 runs
top_3_runs = sorted_runs.head(3)
print("Top 3 models based on cross-validation accuracy:")
for i, (index, run) in enumerate(top_3_runs.iterrows()):
    print(f"{i+1}. Run ID: {run.run_id}, Model: {run.tags.Model}, CV Accuracy: {run.metrics.cv_accuracy:.4f}")

# Load the best model
best_run_id = top_3_runs.iloc[0].run_id
best_model_uri = f"runs:/{best_run_id}/model"
best_model = mlflow.sklearn.load_model(best_model_uri)

# Evaluate on test set
test_data = pd.read_parquet('save_data/x_test.parquet')
test_labels = pd.read_csv('save_data/y_test.csv').iloc[:, 1].values

# If the best model used selected features, we need to use those features for prediction
if 'Feature_Selection' in top_3_runs.iloc[0].tags and top_3_runs.iloc[0].tags.Feature_Selection == 'top-50-percent':
    test_data = pd.read_parquet('save_data/x_test_selected.parquet')

# Evaluate on test set
test_predictions = best_model.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Log the test set evaluation as a new run
with mlflow.start_run():
    mlflow.set_tags({
        "Model": top_3_runs.iloc[0].tags.Model,
        "Dataset": "wine",
        "Stage": "testing",
        "Parent_Run": best_run_id
    })
    
    # Log parameters from the best model
    for param_name in top_3_runs.iloc[0].params.keys():
        mlflow.log_param(param_name, top_3_runs.iloc[0].params[param_name])
    
    # Log test accuracy
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_model")
    
    # Register the model
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/best_model",
        name="wine_classifier"
    )
    
    print(f"\nBest model evaluated on test set. Test accuracy: {test_accuracy:.4f}")
    print(f"Model registered as 'wine_classifier' version {registered_model.version}")
    
    # Move the model to staging
    client.transition_model_version_stage(
        name="wine_classifier",
        version=registered_model.version,
        stage="Staging"
    )
    
    print(f"Model version {registered_model.version} transitioned to 'Staging' stage") 