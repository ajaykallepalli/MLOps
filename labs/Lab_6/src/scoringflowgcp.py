# src/scoringflowgcp.py
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    Flow,
    JSONType,
    conda_base # Added conda_base
)
import pandas as pd
import numpy as np
import os

# Define common conda environment
# Added google-cloud-storage and google-auth
# Using similar versions as training flow for consistency
SCORING_CONDA_DEP = {
    'numpy': '1.23.5',
    'pandas': '1.5.3',
    'mlflow': '2.3.2',
    'google-cloud-storage': '2.7.0',
    'google-auth': '2.16.0',
    # Add scikit-learn ONLY if the model loading/prediction itself requires it
    # Often mlflow.pyfunc handles this, but add if you encounter errors
    # 'scikit-learn': '1.2.2'
}

@conda_base(libraries=SCORING_CONDA_DEP, python='3.9.16')
class ScoringFlowGCP(FlowSpec):
    """
    A Metaflow flow for scoring new wine data from a file using a registered MLflow model.
    Includes conda_base for dependency management.
    """
    input_file_path = Parameter('input_file',
                                help='Path to the input CSV data file for scoring (local or gs://)',
                                default='labs/Lab_6/data/heldout/wine_heldout.csv', # Adjust if needed
                                type=str)

    model_stage = Parameter('stage',
                            help='Model stage to use (e.g., "Staging", "Production") or version number',
                            default='Staging',
                            type=str)

    @step
    def start(self):
        """
        Load the registered model from MLflow and prepare input data.
        """
        import mlflow
        import json

        print("Starting the scoring flow...")
        print(f"Input data file path: {self.input_file_path}")
        print(f"Model stage/version requested: {self.model_stage}")

        # MLflow Setup - use the same URI as training
        mlflow.set_tracking_uri('https://mlops-mlflow-server-917889522197.us-west2.run.app')

        # Load Registered Model - use Lab 7 model name
        model_name = "metaflow-wine-model-lab7"
        model_uri = f"models:/{model_name}/{self.model_stage}"
        print(f"Loading model from URI: {model_uri}")
        try:
            # This requires permissions to access MLflow backend (e.g., GCS)
            self.model = mlflow.pyfunc.load_model(model_uri)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{model_uri}': {e}")
            try:
                print("Attempting to load latest version...")
                model_uri_latest = f"models:/{model_name}/latest"
                self.model = mlflow.pyfunc.load_model(model_uri_latest)
                print(f"Loaded latest model version instead from: {model_uri_latest}")
            except Exception as e_latest:
                 print(f"Failed to load latest model version too: {e_latest}")
                 raise

        # Prepare Input Data from File
        # Handles local or GCS paths
        try:
            if self.input_file_path.startswith('gs://'):
                print(f"Reading input data from GCS path: {self.input_file_path}...")
                # Requires google-cloud-storage installed
                self.input_df = pd.read_csv(self.input_file_path)
            elif os.path.exists(self.input_file_path):
                 print(f"Reading input data from local path: {self.input_file_path}...")
                 self.input_df = pd.read_csv(self.input_file_path)
            else:
                 print(f"Error: Input file not found at {self.input_file_path}")
                 raise FileNotFoundError(f"Input file not found at {self.input_file_path}")

            if 'target' in self.input_df.columns:
                self.input_df = self.input_df.drop(columns=['target'])

            print("Input data prepared (first 5 rows):")
            print(self.input_df.head())
        except Exception as e:
             print(f"Error preparing input data: {e}")
             raise

        self.next(self.predict)

    @step
    def predict(self):
        """
        Make predictions using the loaded model.
        """
        print("Making prediction...")
        try:
            self.predictions = self.model.predict(self.input_df)
            print(f"Raw Predictions (first 5): {self.predictions[:5]}...")

            wine_class_names = ['class_0', 'class_1', 'class_2']
            self.predicted_classes = [wine_class_names[p] for p in self.predictions]

        except Exception as e:
            print(f"Error during prediction: {e}")
            self.predictions = None
            # Provide an error list matching input length
            self.predicted_classes = ["Prediction Error"] * len(self.input_df)
            # Consider not raising if you want the flow to finish and report partial results/errors
            # raise

        self.next(self.end)

    @step
    def end(self):
        """
        Output the predictions.
        """
        print("Scoring flow finished.")
        if self.predictions is not None:
            print("--- Prediction Results (First 5 Rows) ---")
            results_df = self.input_df.head().copy() # Show results for head only
            results_df['raw_prediction'] = self.predictions[:5]
            results_df['predicted_class'] = self.predicted_classes[:5]
            print(results_df)

            # Example: Save results to GCS (ensure bucket exists and permissions are set)
            # output_path = 'gs://storage-mlopsmsds13-metaflow-default/results/scored_data.csv' # Use your bucket
            # print(f"Saving full results would go to a path like: {output_path}")
            # Example save command (uncomment and adjust path if needed):
            # full_results_df = self.input_df.copy()
            # full_results_df['raw_prediction'] = self.predictions
            # full_results_df['predicted_class'] = self.predicted_classes
            # full_results_df.to_csv(output_path, index=False)

        else:
            print("Prediction failed or skipped due to errors.")


if __name__ == '__main__':
    ScoringFlowGCP()