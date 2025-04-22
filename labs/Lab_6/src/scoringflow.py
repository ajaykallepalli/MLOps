# src/scoringflow.py
from metaflow import FlowSpec, step, Parameter, Flow, JSONType
import pandas as pd
import numpy as np # Import numpy
import os # Import os

class ScoringFlow(FlowSpec):
    """
    A Metaflow flow for scoring new wine data from a file using a registered MLflow model.
    """
    # Parameter to accept the path to the input data file
    input_file_path = Parameter('input_file', # Changed name to use underscore
                                help='Path to the input CSV data file for scoring',
                                default='labs/Lab_6/data/heldout/wine_heldout.csv', # Update default path
                                type=str)

    # Parameter to specify which registered model to use (e.g., 'production', 'staging', or version)
    model_stage = Parameter('stage',
                            help='Model stage to use (e.g., "Staging", "Production") or version number',
                            default='Staging', # Default to staging as per lab instructions
                            type=str)

    @step
    def start(self):
        """
        Load the registered model from MLflow and prepare input data.
        """
        import mlflow
        import json # Import json

        print("Starting the scoring flow...")
        print(f"Input data file path: {self.input_file_path}")
        print(f"Model stage/version requested: {self.model_stage}")

        # --- MLflow Setup --- (Use the same URI as in training)
        mlflow.set_tracking_uri('https://mlops-mlflow-server-917889522197.us-west2.run.app') # Example from Lab 6
        # Or use your local sqlite DB: mlflow.set_tracking_uri('sqlite:///mlflow.db')

        # --- Load Registered Model ---
        model_name = "metaflow-wine-model-lab6" # Must match the name used during registration
        model_uri = f"models:/{model_name}/{self.model_stage}"
        print(f"Loading model from URI: {model_uri}")
        try:
            self.model = mlflow.pyfunc.load_model(model_uri) # Load as pyfunc for generic predict
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Optionally, try loading the latest version if stage fails
            try:
                print("Attempting to load latest version...")
                model_uri_latest = f"models:/{model_name}/latest"
                self.model = mlflow.pyfunc.load_model(model_uri_latest)
                print(f"Loaded latest model version instead from: {model_uri_latest}")
            except Exception as e_latest:
                 print(f"Failed to load latest model version too: {e_latest}")
                 raise # Re-raise the exception if loading fails


        # --- Prepare Input Data from File ---
        try:
            if not os.path.exists(self.input_file_path):
                 print(f"Error: Input file not found at {self.input_file_path}")
                 raise FileNotFoundError(f"Input file not found at {self.input_file_path}")

            print(f"Reading input data from {self.input_file_path}...")
            # Read the CSV directly into input_df
            self.input_df = pd.read_csv(self.input_file_path)

            # Drop target column if it exists - model only needs features
            if 'target' in self.input_df.columns:
                self.input_df = self.input_df.drop(columns=['target'])

            print("Input data prepared:")
            print(self.input_df)
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
            print(f"Raw Predictions: {self.predictions}")

             # Map numeric predictions back to wine class names if desired
            wine_class_names = ['class_0', 'class_1', 'class_2'] # From sklearn.datasets.load_wine().target_names
            self.predicted_classes = [wine_class_names[p] for p in self.predictions]

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Potentially log the error or handle it
            self.predictions = None # Indicate prediction failure
            self.predicted_classes = ["Error"]
            raise # Re-raise the exception


        self.next(self.end)

    @step
    def end(self):
        """
        Output the predictions.
        """
        print("Scoring flow finished.")
        if self.predictions is not None:
            print("--- Prediction Results ---")
            # Print results for each input row (here just one)
            for i, pred_class in enumerate(self.predicted_classes):
                 # Convert NumPy array element to native Python type if needed
                 raw_pred = self.predictions[i]
                 if isinstance(raw_pred, np.generic):
                      raw_pred = raw_pred.item() # Convert numpy type to Python int/float
                 print(f"Input Row {i+1}: Predicted Class = {pred_class} (Raw Prediction: {raw_pred})")

        else:
            print("Prediction failed.")

# --- Run the Flow ---
if __name__ == '__main__':
    ScoringFlow()