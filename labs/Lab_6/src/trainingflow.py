# src/trainingflow.py
from metaflow import FlowSpec, step, Parameter
import os

# Define the flow class
class TrainingFlow(FlowSpec):
    """
    A Metaflow flow for training wine classification models.
    """

    # Define parameters (Example: random seed for reproducibility)
    random_seed = Parameter('seed',
                            help='Random seed for train/test split',
                            default=42,
                            type=int)

    # --- Start Step ---
    @step
    def start(self):
        """
        Load data and split into train/test sets.
        """
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import pandas as pd # Import pandas here or globally

        print("Starting the training flow...")

        # Load wine dataset
        wine = datasets.load_wine() # Load the Bunch object
        feature_names = wine.feature_names # pylint: disable=no-member
        X = pd.DataFrame(wine.data, columns=feature_names) # pylint: disable=no-member
        y = wine.target # pylint: disable=no-member

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        print(f"Data loaded and split. Train set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")

        # Define the next steps (branching to train different models)
        self.next(self.train_dt, self.train_rf)

    # --- Placeholder for Model Training Steps ---
    # We will add these next

    # --- Placeholder for Model Choosing Step ---
    # We will add this later

    # --- End Step ---
    @step
    def end(self):
        """
        Final step of the flow.
        """
        print("Training flow finished.")
        # In a real scenario, you might save final artifacts or summaries here.
        # For this lab, the model selection and registration happen before 'end'.
        print(f"Best model selected: {self.best_model_type}")
        print(f"Best model score: {self.best_model_score}")
        print(f"Best model registered in MLflow: {self.registered_model_name} version {self.registered_model_version}")

    # --- Train Decision Tree Step ---
    @step
    def train_dt(self):
        """
        Train a Decision Tree classifier.
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score

        print("Training Decision Tree...")
        # Example hyperparameters (use values from your previous best DT run if known)
        max_depth = 5 # Example value

        self.model_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_seed)
        self.model_dt.fit(self.X_train, self.y_train)

        # Score the model (optional here, can be done in choose_model step)
        # preds_dt = self.model_dt.predict(self.X_test)
        # self.score_dt = accuracy_score(self.y_test, preds_dt)
        # print(f"DT Test Accuracy: {self.score_dt}")

        # Artifacts specific to this branch are automatically saved by Metaflow
        self.model_type = "DecisionTree" # Store model type for merging step

        # Define the next step (joining the branches)
        self.next(self.choose_model)

    # --- Train Random Forest Step ---
    @step
    def train_rf(self):
        """
        Train a Random Forest classifier.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        print("Training Random Forest...")
        # Example hyperparameters (use values from your previous best RF run if known)
        n_estimators = 100  # Example value
        max_features = 4    # Example value

        self.model_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            oob_score=True, #
            random_state=self.random_seed
        )
        self.model_rf.fit(self.X_train, self.y_train)

        # Score the model (optional here, can be done in choose_model step)
        # preds_rf = self.model_rf.predict(self.X_test)
        # self.score_rf = accuracy_score(self.y_test, preds_rf)
        # print(f"RF Test Accuracy: {self.score_rf}")
        # print(f"RF OOB Score: {self.model_rf.oob_score_}") #

        # Artifacts specific to this branch
        self.model_type = "RandomForest" # Store model type for merging step

        # Define the next step (joining the branches)
        self.next(self.choose_model)
    
    # --- Choose Best Model and Register Step ---
    @step
    def choose_model(self, inputs):
        """
        Choose the best model based on test accuracy and register with MLflow.
        """
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import accuracy_score
        import pandas as pd # Added import
        import os # Added import

        print("Choosing the best model...")

        # --- MLflow Setup ---
        # Use the URI from Lab 6 example or your local setup
        mlflow.set_tracking_uri('https://mlops-mlflow-server-917889522197.us-west2.run.app') # Example from Lab 6
        # Or use your local sqlite DB: mlflow.set_tracking_uri('sqlite:///mlflow.db')

        mlflow.set_experiment('metaflow-experiment') # Example from Lab 6

        # --- Evaluate Models ---
        results = []
        for inp in inputs: # inputs contains results from branches (dt, rf)
            model = inp.model_dt if inp.model_type == "DecisionTree" else inp.model_rf
            preds = model.predict(inp.X_test) # Access test data passed down
            score = accuracy_score(inp.y_test, preds) # Access test labels passed down
            results.append({'model_type': inp.model_type, 'model': model, 'score': score})
            print(f"Model: {inp.model_type}, Test Accuracy: {score}")

        # --- Select Best Model ---
        best_result = max(results, key=lambda x: x['score'])
        self.best_model = best_result['model']
        self.best_model_score = best_result['score']
        self.best_model_type = best_result['model_type']
        print(f"Selected best model: {self.best_model_type} with score {self.best_model_score}")

        # --- Merge Artifacts ---
        # Metaflow requires explicitly merging artifacts after a branch.
        # We need X_train, y_train, X_test, y_test for the 'end' step if needed,
        # and the best_model details.
        # Let's merge from the input that corresponds to the best model.
        best_input = next(inp for inp in inputs if inp.model_type == self.best_model_type)
        self.X_train = best_input.X_train
        self.y_train = best_input.y_train
        self.X_test = best_input.X_test
        self.y_test = best_input.y_test
        # self.best_model, self.best_model_score, self.best_model_type are already set on self

        # --- Log and Register with MLflow ---
        registered_model_name = "metaflow-wine-model-lab6" # Give it a lab-specific name

        # Start an MLflow run within the Metaflow step
        with mlflow.start_run(run_name=f"{self.best_model_type}_Training_Run") as run:
            print(f"MLflow Run ID: {run.info.run_uuid}")
            mlflow.log_param("model_type", self.best_model_type)
            mlflow.log_params(self.best_model.get_params()) # Log model hyperparameters
            mlflow.log_metric("test_accuracy", self.best_model_score)

            # Log the model and register it
            # The artifact_path is where the model files are stored within the MLflow run
            model_info = mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model", # Standard path within MLflow run
                registered_model_name=registered_model_name
            )
            print(f"Model logged: {model_info.model_uri}")
            print(f"Model registered: Name={registered_model_name}, Version={model_info.registered_model_version}")

            # Store registration details for the end step
            self.registered_model_name = registered_model_name
            self.registered_model_version = model_info.registered_model_version

            # Log datasets as artifacts (optional, but good practice)
            # Create temporary files to log dataframes
            train_data_path = "train_data.csv"
            test_data_path = "test_data.csv"
            pd.concat([self.X_train, pd.Series(self.y_train, name='target')], axis=1).to_csv(train_data_path, index=False)
            pd.concat([self.X_test, pd.Series(self.y_test, name='target')], axis=1).to_csv(test_data_path, index=False)
            mlflow.log_artifact(train_data_path, artifact_path="data")
            mlflow.log_artifact(test_data_path, artifact_path="data")
            # Clean up temp files
            os.remove(train_data_path)
            os.remove(test_data_path)

        self.next(self.end) # Proceed to the end step

# --- Run the Flow ---
if __name__ == '__main__':
    TrainingFlow()