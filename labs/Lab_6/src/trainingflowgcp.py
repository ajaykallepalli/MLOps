# src/trainingflowgcp.py
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    conda_base,
    resources,
    retry,
    timeout,
    catch,
    kubernetes # Added kubernetes
)
import os

# Define common conda environment for most steps
# Added google-cloud-storage and google-auth for GCP interaction
# Added specific versions as shown in Lab 7 examples
COMMON_CONDA_DEP = {
    'numpy': '1.23.5',
    'scikit-learn': '1.2.2',
    'pandas': '1.5.3', # Example version
    'mlflow': '2.3.2', # Example version, ensure compatibility with your MLflow server
    'google-cloud-storage': '2.7.0', # Example version
    'google-auth': '2.16.0', # Example version
    'databricks-cli': '0.17.7' # Added this line (example version)
}

@conda_base(libraries=COMMON_CONDA_DEP, python='3.9.16')
class TrainingFlowGCP(FlowSpec):
    """
    A Metaflow flow for training wine classification models,
    adapted for execution on GCP Kubernetes with error handling
    and dependency management.
    """

    random_seed = Parameter('seed',
                            help='Random seed for train/test split',
                            default=42,
                            type=int)

    @step
    def start(self):
        """
        Load data and split into train/test sets.
        """
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import pandas as pd

        print("Starting the training flow...")

        wine = datasets.load_wine()
        # Added pylint disable comments to the following 3 lines:
        feature_names = wine.feature_names # pylint: disable=no-member
        X = pd.DataFrame(wine.data, columns=feature_names) # pylint: disable=no-member
        y = wine.target # pylint: disable=no-member

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        print(f"Data loaded and split. Train set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")

        self.next(self.train_dt, self.train_rf)

    # Added decorators as per Lab 7 instructions
    @catch(print_exception=True)
    @retry(times=2)
    @timeout(minutes=15)
    # Corrected memory value to MB (2GB = 2048MB)
    @resources(cpu='1', memory='2048')
    @kubernetes # Run on Kubernetes
    @step
    def train_dt(self):
        """
        Train a Decision Tree classifier on Kubernetes.
        """
        from sklearn.tree import DecisionTreeClassifier

        print("Training Decision Tree on K8s...")
        max_depth = 5

        self.model_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_seed)
        self.model_dt.fit(self.X_train, self.y_train)

        self.model_type = "DecisionTree"
        self.next(self.choose_model)

    # Added decorators as per Lab 7 instructions
    @catch(print_exception=True)
    @retry(times=2)
    @timeout(minutes=15)
    # Corrected memory value to MB (2GB = 2048MB)
    @resources(cpu='1', memory='2048')
    @kubernetes # Run on Kubernetes
    @step
    def train_rf(self):
        """
        Train a Random Forest classifier on Kubernetes.
        """
        from sklearn.ensemble import RandomForestClassifier

        print("Training Random Forest on K8s...")
        n_estimators = 100
        max_features = 4

        self.model_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            oob_score=True,
            random_state=self.random_seed
        )
        self.model_rf.fit(self.X_train, self.y_train)

        self.model_type = "RandomForest"
        self.next(self.choose_model)

    # Added decorators as per Lab 7 instructions
    @catch(print_exception=True)
    @retry(times=3)
    @timeout(minutes=10)
    @resources(cpu='1', memory='1024')
    @kubernetes
    @step
    def choose_model(self, inputs):
        """
        Choose the best model based on test accuracy and register with MLflow.
        Runs on Kubernetes.
        """
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os

        print("Choosing the best model on K8s...")

        mlflow.set_tracking_uri('https://mlops-mlflow-server-917889522197.us-west2.run.app')
        mlflow.set_experiment('metaflow-gcp-lab7-experiment')

        results = []
        for inp in inputs:
            model = inp.model_dt if inp.model_type == "DecisionTree" else inp.model_rf
            preds = model.predict(inp.X_test)
            score = accuracy_score(inp.y_test, preds)
            results.append({'model_type': inp.model_type, 'model': model, 'score': score, 'input_obj': inp})
            print(f"Model: {inp.model_type}, Test Accuracy: {score}")

        best_result = max(results, key=lambda x: x['score'])
        self.best_model = best_result['model']
        self.best_model_score = best_result['score']
        self.best_model_type = best_result['model_type']
        print(f"Selected best model: {self.best_model_type} with score {self.best_model_score}")

        best_input = best_result['input_obj']
        self.X_train = best_input.X_train
        self.y_train = best_input.y_train
        self.X_test = best_input.X_test
        self.y_test = best_input.y_test

    # --- Log and Register with MLflow ---
        registered_model_name = "metaflow-wine-model-lab7" # New name for Lab 7

        with mlflow.start_run(run_name=f"{self.best_model_type}_GCP_Training_Run") as run:
            print(f"MLflow Run ID: {run.info.run_uuid}")
            mlflow.log_param("model_type", self.best_model_type)
            mlflow.log_params(self.best_model.get_params())
            mlflow.log_metric("test_accuracy", self.best_model_score)

            model_info = mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
            print(f"Model logged: {model_info.model_uri}")

            # --- MODIFIED PART ---
            # Comment out or remove the lines that cause the AttributeError
            # print(f"Model registered: Name={registered_model_name}, Version={model_info.version}") # Causes error

            # Store only the name for the end step
            self.registered_model_name = registered_model_name
            # self.registered_model_version = model_info.version # Causes error
            # --- END OF MODIFIED PART ---

        self.next(self.end)

    @step
    def end(self):
        """
        Final step of the flow.
        """
        print("Training flow finished.")
        # Accessing artifacts requires them to be explicitly set on 'self' in the previous step
        # --- MODIFIED PART ---
        # Adjusted print statement as we no longer store the specific version number
        if hasattr(self, 'best_model_type') and hasattr(self, 'best_model_score') and hasattr(self, 'registered_model_name'):
            print(f"Best model selected: {self.best_model_type}")
            print(f"Best model score: {self.best_model_score}")
            print(f"Best model registered in MLflow: Name={self.registered_model_name} (a version was created/updated)")
        # --- END OF MODIFIED PART ---
        else:
            print("Could not access all results from previous step(s).")

if __name__ == '__main__':
    TrainingFlowGCP()