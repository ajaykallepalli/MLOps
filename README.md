# MLOps Course Repository

This repository contains the labs and projects for the MLOps course. The repository is structured to support machine learning development workflows, including data management, model training, and experiment tracking.

## Project Structure

- `data/`: Directory for storing datasets
- `notebooks/`: Jupyter notebooks for development and analysis
- `models/`: Directory for storing trained models
- `mlops/`: Virtual environment for the project
- `save_data/`: Directory for storing training and test datasets
- `images/`: Directory for storing visualizations

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv mlops
source mlops/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## MLflow Experiment Tracking

This project includes scripts for tracking machine learning experiments using MLflow:

1. `mlflow_tracking.py`: Runs multiple experiments with different algorithms (Decision Trees, Random Forests, and Logistic Regression) with various hyperparameters. Also implements feature selection and tracks all experiments in MLflow.

2. `register_best_model.py`: Identifies the top 3 models from the experiments, evaluates the best model on the test set, and registers it in the MLflow model registry.

### Running the Experiments

1. First, run the experiment tracking script:
   ```bash
   python mlflow_tracking.py
   ```

2. View the results in the MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

3. After reviewing the experiments, register the best model:
   ```bash
   python register_best_model.py
   ```

## Dependencies

- mlflow==2.15.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1

## Project Details

This implementation demonstrates several key MLOps concepts:
- Experiment tracking with MLflow
- Hyperparameter tuning for multiple algorithms
- Feature selection
- Model evaluation and comparison
- Model registration and staging 