import mlflow
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import os

# Load the wine dataset
wine = load_wine()
df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = wine.target
X = df_wine

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create directories for saving data and plots
os.makedirs('save_data', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Save the datasets
X_train.to_parquet('save_data/x_train.parquet')
pd.Series(y_train).to_csv('save_data/y_train.csv')
X_test.to_parquet('save_data/x_test.parquet')
pd.Series(y_test).to_csv('save_data/y_test.csv')

# Set up MLflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('wine-classification')

# 1. Decision Tree experiments
max_depths = [3, 4, 5, 6, 7, 8, 9, 10, None]
min_samples_splits = [2, 5, 10]

for depth in max_depths:
    for min_split in min_samples_splits:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "decision-tree",
                "Dataset": "wine",
                "Feature_Selection": "all-features"
            })
            
            # Train the model
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=42)
            dt.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, dt.predict(X_train))
            val_acc = cross_val_score(dt, X_train, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "max_depth": depth,
                "min_samples_split": min_split
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(dt, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

# 2. Random Forest experiments
n_estimators = [50, 100, 200]
max_features = [2, 4, 6, 8, 'sqrt', 'log2']

for n_est in n_estimators:
    for max_feat in max_features:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "random-forest",
                "Dataset": "wine",
                "Feature_Selection": "all-features"
            })
            
            # Train the model
            rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, random_state=42)
            rf.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, rf.predict(X_train))
            val_acc = cross_val_score(rf, X_train, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": n_est,
                "max_features": max_feat
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(rf, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

# 3. Logistic Regression experiments
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
solvers = ['lbfgs', 'liblinear', 'saga']

for c in C_values:
    for solver in solvers:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "logistic-regression",
                "Dataset": "wine",
                "Feature_Selection": "all-features"
            })
            
            # Train the model
            lr = LogisticRegression(C=c, solver=solver, max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, lr.predict(X_train))
            val_acc = cross_val_score(lr, X_train, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "C": c,
                "solver": solver
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(lr, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

# Feature selection - using only half of the features based on importance
# First, train a random forest to get feature importances
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
feature_selector.fit(X_train, y_train)
importances = feature_selector.feature_importances_

# Select top 50% features
feature_indices = np.argsort(importances)[::-1]
top_features = feature_indices[:len(feature_indices)//2]
selected_features = X_train.columns[top_features]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Save selected datasets
X_train_selected.to_parquet('save_data/x_train_selected.parquet')
X_test_selected.to_parquet('save_data/x_test_selected.parquet')

# Repeat experiments with selected features
# 1. Decision Tree with selected features
for depth in max_depths[:5]:  # Use fewer values for demonstration
    for min_split in min_samples_splits[:2]:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "decision-tree",
                "Dataset": "wine",
                "Feature_Selection": "top-50-percent"
            })
            
            # Train the model
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=42)
            dt.fit(X_train_selected, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, dt.predict(X_train_selected))
            val_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "max_depth": depth,
                "min_samples_split": min_split,
                "selected_features": list(selected_features)
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(dt, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

# 2. Random Forest with selected features
for n_est in n_estimators[:2]:
    for max_feat in max_features[:3]:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "random-forest",
                "Dataset": "wine",
                "Feature_Selection": "top-50-percent"
            })
            
            # Train the model
            rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, random_state=42)
            rf.fit(X_train_selected, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, rf.predict(X_train_selected))
            val_acc = cross_val_score(rf, X_train_selected, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": n_est,
                "max_features": max_feat,
                "selected_features": list(selected_features)
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(rf, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

# 3. Logistic Regression with selected features
for c in C_values[:2]:
    for solver in solvers[:2]:
        with mlflow.start_run():
            # Set tags
            mlflow.set_tags({
                "Model": "logistic-regression",
                "Dataset": "wine",
                "Feature_Selection": "top-50-percent"
            })
            
            # Train the model
            lr = LogisticRegression(C=c, solver=solver, max_iter=1000, random_state=42)
            lr.fit(X_train_selected, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, lr.predict(X_train_selected))
            val_acc = cross_val_score(lr, X_train_selected, y_train, cv=5).mean()
            
            # Log parameters
            mlflow.log_params({
                "C": c,
                "solver": solver,
                "selected_features": list(selected_features)
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "cv_accuracy": val_acc
            })
            
            # Log model
            mlflow.sklearn.log_model(lr, "model")
            
            # Log datasets
            mlflow.log_artifacts("save_data")

print("Experiment tracking complete! Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view results.") 