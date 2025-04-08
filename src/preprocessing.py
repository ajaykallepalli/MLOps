import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def preprocess_data():
    # Create directory for processed data if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load data
    X_train = pd.read_parquet('save_data/x_train.parquet')
    X_test = pd.read_parquet('save_data/x_test.parquet')
    y_train = pd.read_csv('save_data/y_train.csv').iloc[:, 1]
    y_test = pd.read_csv('save_data/y_test.csv').iloc[:, 1]

    # Create preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, X_train.columns.tolist())
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    # Fit and transform the data
    pipeline.fit(X_train)
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Convert to dataframes
    X_train_processed_df = pd.DataFrame(
        X_train_processed, 
        columns=X_train.columns
    )
    X_test_processed_df = pd.DataFrame(
        X_test_processed, 
        columns=X_test.columns
    )

    # Add target column
    X_train_processed_df['target'] = y_train.values
    X_test_processed_df['target'] = y_test.values

    # Save processed data
    X_train_processed_df.to_csv('data/processed/processed_train_data.csv', index=False)
    X_test_processed_df.to_csv('data/processed/processed_test_data.csv', index=False)

    # Save pipeline
    with open('data/processed/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Preprocessing completed. Files saved to data/processed/ directory.")

if __name__ == "__main__":
    preprocess_data() 