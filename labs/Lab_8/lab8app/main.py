import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Define the input data structure using Pydantic
# Based on the wine dataset features from Lab 2
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float # Corrected name if needed based on notebook
    proline: float

class PredictionInput(BaseModel):
    data: List[WineFeatures]

# Configuration
MLFLOW_TRACKING_URI = 'https://mlops-mlflow-server-917889522197.us-west2.run.app'
EXPERIMENT_NAME = 'metaflow-gcp-lab7-experiment'
MODEL_NAME = 'model'

# Load the model at startup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Find the latest run in the specified experiment
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs:
        latest_run_id = runs[0].info.run_id
        model_uri = f"runs:/{latest_run_id}/{MODEL_NAME}"
        print(f"Loading model from: {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    else:
        print(f"Error: No runs found for experiment '{EXPERIMENT_NAME}'.")
        loaded_model = None
else:
    print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
    loaded_model = None


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Wine Classifier API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if loaded_model is None:
        return {"error": "Model not loaded. Check application logs."}

    try:
        # Convert Pydantic model list to DataFrame
        # Ensure column order matches the training data
        feature_columns = [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280/od315_of_diluted_wines', 'proline'
        ] # Match exact names from df_wine.columns in Lab 2
        
        # Correct potential mismatch in feature name
        # Check Lab 2 notebook for exact names if errors occur
        input_list = [item.dict(by_alias=True) for item in input_data.data]
        
        # Rename keys if necessary to match sklearn's expectations 
        # (e.g., 'od280_od315_of_diluted_wines' vs 'od280/od315_of_diluted_wines')
        corrected_input_list = []
        for item_dict in input_list:
             corrected_item = {}
             for key, value in item_dict.items():
                 # Assuming Pydantic uses underscores but model needs slash
                 corrected_key = key.replace('_od315_', '/od315_') 
                 corrected_item[corrected_key] = value
             corrected_input_list.append(corrected_item)

        df = pd.DataFrame(corrected_input_list, columns=feature_columns)
        
        # Make predictions
        predictions = loaded_model.predict(df)
        
        # Convert predictions to list for JSON response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

# To run the app: uvicorn main:app --reload --port 8000 