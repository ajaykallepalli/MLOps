import requests
import json

# URL of the running FastAPI application
url = "http://127.0.0.1:8000/predict" 

# Sample input data (using the first row from the wine dataset as an example)
# Ensure feature names match the Pydantic model in main.py (using underscores)
sample_data = {
    "data": [
        {
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127.0,
            "total_phenols": 2.80,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280_od315_of_diluted_wines": 3.92, # Use underscore here
            "proline": 1065.0
        }
        # Add more samples if needed
    ]
}

# Send POST request
try:
    response = requests.post(url, json=sample_data)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    # Print the response
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"Error sending request: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response:")
    print(response.text) 