# MLOps Course Repository

This repository contains the labs and projects for the MLOps course. The repository is structured to support machine learning development workflows, including data management, model training, and experiment tracking.

## Project Structure

- `data/`: Directory for storing datasets
- `notebooks/`: Jupyter notebooks for development and analysis
- `models/`: Directory for storing trained models
- `mlops/`: Virtual environment for the project

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

## Dependencies

- mlflow==2.15.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1 