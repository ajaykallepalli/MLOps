# labs/Lab_6/scripts/create_heldout.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# --- Configuration ---
HELDOUT_SIZE = 0.20 # 20% for held-out set
RANDOM_STATE = 123 # Use a different seed than the training flow

# Define output path relative to the MLOps project root
OUTPUT_DIR = "labs/Lab_6/data/heldout"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wine_heldout.csv")

# --- Load Original Data ---
print("Loading original wine dataset...")
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')
print(f"Original dataset shape: {X.shape}")

# --- Split Data ---
# We only need the held-out part here. The rest is implicitly for training/validation.
print(f"Splitting data ({1-HELDOUT_SIZE:.0%} train/validation, {HELDOUT_SIZE:.0%} held-out)...")
_, X_heldout, _, y_heldout = train_test_split(
    X, y, test_size=HELDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Held-out set shape: {X_heldout.shape}")

# Combine features and target for the held-out set
heldout_df = pd.concat([X_heldout, y_heldout], axis=1)

# --- Save Held-out Data ---
print(f"Saving held-out data to {OUTPUT_FILE}...")
# Create directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
heldout_df.to_csv(OUTPUT_FILE, index=False)

print("Held-out dataset created successfully.") 