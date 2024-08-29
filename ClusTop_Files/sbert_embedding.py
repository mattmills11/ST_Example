# Main libraries
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Environment Variables
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
ABSTRACTS_NAME = "Matt_clustop_abstracts.parquet"  # Fixed name for abstracts file
EMBEDDINGS_NAME = "Matt_embeddings.npy"  # Fixed name for embeddings file
MODEL_NAME = os.environ.get("MODEL_NAME", "distiluse-base-multilingual-cased-v2")

# Data Loading
file_path = os.path.join(FILESPATH, ABSTRACTS_NAME)

# Check if file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load the data
abstracts = pd.read_parquet(file_path, engine='pyarrow')
print("Original DataFrame:")
print(abstracts.head())
print("Original columns:", abstracts.columns)

# Rename the column
if len(abstracts.columns) == 1:
    abstracts.columns = ['sentences']
else:
    raise ValueError("Expected exactly one column in the DataFrame.")

# Print DataFrame columns after renaming
print("Column names after renaming:", abstracts.columns)
print("DataFrame after renaming columns:")
print(abstracts.head())

# Save the renamed DataFrame
renamed_file_path = os.path.join(FILESPATH, "Matt_renamed_abstracts.parquet")  # Updated fixed name
abstracts.to_parquet(renamed_file_path, engine='pyarrow')

# Extract the 'sentences' column to a list
abstracts_list = abstracts['sentences'].tolist()

# Generate embeddings
embedding_model = SentenceTransformer(MODEL_NAME)
embeddings = embedding_model.encode(abstracts_list, show_progress_bar=True)

# Save embeddings
embeddings_path = os.path.join(FILESPATH, EMBEDDINGS_NAME)  # Updated fixed name
np.save(embeddings_path, embeddings)

print(f"Embeddings saved to: {embeddings_path}")
print(f"Renamed abstracts saved to: {renamed_file_path}")
