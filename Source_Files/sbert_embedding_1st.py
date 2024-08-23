# Main libraries
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Environments
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
ABSTRACTS_NAME = os.environ.get("FILE_NAME", "abstracts.parquet")
EMBEDDINGS_NAME = os.environ.get("EMBEDDINGS_NAME", "embeddings.npy")
MODEL_NAME = os.environ.get("MODEL_NAME", "distiluse-base-multilingual-cased-v2")

# Data Loading
current_date = datetime.now().strftime("%Y%m%d")
file_path = os.path.join(FILESPATH, f"{current_date}_Matt_{ABSTRACTS_NAME}")

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
renamed_file_path = os.path.join(FILESPATH, f"{current_date}_Matt_renamed_{ABSTRACTS_NAME}")
abstracts.to_parquet(renamed_file_path, engine='pyarrow')

# Extract the 'sentences' column to a list
abstracts_list = abstracts['sentences'].tolist()

# Generate embeddings
model = MODEL_NAME
embedding_model = SentenceTransformer(model)
embeddings = embedding_model.encode(abstracts_list, show_progress_bar=True)

# Save embeddingsnp.save(os.path.join(FILESPATH, f"{current_date}_Matt_{EMBEDDINGS_NAME}"), embeddings)
# Save embeddings
np.save(os.path.join(FILESPATH, f"{current_date}_Matt_{EMBEDDINGS_NAME}"), embeddings)
