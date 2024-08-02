"""
sbert_embedding.py

This script re-embeds sentences after model fine-tuning using the SentenceTransformer model.

Libraries
---------
- os: Provides a way of using operating system dependent functionality.
- datetime: Supplies classes for manipulating dates and times.
- numpy: A package for scientific computing with Python.
- pandas: A data manipulation and analysis library.
- sentence_transformers: A library for computing sentence embeddings using pretrained models and fine-tuning them.

Environment Variables
----------------------
- FILESPATH: Path to the directory containing the data files. Default is set via environment variable.
- ABSTRACTS_NAME: Name of the parquet file containing the cleaned text data. Default is "abstracts.parquet".
- EMBEDDINGS_NAME: Name of the numpy file to save the updated sentence embeddings. Default is "updated_embeddings.npy".
- STUDENT_MODEL: Name of the fine-tuned SentenceTransformer model to be used. Default is "make-multilingual-simcse-class".
"""

# Main Libraries
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Environment Variables
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
ABSTRACTS_NAME = os.environ.get("ABSTRACTS_NAME", "abstracts.parquet")
EMBEDDINGS_NAME = os.environ.get("EMBEDDINGS_NAME", "updated_embeddings.npy")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "make-multilingual-simcse-class")

# Load Cleaned Text Data
current_date = datetime.now().strftime("%Y%m%d")
abstracts_file_path = os.path.join(FILESPATH, f"{current_date}_Matt_{ABSTRACTS_NAME}")

# Check if file exists
if not os.path.isfile(abstracts_file_path):
    raise FileNotFoundError(f"The file {abstracts_file_path} does not exist.")

# Load the data
abstracts = pd.read_parquet(abstracts_file_path, engine='pyarrow')
print("Loaded DataFrame:")
print(abstracts.head())
print("Columns in the DataFrame:", abstracts.columns)

# Ensure the column name is 'sentences'
if len(abstracts.columns) == 1:
    abstracts.columns = ['sentences']
else:
    raise ValueError("Expected exactly one column in the DataFrame.")

# Extract the 'sentences' column to a list
sentences = abstracts['sentences'].tolist()

# Load Fine-tuned Model
model_path = os.path.join(FILESPATH, STUDENT_MODEL)
fine_tuned_model = SentenceTransformer(model_path)

# Embed Sentences
embeddings = fine_tuned_model.encode(sentences, show_progress_bar=True)

# Save Embeddings
embeddings_file_path = os.path.join(FILESPATH, f"{current_date}_{EMBEDDINGS_NAME}")
np.save(embeddings_file_path, embeddings)

print(f"Embeddings saved to {embeddings_file_path}")
