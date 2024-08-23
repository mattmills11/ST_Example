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
model_dir_path = os.path.join(FILESPATH, f"{current_date}_{STUDENT_MODEL}")

# Check if the model directory exists
if not os.path.isdir(model_dir_path):
    raise FileNotFoundError(f"Model directory {model_dir_path} does not exist. Please check the fine-tuning step.")

# Initialize the SentenceTransformer model from the directory
fine_tuned_model = SentenceTransformer(model_dir_path)
print(f"Model loaded from {model_dir_path}")

# Embed Sentences
embeddings = fine_tuned_model.encode(sentences, show_progress_bar=True)

# Save Embeddings
embeddings_file_path = os.path.join(FILESPATH, f"{current_date}_{EMBEDDINGS_NAME}")
np.save(embeddings_file_path, embeddings)

print(f"Embeddings saved to {embeddings_file_path}")
