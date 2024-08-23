"""
Preprocessing Script

This script is designed to preprocess text data for analysis. The script reads a source Parquet file, cleans the text data, and saves the cleaned data to a new Parquet file.

Libraries
----------
- os: Provides a way of using operating system dependent functionality.
- re: Provides regular expression matching operations.
- datetime: Supplies classes for manipulating dates and times.
- pandas: A data manipulation and analysis library.

Environment Variables
----------------------
- FILESPATH: Path to the directory containing the source file. Default is set via environment variable.
- SOURCE_FILE: Name of the source file to be processed. Default is "pruebas".
- FILE_NAME: Name of the output file for the cleaned data. Default is "abstracts.parquet".
"""

# Main libraries
import os
import re
from datetime import datetime

import pandas as pd

"""
Environment Variables
"""
FILESPATH = "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/" # /home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/
SOURCE_FILE = os.environ.get("SOURCE_FILE", "requisicionClean.parquet") # "requisicionClean.parquet"
ABSTRACTS_NAME = os.environ.get("FILE_NAME", "abstracts.parquet") # 'abstracts.parquet'

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    - Replace special symbols with a space or another replacement text
    - Remove digits
    - Replace multiple spaces with a single space
    - Remove leading and trailing spaces

    Parameters
    ----------
    text : str
        The text to be cleaned.

    Returns
    -------
    str
        The cleaned text.
    """
    
    # Replace special symbols with a space or another replacement text
    text = re.sub(r"[^\w\s]", '', text) 
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text

# File path to the source data file
source_path = (
    f"{FILESPATH}{SOURCE_FILE}"
)

# Load data
df_requisiciones_clean = pd.read_parquet(source_path)

# Clean each text column
for col in df_requisiciones_clean.select_dtypes(include='object'): # This selects only text columns
    df_requisiciones_clean[col] = df_requisiciones_clean[col].astype(str).apply(clean_text)

abstracts=df_requisiciones_clean["sentences"].to_list()
print("num sentences:", len(abstracts))

# Save abstracts
current_date = datetime.now().strftime("%Y%m%d")

# Convert the list to a pandas DataFrame
df = pd.DataFrame(abstracts)

# Save the DataFrame as a Parquet file
df.to_parquet(f"{FILESPATH}{current_date}_Matt_{ABSTRACTS_NAME}", engine='pyarrow')
print(f"{FILESPATH}{current_date}_{ABSTRACTS_NAME}")
