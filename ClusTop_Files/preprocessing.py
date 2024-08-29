# Main libraries
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

"""
Environment Variables
"""
# Update the file path to the correct location of your files
FILESPATH = "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/"  # Adjust this to your environment
SOURCE_FILE = "requisicionClean.parquet"  # Correctly referencing the source file
ABSTRACTS_NAME = "Matt_clustop_abstracts.parquet"  # Output file name

# Load stop words for Spanish
stop_words = set(stopwords.words('spanish'))

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    - Replace special symbols with a space or another replacement text
    - Remove digits
    - Replace multiple spaces with a single space
    - Remove leading and trailing spaces
    - Remove stop words (Spanish)

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

    # Remove stop words (Spanish)
    words = text.split()
    cleaned_text = ' '.join([word for word in words if word.lower() not in stop_words])
    
    return cleaned_text

# Correct the file path to the source data file
source_path = os.path.join(FILESPATH, SOURCE_FILE)

# Check if the file exists before loading it
if not os.path.exists(source_path):
    raise FileNotFoundError(f"The file {source_path} does not exist.")

# Load data
df_requisiciones_clean = pd.read_parquet(source_path)

# Clean each text column, including stop word removal
for col in df_requisiciones_clean.select_dtypes(include='object'):  # This selects only text columns
    df_requisiciones_clean[col] = df_requisiciones_clean[col].astype(str).apply(clean_text)

# Convert the cleaned sentences to a list
abstracts = df_requisiciones_clean["sentences"].to_list()
print("Number of sentences:", len(abstracts))

# Convert the list to a pandas DataFrame
df = pd.DataFrame(abstracts, columns=["sentences"])

# Save the DataFrame as a Parquet file
output_path = os.path.join(FILESPATH, ABSTRACTS_NAME)
df.to_parquet(output_path, engine='pyarrow')

print(f"Data saved to: {output_path}")
