"""
clustop-simcse_finetuning.py

This script fine-tunes a SentenceTransformer model using the ClusTop-SimCSE method on clustered data.

Libraries
---------
- os: Provides a way of using operating system dependent functionality.
- ast: Safely evaluates strings containing Python expressions.
- datetime: Supplies classes for manipulating dates and times.
- numpy: A package for scientific computing with Python.
- pandas: A data manipulation and analysis library.
- torch: A deep learning framework.
- sentence_transformers: A library for computing sentence embeddings using pretrained models and fine-tuning them.

Environment Variables
----------------------
- FILESPATH: Path to the directory containing the data files. Default is set via environment variable.
- CLUSTERS_DATAFRAME_NAME: Name of the CSV file containing the clustered data. Default is "df_cluster.csv".
- TEACHER_MODEL: Name of the pretrained SentenceTransformer model to be used as the teacher model. Default is "sentence-transformers/distiluse-base-multilingual-cased-v2".
- STUDENT_MODEL: Name to save the fine-tuned student model. Default is "make-multilingual-simcse-class".
"""

# Main Libraries
import os
import ast
from datetime import datetime
import numpy as np
import pandas as pd
from torch import cuda, nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, models, losses

# Environment Variables
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
CLUSTERS_DATAFRAME_NAME = os.environ.get("CLUSTERS_DATAFRAME_NAME", "df_cluster.csv")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "sentence-transformers/distiluse-base-multilingual-cased-v2")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "make-multilingual-simcse-class")

# Load Data
def convert_to_list(val):
    """
    Converts a string representation of a list back to an actual list.

    Parameters
    ----------
    val : str
        The string representation of a list.

    Returns
    -------
    list
        The evaluated list.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

# Read the CSV file
current_date = datetime.now().strftime("%Y%m%d")
df_cluster = pd.read_csv(f"{FILESPATH}{current_date}_Matt_{CLUSTERS_DATAFRAME_NAME}")

# Prepare data for training
df_filtered = df_cluster[df_cluster['cluster'] != -1]
texts = df_filtered['documents'].tolist()
labels = df_filtered['cluster'].tolist()

# Debugging: Print the first few texts and their types to check
for i, text in enumerate(texts[:10]):
    print(f"text[{i}]: {text}, type: {type(text)}")

# Ensure texts and labels are in the correct format
cleaned_texts = []
cleaned_labels = []

for text, label in zip(texts, labels):
    if isinstance(text, str):
        cleaned_texts.append(text)
        cleaned_labels.append(label)
    else:
        try:
            cleaned_texts.append(str(text))
            cleaned_labels.append(label)
        except Exception as e:
            print(f"Skipping invalid entry: text={text}, label={label}, error={e}")

texts = cleaned_texts
labels = cleaned_labels

# Debugging: Print the cleaned texts and labels
print("Cleaned Sample texts:", texts[:5])
print("Cleaned Sample labels:", labels[:5])

num_labels = len(set(labels))
print("Number of labels:", num_labels)

# Initialize SentenceTransformer models
word_embedding_model = models.Transformer(TEACHER_MODEL, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=num_labels,
    activation_function=nn.Tanh()
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Prepare training examples
train_examples = [InputExample(texts=[text, text], label=label) for text, label in zip(texts, labels)]

# Debugging: Print the number of valid training examples
print("Number of valid training examples:", len(train_examples))

# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, collate_fn=model.smart_batching_collate)

# Loss Function
train_loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=num_labels
)

# Fine-tune model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=15, warmup_steps=100)

# Save model
model.save(f"{FILESPATH}{current_date}_{STUDENT_MODEL}")
