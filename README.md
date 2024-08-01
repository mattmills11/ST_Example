# Text Data Analysis Pipeline

This repository contains a series of Python scripts designed to preprocess, analyze, and interpret text data using various natural language processing techniques.

## Scripts Overview

### 1. preprocessing.py
* __Purpose:__  Preprocesses text data for analysis.
* __Steps:__ 
    * Reads a source Parquet file containing text data.
    * Cleans the text data by removing special symbols, digits, and extra spaces.
    * Saves the cleaned data to a new Parquet file.

### 2. sbert_embedding.py
* __Purpose:__ Embeds sentences using SentenceTransformers.
* __Steps:__ 
    * Loads cleaned text data from the output of `preprocessing.py`.
    * Embeds each sentence using a pretrained SentenceTransformer model.
    * Saves the sentence embeddings to a file for further analysis.

### 3. dimensionality_reduction.py
* __Purpose:__  Reduces dimensionality of sentence embeddings and clusters them.
* __Steps:__
    * Loads sentence embeddings from the output of `sbert_embedding.py`.
    * Applies UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.
    * Clusters the reduced embeddings using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
    * Saves the reduced embeddings and clustering results to files.

### 4. clustop-simcse_finetuning.py
* __Purpose:__ Fine-tunes a SentenceTransformer model using ClusTop-SimCSE method.
* __Steps:__
    * Utilizes clustered data generated from `dimensionality_reduction.py`.
    * Fine-tunes a SentenceTransformer model using the ClusTop-SimCSE technique.
    * Saves the fine-tuned model for future use.

### 5.  sbert_embedding.py (again)
* __Purpose:__ Re-embeds sentences after model fine-tuning.
* __Steps:__
    * Re-loads cleaned text data from the output of `preprocessing.py`.
    * Embeds each sentence using the fine-tuned SentenceTransformer model.
    * Saves the updated sentence embeddings to a file.

### 6. dimensionality_reduction.py (again)
* __Purpose:__  Reduces dimensionality of updated sentence embeddings and re-clusters them.
* __Steps:__
    * Loads updated sentence embeddings from the output of the second `sbert_embedding.py`.
    * Re-applies UMAP for dimensionality reduction.
    * Re-clusters the reduced embeddings using HDBSCAN.
    * Saves the updated reduced embeddings and clustering results to files.

### 7. agentic_cluster_interpretation.py
* __Purpose:__ Interprets clusters using the Ollama platform for entity identification and evaluation.
* __Steps:__
    * Loads clustering results from the output of `dimensionality_reduction.py`.
    * Uses the Ollama platform to identify and evaluate entities within each cluster.
    * Saves the interpreted entities and evaluations to CSV and numpy array files.

## Execution Order
### 1. Initial Data Preparation and Embedding:
* `preprocessing.py`
* `sbert_embedding.py`

### 2. Dimensionality Reduction and Clustering:
* `dimensionality_reduction.py`

### 3. Model Fine-Tuning:
* `clustop-simcse_finetuning.py`

### 4. Re-embedding and Re-Clustering:
* `sbert_embedding.py` (again)
* `dimensionality_reduction.py` (again)

### 5. Cluster Interpretation:
* `agentic_cluster_interpretation.py`

Each script builds upon the results of the previous one, forming a comprehensive pipeline for text data analysis and interpretation.

## Ackwoledgment
This project utilizes techniques and methodologies described in the [ SentenceTransformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html) documentation.