# Main Libraries
import os
from datetime import datetime
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN

# Environment Variables
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
UPDATED_EMBEDDINGS_NAME = os.environ.get("UPDATED_EMBEDDINGS_NAME", "updated_embeddings.npy")
REDUCED_EMBEDDINGS_NAME = os.environ.get("REDUCED_EMBEDDINGS_NAME", "updated_reduced_embeddings.npy")
UPDATED_CLUSTERS_DATAFRAME_NAME = os.environ.get("UPDATED_CLUSTERS_DATAFRAME_NAME", "updated_df_cluster.csv")
ABSTRACTS_NAME = os.environ.get("ABSTRACTS_NAME", "abstracts.parquet")

# Load Embeddings Data
current_date = datetime.now().strftime("%Y%m%d")
embeddings_path = os.path.join(FILESPATH, f"{current_date}_{UPDATED_EMBEDDINGS_NAME}")

# Check if the embeddings file exists
if not os.path.isfile(embeddings_path):
    raise FileNotFoundError(f"Embeddings file {embeddings_path} does not exist.")
embeddings = np.load(embeddings_path, allow_pickle=True)

print(f"Embeddings shape: {embeddings.shape}")

# Add noise to the embeddings to aid spectral initialization
noise = np.random.normal(loc=0, scale=0.01, size=embeddings.shape)
noisy_embeddings = embeddings + noise

# UMAP Projection
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', init='random', n_jobs=-1)
reduced_embeddings = umap_model.fit_transform(noisy_embeddings)

# Save reduced embeddings
reduced_embeddings_path = os.path.join(FILESPATH, f"{current_date}_{REDUCED_EMBEDDINGS_NAME}")
np.save(reduced_embeddings_path, reduced_embeddings)

# Label the clustering
df_cluster = pd.DataFrame(reduced_embeddings, columns=["umap_x", "umap_y"])

# HDBSCAN Clustering
hdbscan_model = HDBSCAN(min_cluster_size=300, min_samples=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
hdbscan_model.fit(df_cluster[['umap_x', 'umap_y']])
print('Clusterer labels shape:', hdbscan_model.labels_.shape)

# Label the clusters on df_cluster
df_cluster['cluster'] = hdbscan_model.labels_
print('Number of clusters:', df_cluster['cluster'].nunique())

# Load the original sentences to add to the DataFrame
renamed_file_path = os.path.join(FILESPATH, f"{current_date}_Matt_renamed_{ABSTRACTS_NAME}")

# Check if the renamed abstracts file exists
if not os.path.isfile(renamed_file_path):
    raise FileNotFoundError(f"Renamed abstracts file {renamed_file_path} does not exist.")
df_abstracts = pd.read_parquet(renamed_file_path, engine='pyarrow')

print("Column names in the loaded DataFrame:", df_abstracts.columns)

# Ensure the column name is 'sentences'
if 'sentences' in df_abstracts.columns:
    abstracts = df_abstracts['sentences'].tolist()
else:
    print("DataFrame contents:")
    print(df_abstracts.head())
    raise KeyError("The column 'sentences' does not exist in the DataFrame")

df_cluster["documents"] = abstracts[:len(df_cluster)]

# Save the updated clustering results
updated_clusters_path = os.path.join(FILESPATH, f"{current_date}_{UPDATED_CLUSTERS_DATAFRAME_NAME}")
df_cluster.to_csv(updated_clusters_path, index=False)

print(f"Reduced embeddings saved to {reduced_embeddings_path}")
print(f"Updated clustering results saved to {updated_clusters_path}")
