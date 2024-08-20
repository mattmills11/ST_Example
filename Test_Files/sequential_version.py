# Main Libraries
import os
from datetime import datetime
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import time

# Environment Variables
FILESPATH = os.environ.get("FILESPATH", "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/")
ABSTRACTS_NAME = os.environ.get("FILE_NAME", "abstracts.parquet")
EMBEDDINGS_NAME = os.environ.get("EMBEDDINGS_NAME", "Matt_embeddings.npy")
REDUCED_EMBEDDINGS_NAME = os.environ.get("REDUCED_EMBEDDINGS_NAME", "reduces_embeddings.npy")
CLUSTERS_DATAFRAME_NAME = os.environ.get("CLUSTERS_DATAFRAME_NAME", "df_cluster.csv")

# Load Data
current_date = datetime.now().strftime("%Y%m%d")
renamed_file_path = os.path.join(FILESPATH, f"{current_date}_Matt_renamed_{ABSTRACTS_NAME}")
embeddings_path = os.path.join(FILESPATH, f"{current_date}_{EMBEDDINGS_NAME}")
embeddings = np.load(embeddings_path, allow_pickle=True)

print(f"Embeddings shape: {embeddings.shape}")

# Add noise to the embeddings to aid spectral initialization
noise = np.random.normal(loc=0, scale=0.01, size=embeddings.shape)
noisy_embeddings = embeddings + noise

# Measure execution time
start_time = time.time()

# UMAP Projection
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', init='random', n_jobs=1)
reduced_embeddings = umap_model.fit_transform(noisy_embeddings)

# Save reduced embeddings
np.save(f"{FILESPATH}{current_date}_Matt_{REDUCED_EMBEDDINGS_NAME}", reduced_embeddings)

# Label the clustering
df_cluster = pd.DataFrame(reduced_embeddings, columns=["umap_x", "umap_y"])

# HDBSCAN Clustering
hdbscan_model = HDBSCAN(min_cluster_size=300, min_samples=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
hdbscan_model.fit(df_cluster[['umap_x', 'umap_y']])
print('clusterer labels shape:', hdbscan_model.labels_.shape)

# Label the clusters on df_cluster
df_cluster['cluster'] = hdbscan_model.labels_
print('num clusters:', df_cluster['cluster'].nunique())

# Add documents to dataframe with clustering
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

# Save df_cluster
df_cluster.to_csv(f"{FILESPATH}{current_date}_Matt_{CLUSTERS_DATAFRAME_NAME}", index=False)

end_time = time.time()
print(f"Sequential version execution time: {end_time - start_time} seconds")
