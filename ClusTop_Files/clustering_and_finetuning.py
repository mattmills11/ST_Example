# Import Libraries and Set Environment Variables

# Imports and Setup
import os
import numpy as np
import pandas as pd
from torch import cuda, nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, models, losses
from umap import UMAP
from hdbscan import HDBSCAN

# Environment Variables
FILESPATH = "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/"
EMBEDDINGS_NAME = "Matt_embeddings.npy"
CLUSTERS_DATAFRAME_NAME = "Matt_df_cluster.csv"  # Fixed name for clustered data CSV
TEACHER_MODEL = "/home/tulipan16372/storage_NAS/Misc/Dani_Amaya/sentence-transformers/make-multilingual-en-es-2020-10-31_19-04-26"  # The pretrained model you've been using
STUDENT_MODEL = "make-multilingual-simcse-class"  # Name to save the fine-tuned student model

# Load Embeddings
# Debug print to ensure environment variables are set correctly
print(f"FILESPATH: {FILESPATH}")
print(f"CLUSTERS_DATAFRAME_NAME: {CLUSTERS_DATAFRAME_NAME}")

# Load embeddings
embeddings_path = os.path.join(FILESPATH, EMBEDDINGS_NAME)
embeddings = np.load(embeddings_path, allow_pickle=True)
print(f"Embeddings loaded from: {embeddings_path}")

# UMAP Projections
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', init='random', n_jobs=-1)
reduced_embeddings = umap_model.fit_transform(embeddings)
print(f"UMAP projection completed. Reduced embeddings shape: {reduced_embeddings.shape}")

# HDBSCAN Clustering
hdbscan_model = HDBSCAN(min_cluster_size=300, min_samples=50, metric='euclidean', cluster_selection_method='eom')
hdbscan_model.fit(reduced_embeddings)

# Save Clusters
df_cluster = pd.DataFrame(reduced_embeddings, columns=["umap_x", "umap_y"])
df_cluster['cluster'] = hdbscan_model.labels_
clusters_dataframe_path = os.path.join(FILESPATH, CLUSTERS_DATAFRAME_NAME)
df_cluster.to_csv(clusters_dataframe_path, index=False)
print(f"Clustered data saved to: {clusters_dataframe_path}")

# Fine-tuning and Saving Model

# Load Data
df_cluster = pd.read_csv(os.path.join(FILESPATH, CLUSTERS_DATAFRAME_NAME))

# Extract labels and ensure valid clusters
y = df_cluster.loc[df_cluster['cluster'] != -1, "cluster"]
labels = np.array(y)
num_labels = len(set(labels))

# Use index as text (since the 'documents' column is not present)
texts = df_cluster.index.tolist()

# Load the teacher model and prepare for ClusTop fine-tuning
word_embedding_model = models.Transformer(TEACHER_MODEL, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=num_labels,
    activation_function=nn.Tanh(),
)

# Initialize SentenceTransformer model with the components
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Prepare training examples
train_examples = [
    InputExample(texts=[str(text), str(text)], label=int(label))
    for text, label in zip(texts, labels) if label != -1
]

# DataLoader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16,
    collate_fn=model.smart_batching_collate
)

# Loss Function
train_loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=num_labels
)

# Fine-tune the model
print("Fine-tuning the model...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=15,
    warmup_steps=100,
    show_progress_bar=True,
    use_amp=True  # Mixed precision for faster training on GPU
)

# Save the fine-tuned model
student_model_save_path = os.path.join(FILESPATH, "Matt_ClusTop_fine_tuned_model")
model.save(student_model_save_path)
print(f"Fine-tuned model saved to: {student_model_save_path}")
