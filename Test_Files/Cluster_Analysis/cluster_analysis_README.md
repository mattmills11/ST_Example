# Cluster Analysis Pipeline

This repository contains a series of Jupyter notebooks designed to analyze clusters of text data using various techniques such as summarization, topic modeling, named entity recognition, and word cloud visualizations.

## Getting Started

### Setting Up Your Environment

1. **Install Dependencies:** 
    Ensure you have the necessary Python packages installed. You can install the required packages by running the following command:
    
    ```bash
    pip install -r requirements.txt
    ```

2. **Open the Notebooks:**
    After installing the dependencies, you can open any of the notebooks in Jupyter Notebook or Visual Studio Code with Jupyter extensions enabled.

---

## Notebooks Overview

### 1. Summarize_Documents_in_Each_Cluster.ipynb
* **Purpose:** Summarizes the documents in each cluster.
* **Steps:**
    * **Targeting a Specific Cluster:** When you identify a cluster number from another analysis (e.g., by hovering over points on a dynamic graph), you can use this notebook to generate a summary for that specific cluster.
    * **Instructions:** 
        1. Open the `Summarize_Documents_in_Each_Cluster.ipynb` notebook.
        2. Set the `cluster_number` variable to the cluster you want to analyze (e.g., `cluster_number = 5`).
        3. Run the cells to load the data, process the documents, and generate the summary for the specified cluster.

    ```python
    # Example: Summarizing cluster number 5
    cluster_number = 5  # Set this to the cluster you want to analyze
    ```

    * The output will display a summary of the documents in that specific cluster.

---

### 2. BERTopic_Method.ipynb
* **Purpose:** Extracts topics from documents in each cluster using BERTopic.
* **Steps:**
    * **Targeting a Specific Cluster:** Use this notebook to analyze topics in a particular cluster, for example, cluster number 5.
    * **Instructions:**
        1. Open the `BERTopic_Method.ipynb` notebook.
        2. Set the `cluster_number` variable to the cluster you want to analyze.
        3. Run the notebook to apply BERTopic and generate topics for that cluster.

    ```python
    # Example: Analyzing topics for cluster number 5
    cluster_number = 5  # Set this to the cluster you want to analyze
    ```

    * The output will display the top topics for the selected cluster and their probabilities.

---

### 3. Named_Entity_Recognition.ipynb
* **Purpose:** Identifies named entities in the documents of each cluster.
* **Steps:**
    * **Targeting a Specific Cluster:** Use this notebook to extract named entities from a specific cluster, for example, cluster number 5.
    * **Instructions:**
        1. Open the `Named_Entity_Recognition.ipynb` notebook.
        2. Set the `cluster_number` variable to the cluster you want to analyze.
        3. Run the notebook to extract and display named entities such as people, organizations, and locations from that cluster.

    ```python
    # Example: Extracting named entities for cluster number 5
    cluster_number = 5  # Set this to the cluster you want to analyze
    ```

    * The output will display the most frequent entities for the selected cluster.

---

### 4. Word_Cloud_Visualization.ipynb
* **Purpose:** Visualizes the most frequent words in each cluster using word clouds.
* **Steps:**
    * **Targeting a Specific Cluster:** Use this notebook to generate a word cloud for a particular cluster, for example, cluster number 5.
    * **Instructions:**
        1. Open the `Word_Cloud_Visualization.ipynb` notebook.
        2. Set the `cluster_number` variable to the cluster you want to analyze.
        3. Run the notebook to generate and display a word cloud for that cluster.

    ```python
    # Example: Generating a word cloud for cluster number 5
    cluster_number = 5  # Set this to the cluster you want to analyze
    ```

    * The output will display a word cloud that highlights the most frequent words in the selected cluster.

---

### 5. Cluster_Summary_Based_on_Keywords.ipynb
* **Purpose:** Extracts and displays the top keywords from each cluster using TF-IDF.
* **Steps:**
    * **Targeting a Specific Cluster:** Use this notebook to extract the most important keywords from a specific cluster, for example, cluster number 5.
    * **Instructions:**
        1. Open the `Cluster_Summary_Based_on_Keywords.ipynb` notebook.
        2. Set the `cluster_number` variable to the cluster you want to analyze.
        3. Run the notebook to display the top 10 keywords for that cluster.

    ```python
    # Example: Extracting top keywords for cluster number 5
    cluster_number = 5  # Set this to the cluster you want to analyze
    ```

    * The output will display the top 10 keywords for the selected cluster.

---

## Example Use Case: Hovering Over Points in a Dynamic Graph

If you are using dynamic graphs to visualize the embeddings of your clusters and you want to analyze a particular cluster (e.g., cluster number 5):

1. **Identify the Cluster Number:** Hover over a point on the graph to find the cluster number.
2. **Choose the Notebook:** Decide which analysis method you want to use (e.g., summarization, topic modeling, NER, etc.).
3. **Target the Cluster:** Open the corresponding notebook, set the `cluster_number` variable to the cluster number you want to analyze (e.g., `cluster_number = 5`), and run the notebook.
4. **Analyze the Results:** The notebook will output a summary, topics, named entities, word cloud, or keywords for the specified cluster.

This pipeline allows you to flexibly analyze different clusters in-depth using multiple methods. Simply adjust the `cluster_number` variable to target specific clusters as needed.
