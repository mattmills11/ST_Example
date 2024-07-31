"""
agentic_cluster_interpretation.py

This script performs interpretation of clusters using the Ollama platform for entity identification and evaluation.


Ollama Installation
-------------------
To use Ollama, follow these installation steps:

```
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3:70b
```

-------------------

Main Libraries
-------------------
- requests: For making HTTP requests to the Ollama API.
- json: For handling JSON data.
- os: Provides a way of using operating system dependent functionality.
- re: Provides support for regular expressions.
- ast: Safely evaluates strings containing Python expressions.
- unicodedata: Provides access to Unicode character database.
- datetime: Supplies classes for manipulating dates and times.
- numpy: A package for scientific computing with Python.
- pandas: A data manipulation and analysis library.
- IPython.display: Provides functions for displaying objects in an IPython environment.

Environment Variables
----------------------
- FILESPATH: Path to the directory containing the data files. Default is set via environment variable.
- CLUSTERS_DATAFRAME_NAME: Name of the CSV file containing the clustered data. Default is "df_cluster.csv".
- ENTITIES_NAME: Name of the output file to save entity-agent results. Default is "multilingual_cluster_agente_entidades_Ollama3.npy".
- EVALUATORS_NAME: Name of the output file to save evaluator-agent results. Default is "multilingual_cluster_agente_evaluador_Ollama3.npy".

Functions
-------------
- convert_to_list(val): Converts a string representation of a list back to an actual list.
- consulta_llama(documentos): Queries the Ollama platform to identify key entities in a cluster of operational and logistical activities.
- eliminar_tildes(texto): Removes accents from a given text.
- llama_text_to_list(texto_entidades_clave): Extracts entities listed in the response from Ollama.
- agente_evaluador_llama(documentos): Queries the Ollama platform to evaluate entity lists derived from summaries.

Workflow
------------
1. Load Data:
   - Reads the clustered data from a CSV file, converting specified columns to lists.

2. Entity Identification Agent:
   - Iterates through each cluster, querying Ollama to identify key entities in operational and logistical activities.
   - Saves the results in a numpy array and a CSV file.

3. Output Interpretation:
   - Processes the Ollama responses to standardize entity lists, handling special cases with few documents.
   - Saves the interpreted entity lists in a CSV file.

4. Evaluator Agent:
   - Iterates through each cluster, querying Ollama to evaluate entity lists derived from summaries.
   - Saves the evaluator-agent results in a numpy array.

Note: Ensure that the Ollama platform is running locally for the scripts to successfully query the API.

"""

# Main Libraries
import requests
import json

import os
import re
import ast
import unicodedata

from datetime import datetime

import numpy as np
import pandas as pd

# Environment Variables
FILESPATH = os.environ.get("FILESPATH") 
CLUSTERS_DATAFRAME_NAME = os.environ.get("CLUSTERS_DATAFRAME_NAME", "df_cluster.csv")
ENTITIES_NAME = os.environ.get("ENTITIES_NAME","multilingual_cluster_agente_entidades_Ollama3") # "multilingual_cluster_entidades_Ollama3.npy"
EVALUATORS_NAME = os.environ.get("EVALUATORS_NAME", "multilingual_cluster_agente_evaluador_Ollama3") # "multilingual_cluster_agente_evaluador_Ollama3"

# Load Data
# Define a converter function to evaluate the string representation of a list
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

# Specify the columns that should be converted using the converter function
converters = {
    'list_column': convert_to_list  # Replace 'list_column' with the name of your list column
}

# Read the CSV file, using the converters parameter
current_date = datetime.now().strftime("%Y%m%d")
df_cluster = pd.read_csv(f"{FILESPATH}{current_date}_{CLUSTERS_DATAFRAME_NAME}", converters=converters)

# Drop duplicates inside each cluster
df_cluster = df_cluster.groupby(["cluster", "documents"]).agg({"umap_x":"first", "umap_y":"first"}).reset_index()


# Entity identification Agent
def consulta_llama(documentos):
    """
    Queries the Ollama platform to identify key entities in a cluster of operational and logistical activities.

    Parameters
    ----------
    documentos : list
        List of documents (texts) belonging to a cluster.

    Returns
    -------
    str
        Identified key entities from Ollama's response.
    """
 
    url = "http://localhost:11434/api/generate"

    headers = {
        'Content-Type':'application/json',
    }

    data = {
        "model":"llama3:70b",
        "stream":False,
        "temperature":0,
        "prompt":f"""
            Eres un experto en actividades operacionales y logisticas.
            \nEres conciso y escribes lo mas relevante e importante en español.
            \nNo das explicaciones.
            \nEnumera las entidades clave que aparecen en este grupo de actividades operacionales: {documentos}.
            \nEscribe aca las entidades:
            """
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_text=response.text
        data=json.loads(response_text)
        actual_response = data["response"]
        print(actual_response)
        return actual_response
    
    else:
        print("Error:", response.status_code, response.text)

num_cluster = df_cluster["cluster"].max()
lista_entidades_llama=[]

for cluster in range(0, num_cluster+1):
    print(cluster, "from", num_cluster)
    
    documentos = df_cluster.loc[df_cluster["cluster"]==cluster, "documents"].tolist()
    lista_entidades_llama += [(cluster, consulta_llama(documentos))]

# Save entity-agent output
# Convert the list to a numpy array with dtype=object
array_entidades_llama = np.array(lista_entidades_llama, dtype=object)

# Save the numpy array to a .npy file
np.save(f"{FILESPATH}{current_date}_{ENTITIES_NAME}.npy", array_entidades_llama)

# Output Interpretation
def eliminar_tildes(texto):
    """
    Removes accents from a given text.

    Parameters
    ----------
    texto : str
        Text containing accented characters.

    Returns
    -------
    str
        Text with accents removed.
    """
    # Normaliza el texto para separar los caracteres base de los acentos
    texto_normalizado = unicodedata.normalize('NFD', texto)
    # Filtra los caracteres que no son de acentuación
    texto_sin_tildes = ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')
    # Normaliza nuevamente para formar los caracteres compuestos
    return unicodedata.normalize('NFC', texto_sin_tildes)

def llama_text_to_list(texto_entidades_clave):
    """
    Extracts entities listed in the response from Ollama.

    Parameters
    ----------
    texto_entidades_clave : str
        Text containing key entities identified by Ollama.

    Returns
    -------
    list
        List of extracted entities.
    """
    text = texto_entidades_clave
    
    # Replace accents
    text = eliminar_tildes(text)

    # Use a regular expression to find all listed items
    pattern = r'\d+\.\s+\*?\*?([A-Za-z\s]+)\*?\*?'
    matches = re.findall(pattern, text)

    # Strip any leading/trailing whitespace from the matched items
    entities = [match.strip() for match in matches]
    entities = [x.upper() for x in entities]
    entities = list(set(entities))
    entities = [x for x in entities if len(x)>2]

    #print(entities)
    
    return entities

df_entidades_llama = pd.DataFrame(array_entidades_llama, columns=["cluster", "llm_summary"])
df_entidades_llama["documents"] = df_entidades_llama["cluster"].map(df_cluster.loc[df_cluster["cluster"]!=-1].groupby("cluster").agg({"documents":list}).to_dict()['documents'])
df_entidades_llama["entities"]=df_entidades_llama["llm_summary"].apply(llama_text_to_list)

# Special cases - Few documents
df_entidades_llama_few_docs = df_entidades_llama.loc[df_entidades_llama["documents"].apply(lambda x: len(x)<4)]
list_of_lists = df_entidades_llama_few_docs["documents"].apply(lambda x: [y.split() for y in x])
def flatten_sublist(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
flattened_list = [flatten_sublist(x) for x in list_of_lists]
df_entidades_llama_few_docs["entities_2"] = flattened_list
df_entidades_llama_few_docs["entities_2"] = df_entidades_llama_few_docs["entities_2"].apply(lambda x: [y for y in list(set(x)) if len(y)>2])

df_entidades_llama.loc[df_entidades_llama["documents"].apply(lambda x: len(x)<4),"entities"] = df_entidades_llama_few_docs["entities_2"]

# Save df_entidades_llama
df_entidades_llama.to_csv(f"{FILESPATH}{current_date}_{ENTITIES_NAME}.csv", index=False)

# Evaluator Agent
def agente_evaluador_llama(documentos):
    """
    Queries the Ollama platform to evaluate entity lists derived from summaries.

    Parameters
    ----------
    documentos : list
        List of summaries to evaluate.

    Returns
    -------
    str
        Evaluated entity lists from Ollama's response.
    """
    
    url = "http://localhost:11434/api/generate"

    headers = {
        'Content-Type':'application/json',
    }

    data = {
        "model":"llama3:70b", #"kikes"
        "stream":False,
        "temperature":0,
        "prompt":f"""
            \nTarea: Leer el resumen de otro agente y listar las entidades encontradas por:
            **persona**,
            **equipo**,
            **empresa**,
            **lugar**,
            **material o producto**,
            **actividad o accion**,
            **otro**
            \nEres un experto evaluador con conocimiento en actividades operacionales y logisticas.
            \nEres conciso y escribes lo mas relevante e importante en español.
            \nNo das explicaciones.
            \nIncluye las entidades entre paréntesis.
            \nLee este resumen: {documentos}.
            \nEscribe aca las entidades:
            """
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_text=response.text
        data=json.loads(response_text)
        actual_response = data["response"]
        print(actual_response)
        return actual_response
    
    else:
        print("Error:", response.status_code, response.text)
    
# Run All
from IPython.display import clear_output
import time

num_cluster = df_cluster["cluster"].max()
lista_evaluator_llama=[]

for cluster in range(0, num_cluster+1):
    if df_entidades_llama["documents"].apply(lambda x: len(x)>=4):
        clear_output(wait=True)
        print(cluster, "from", num_cluster)
        time.sleep(0.1)
        
        documentos = df_entidades_llama.loc[df_entidades_llama["cluster"]==cluster, "llm_summary"].tolist()
        lista_evaluator_llama += [(cluster, agente_evaluador_llama(documentos))]

# Save evaluator-agent output
# Convert the list to a numpy array with dtype=object
array_evaluator_llama = np.array(lista_evaluator_llama, dtype=object)

# Save the numpy array to a .npy file
np.save(f"{FILESPATH}{current_date}_{EVALUATORS_NAME}.npy", array_evaluator_llama)