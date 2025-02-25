"""
Annoy is a vector database that uses **cocene distance** as a metric,
This means that:
    - Cosine similarity measures the orientation (or direction) of two vectors. 
      It indicates how similar two vectors are in terms of their angle between them, 
      consider their magnitude. Cosine similarity values are between -1 and 1.
      The score is:
        1 if the vectors are identical in terms of direction (the angle between them is 0°),
        0 if they are perpendicular (the angle between them is 90°), and
        -1 if they are opposite (the angle between them is 180°).

    - Cosine distance is simply the difference between 1 and cosine similarity. 
      It is used to measure the difference between vectors rather than their similarity, 
      and its value is in the range of 0 to 2
      The score is:
        0 means the vectors are identical (cosine similarity is 1).
        1 means the vectors are perpendicular (cosine similarity is 0).
        2 means the vectors are completely opposite (cosine similarity is -1).

Can also be configured with other metrics
    ["angular", "euclidean", "manhattan", "hamming", "dot"]

Official documentation
    https://github.com/spotify/annoy

    Cosine distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))
"""

import json
import time
import numpy as np
import streamlit as st
import pandas as pd

from pprint import pprint
from annoy import AnnoyIndex
from langchain.schema import Document
from langchain_community.vectorstores import Annoy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

language='es'

def create_vector_store(model_name="sentence-transformers/all-mpnet-base-v2", model_size=768, add_data=False):
    # Load Model
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)
    
    # Create index Annoy
    pprint('Creating Annoy index')
    metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
    index = AnnoyIndex(
        f=model_size, # len(embed_query(documents[0].page_content)) 
        metric=metric) 
    vector_store = Annoy(
        embedding_function= embeddings_model,
        index=index,
        metric=metric,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    if add_data:
        documents= []

        pprint('Load unesco-dataset')
        with open('../data/unesco-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
            unesco_dataset= json.load(f)

        pprint('Convert to LangChain documents')
        for d in unesco_dataset:
            documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
        
        pprint('Load agrovoc-dataset')
        with open('../data/agrovoc-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
            agrovoc_dataset= json.load(f)

        pprint('Convert to LangChain documents')
        for d in agrovoc_dataset:
            documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
        
        pprint('Load eurovoc-dataset')
        with open('../data/eurovoc-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
            eurovoc_dataset= json.load(f)

        pprint('Convert to LangChain documents')
        for d in eurovoc_dataset:
            documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))

        pprint('Indexing documents to the vector store')
        start = time.time()
        annoy_db = vector_store.from_documents(
            documents=documents, 
            embedding=embeddings_model,
            n_trees=100, 
            n_jobs=1)
        end = time.time()
        return annoy_db, start, end
    return vector_store, None, None

# model                                                 dimensional dense vector
# sentence-transformers/all-mpnet-bembeddingsase-v2         768
# sentence-transformers/all-MiniLM-L12-v2                   384
# sentence-transformers/all-MiniLM-L6-v2                    384 (have the best results in the example)

MODELS =[
    {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'size':384        
    },
    {
        'name': 'sentence-transformers/all-MiniLM-L12-v2',
        'size':384        
    },
    {
        'name': 'sentence-transformers/all-mpnet-bembeddingsase-v2',
        'size':768        
    }
]

model_name= MODELS[0]['name'] # 'sentence-transformers/all-MiniLM-L6-v2'
model_size= MODELS[0]['size']
k= 10

with st.form('save_index_form'):
    st.subheader("Create index from scratch")
    selection =st.selectbox(label="Select the model", 
                    options=[i['name'] for i in MODELS ]
                    )
    save_index_button = st.form_submit_button()

if save_index_button:
    start = time.time()
    vector_store, start, end= create_vector_store(
        model_name= selection, 
        model_size= next((s['size'] for s in MODELS if s["name"] == selection), None),
        add_data=True
    )
    end = time.time()
    st.write(f"Time for **create** vector store : {end - start:.4f} s")
    
    start = time.time()
    vector_store.save_local(folder_path="../data/annoy_db")
    end = time.time()
    st.write(f"Time for **save to local** the vector store : {end - start:.4f} s")

with st.form('search_form'):
    st.subheader('Similarity Search')
    top_k = st.number_input("Top-k Retrieved Documents",min_value=1, value=k)
    alpha = st.number_input("Alpha value to manage the error percentage",min_value=0.0, value=0.05)
    query = st.text_input("Query", placeholder="educación infantil")
    search_button = st.form_submit_button('Submit')

if search_button:
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)

    vector_store, start, end= create_vector_store(model_name=model_name, model_size= model_size)
    start = time.time()
    vector_store= vector_store.load_local(folder_path="../data/annoy_db", embeddings=embeddings_model, allow_dangerous_deserialization=True)
    end = time.time()
    st.write(f"Time for **load from local** the vector store : {end - start:.4f} s")

    start = time.time()
    # TODO: Find information about `search_k` parameter and how it affects the result
    results = vector_store.similarity_search_with_score(query=query,k=top_k)
    end = time.time()
    st.write(f"Search time: {end - start:.4f} s")
    
    # results.sort( key=lambda x: x[1], reverse=True)
    
    # 4. Precision, Recall, and F1-Score vs. Top-k Retrieved

    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    f1_scores = []


    # Define threshold to consider documents as relevant
    # threshold = 0.71
    # Define alpha to consider the percentage of error when retrieving relevant documents
    # alpha = 0.05
    thresholds = np.linspace(start=0, stop=1, num=100, dtype='float32')

    # TODO: Since Annoy uses cosine distance, 
    # the prediction values ​​must be inverted since the closer they are to zero, 
    # the better the result.
    for threshold in thresholds:
        # Determine relevant documents based on threshold
        # If the result is above the threshold, they are relevant.
        predicted_relevant_docs = [r for r in results if r[1] >= threshold]
        # If the result is relevant but is within the margin of error, we consider it a false positive.
        predicted_relevant_falses = [r for r in results if r[1] >= threshold and r[1] <= threshold + alpha ]
        # If the result is irrelevant but is within the margin of error, we consider it a false negative.
        predicted_no_relevant = [r for r in results if r[1] < threshold and r[1] >= threshold - alpha ]
        
        # Construction of the confusion matrix
        TP = len(predicted_relevant_docs) - len(predicted_relevant_falses)
        FP = len(predicted_relevant_falses)
        FN = len(predicted_no_relevant)
        TN = len(results) - (TP + FP + FN)

        # Calculate evaluation metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1_score)
        
    # Show Results
    df = pd.DataFrame(([precision_scores[i], recall_scores[i], f1_scores[i], thresholds[i]] for i in range(len(thresholds))), columns=('Precisión', 'Recall', 'F1-Score', 'Threshold'))
    
    with st.container(border=True):
        st.subheader("Precision, Recall, and F1-Score vs. Thresholds.")
        st.write("This graph is used to define the best threshold.")
        st.line_chart(data=df, x_label='Thresholds', y_label='Score')
    
    st.subheader("Table of results with Precision, Recall, and F1-Score vs. Thresholds.")
    table = st.table(df)

    df_results = pd.DataFrame(
        data=([r[1], r[0].metadata['conceptLabel'], r[0].metadata['conceptUri']] for r in results),
        columns=('Score', 'conceptLabel', 'conceptUri')
    )
    st.subheader(f"Table of the {top_k} results, according to cosine similarity")
    st.table(df_results)
    