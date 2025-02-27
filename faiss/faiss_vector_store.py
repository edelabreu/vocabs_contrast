from pprint import pprint
import time
import json
import faiss
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

language= 'es'

def create_vector_store(model_name="sentence-transformers/all-mpnet-base-v2", model_size=768, add_data=False):
    # Load Model
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)

    pprint('Creating FAISS index')
    index = faiss.IndexHNSW(model_size)
    
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy= DistanceStrategy.COSINE,
        # normalize_L2=True
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
        faiss_db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        end = time.time()
        return faiss_db, start, end
    
    return vector_store, None, None

def metrics_calc(results:list, threshold:float, alpha:float):
    # # Determine relevant documents based on threshold
    # # If the result is less than or equal to the threshold, they are relevant.
    # predicted_relevant_docs = [r for r in results if 1 - r[1] <= threshold]
    # # If the result is relevant but is within the margin of error, we consider it a false positive.
    # predicted_relevant_falses = [r for r in results if 1 - r[1] <= threshold and 1 - r[1] >= threshold - alpha ]
    # # If the result is irrelevant but is within the margin of error, we consider it a false negative.
    # predicted_no_relevant = [r for r in results if 1 - r[1] > threshold and 1 - r[1] <= threshold + alpha ]
    
    # If the result is above the threshold, they are relevant.
    predicted_relevant_docs = [r for r in results if 1 - r[1] >= threshold]
    # If the result is relevant but is within the margin of error, we consider it a false positive.
    predicted_relevant_falses = [r for r in results if 1 - r[1] >= threshold and 1 - r[1] <= threshold + alpha ]
    # If the result is irrelevant but is within the margin of error, we consider it a false negative.
    predicted_no_relevant = [r for r in results if 1 - r[1] < threshold and 1 - r[1] >= threshold - alpha ]
    
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
    
    # Tasa de Verdaderos Positivos (TPR): 
    #   También conocida como recall o sensibilidad, 
    #   indica la proporción de positivos correctamente identificados por el modelo.
    TPR = recall
    # Tasa de Falsos Positivos (FPR): 
    #   Es la proporción de negativos que han sido clasificados incorrectamente como positivos.
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'TPR': TPR,
            'FPR': FPR,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'relevance': f"TP:{TP} FP:{FP} FN:{FN} TN:{TN}"
        }

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
    selection_language =st.selectbox(label="Select the language", options=['es','en' ])
    save_index_button = st.form_submit_button()

if save_index_button:
    start = time.time()
    language = selection_language
    vector_store, start, end= create_vector_store(
        model_name= selection, 
        model_size= next((s['size'] for s in MODELS if s["name"] == selection), None),
        add_data=True
    )
    end = time.time()
    st.write(f"Time for **create** vector store : {end - start:.4f} s")
    
    start = time.time()
    vector_store.save_local(folder_path="../data/faiss_db", index_name='flat_'+language)
    end = time.time()
    st.write(f"Time for **save to local** the vector store : {end - start:.4f} s")

with st.form('search_form'):
    st.subheader('Similarity Search')
    threshold_input = st.number_input("Number of **thresholds** to evaluate",min_value=1, value=100)
    threshold_aim = st.number_input("Target threshold value", value=0.4)
    top_k = st.number_input("Top-k Retrieved Documents",min_value=1, value=k)
    alpha = st.number_input("Alpha value to manage the error percentage",min_value=0.0, value=0.05)
    query = st.text_input("Query", placeholder="educación infantil")
    search_button = st.form_submit_button('Submit')

if search_button:
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)

    # TODO: Find a way to know the size and model name... also the index_name. 
    # For now they are the defaults
    vector_store, start, end= create_vector_store(model_name=model_name, model_size= model_size)
    start = time.time()
    vector_store= vector_store.load_local(
        folder_path="../data/faiss_db", 
        embeddings=embeddings_model,
        index_name='flat_'+language, 
        allow_dangerous_deserialization=True)
    end = time.time()
    st.write(f"Time for **load from local** the vector store : {end - start:.4f} s")

    start = time.time()
    # TODO: Find information about `search_k` parameter and how it affects the result
    results = vector_store.similarity_search_with_score(query=query,k=top_k)
    end = time.time()
    st.write(f"Search time: {end - start:.4f} s")
    
    # Precision, Recall, and F1-Score vs. Top-k Retrieved
    metrics = []

    # Define threshold to consider documents as relevant
    # threshold = 0.71
    # Define alpha to consider the percentage of error when retrieving relevant documents
    # alpha = 0.05
    thresholds = np.linspace(start=0, stop=1, num=threshold_input, dtype='float32')

    alphas = np.linspace(start=0, stop=0.2, num=20, dtype='float32')
    # TODO: `similarity_search_with_score` returns a list of documents most similar 
    # to the query text with L2 distance in float.
    # L2 is the default and is used to calculate the Euclidean distance, 
    # which must be normalized (vectors of the same size). 
    # It is not required to calculate the cosine distance.
    # **Lower score represents more similarity**. 
    # the prediction values ​​must be inverted since the closer they are to zero, 
    # the better the result.
    for threshold in thresholds:
        m = metrics_calc(results=results, threshold=threshold, alpha=alpha)
        metrics.append(m)
    
    d = []
    for a in alphas:
        fila = []
        fila.append(round(a,2))
        for threshold in thresholds:
            m = metrics_calc(results=results, threshold=threshold, alpha=round(a,2))
            # Saving the bests results, 
            # in this case must be over the threshold and 
            # when aubtracting precision and recall is less than 0.2
            ab = fila.append(1) if m['precision'] < 1 and m['precision'] >= threshold and m['recall'] < 1 and m['recall'] >= threshold and abs(m['precision'] - m['recall']) < 0.2 else fila.append(0)
        d.append(fila)

    df = pd.DataFrame(d)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('../data/archivo.xlsx', index=False, header=False)

    # Show Results
    df = pd.DataFrame(
        data=([metrics[i]['precision'], metrics[i]['recall'], metrics[i]['f1_score'], thresholds[i]] for i in range(len(thresholds))), 
        columns=('Precisión', 'Recall', 'F1-Score', 'Threshold'))
    
    with st.container(border=True):
        st.subheader("Precision, Recall, and F1-Score vs. Thresholds.")
        st.write("This graph is used to define the best threshold.")
        st.line_chart(data=df, x_label='Threshold index', y_label='Score')
        
        
        top_k_metric = []
        for i in range(top_k):
            top_k_metric.append(metrics_calc(results=results[:i], threshold=threshold_aim, alpha=alpha))
        
        df = pd.DataFrame(
        data=([top_k_metric[i]['precision'], top_k_metric[i]['recall'], top_k_metric[i]['f1_score']] for i in range(top_k)), 
        columns=('Precisión', 'Recall', 'F1-Score'))
        st.subheader("Precision, Recall and F1-score vs. Top-k Retrieved")
        st.write(f"Metrics for threshold: {threshold_aim} Alpha: {alpha} and Top-k: {top_k}")
        st.write("It provides a better understanding of how precision, recall and F1-score are behaving with different retrieval sizes. It can highlight the impact of increasing or decreasing k")
        st.line_chart(data=df, x_label='Top-k Retrieved Documents', y_label='Score')


        # # confusion matrix
        # Build the Confusion Matrix for a specific threshold
        index = int(threshold_aim * 100) 
        conf_matrix = [
            [metrics[index]['TP'], metrics[index]['FP']],  # [TP, FP]
            [metrics[index]['FN'], metrics[index]['TN']]   # [FN, TN]
        ]

        # Creating the visualization using seaborn
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 8}, ax=ax) #

        plt.title(f'Confusion Matriz for { top_k }', fontsize=8) #
        plt.xlabel('Predicciones', fontsize=6) #, 
        plt.ylabel('Valores Reales', fontsize=6) #, fontsize=8
        
        st.pyplot(fig)
        st.write(f"TP: {metrics[index]['TP']} FP: {metrics[index]['FP']} FN: {metrics[index]['FN']} TN: {metrics[index]['TN']}")
        plt.close()

        
    df = pd.DataFrame(
        data=([metrics[i]['precision'], metrics[i]['recall'], metrics[i]['f1_score'], thresholds[i], metrics[i]['relevance']] for i in range(len(thresholds))), 
        columns=('Precisión', 'Recall', 'F1-Score', 'Threshold', 'Relevance'))
    
    st.subheader("Table of results with Precision, Recall, and F1-Score vs. Thresholds.")
    table = st.table(df)

    df_results = pd.DataFrame(
        data=([r[1], 1 - r[1], r[0].metadata['conceptLabel'], r[0].metadata['conceptUri']] for r in results),
        columns=('Score', '1 - Score', 'conceptLabel', 'conceptUri')
    )
    st.subheader(f"Table of the {top_k} results, according to cosine similarity")
    st.table(df_results)
