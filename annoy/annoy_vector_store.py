import json
import time
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from annoy import AnnoyIndex
from langchain.schema import Document
from langchain_community.vectorstores import Annoy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

language='es'

def create_vector_store(model_name="sentence-transformers/all-mpnet-base-v2"):
    # Load Model
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)

    documents= []

    pprint('Load unesco-dataset')
    with open('../data/unesco-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
        unesco_dataset= json.load(f)

    # Convert to LangChain documents
    pprint('Convert to LangChain documents')
    for d in unesco_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
    
    pprint('Load agrovoc-dataset')
    with open('../data/agrovoc-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
        agrovoc_dataset= json.load(f)

    # Convert to LangChain documents
    pprint('Convert to LangChain documents')
    for d in agrovoc_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
    
    pprint('Load eurovoc-dataset')
    with open('../data/eurovoc-dataset-'+language+'.json', 'r', encoding='utf-8') as f:
        eurovoc_dataset= json.load(f)

    # Convert to LangChain documents
    pprint('Convert to LangChain documents')
    for d in eurovoc_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
    
    # Create index Annoy
    pprint('Creating Annoy index')
    metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
    index = AnnoyIndex(
        f=len(embeddings_model.embed_query(documents[0].page_content)), 
        metric=metric) 
    vector_store = Annoy(
        embedding_function= embeddings_model,
        index=index,
        metric=metric,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    pprint('Indexing documents to the vector store')
    start = time.time()
    annoy_db = vector_store.from_documents(
        documents=documents, 
        embedding=embeddings_model,
        n_trees=100, 
        n_jobs=1)
    end = time.time()
    return annoy_db, start, end


model_name= 'sentence-transformers/all-MiniLM-L6-v2'

k= 10
query= "educación infantil"
vector_store, start, end= create_vector_store(model_name=model_name)
pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
# TODO: Find information about `search_k` parameter and how it affects the result
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
pprint(f"Search time: {end - start:.4f} s")
results.sort( key=lambda x: x[1], reverse=True)
# for res in results:
#     pprint(res)


# 4. Precision, Recall, and F1-Score vs. Top-k Retrieved

# For varying k (top-k)
k_values = range(1, 11)  # Top-1 to Top-10
precision_scores = []
recall_scores = []
f1_scores = []


# 🔹 5. Definir umbral para considerar documentos como relevantes
# threshold = 0.71  # Ajustar según el caso
around = 0.05
thresholds = np.linspace(start=0, stop=1, num=100, dtype='float32')

for threshold in thresholds:
    # results = vector_store.similarity_search_with_score(query, k)
    # top_k_scores = [result[1] for result in results]
    # 🔹 6. Determinar documentos relevantes según el umbral
    predicted_relevant_docs = [r for r in results if r[1] >= threshold]
    predicted_relevant_falses = [r for r in results if r[1] >= threshold and r[1] <= threshold + around ]
    predicted_no_relevant = [r for r in results if r[1] < threshold and r[1] >= threshold - around ]
    # predicted_falses = [r for r in results if r[1] < threshold - around ]

    TP = len(predicted_relevant_docs) - len(predicted_relevant_falses)
    FP = len(predicted_relevant_falses)
    FN = len(predicted_no_relevant)
    TN = len(results) - (TP + FP + FN)

    # 🔹 7. Construcción automática de la matriz de confusión
    # TP = len(set(predicted_relevant_docs))  # Recuperados correctamente
    # FP = len(set(predicted_relevant_docs) - set(all_docs_text))  # Recuperados pero irrelevantes
    # FN = len(set(all_docs_text) - set(predicted_relevant_docs))  # No recuperados pero relevantes
    # TN = len(all_docs_text) - (TP + FP + FN)  # Documentos correctamente no recuperados

    # 🔹 8. Calcular métricas de evaluación
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_score)
    
    # 🔹 9. Mostrar Resultados
    # print("Documentos Recuperados:", predicted_relevant_docs)
    print("all_docs_text",len(results))
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, threshold: {threshold}")
    print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.2f}")

# Plot Precision, Recall, and F1-Score vs. Top-k Retrieved
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision', color='blue', lw=2)
plt.plot(thresholds, recall_scores, label='Recall', color='red', lw=2)
plt.plot(thresholds, f1_scores, label='F1-Score', color='green', lw=2)
plt.plot(thresholds, thresholds, label='Thresholds', color='yellow', lw=2)
plt.xlabel('Top-k Retrieved Documents')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score vs. Top-k Retrieved')
plt.legend()
plt.grid(True)
plt.show()