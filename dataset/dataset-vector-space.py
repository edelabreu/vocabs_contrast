
import os
import json
import time
import chromadb
import faiss
import psutil
import pandas as pd
import numpy as np

from pprint import pprint

from langchain.schema import Document
from langchain_milvus import Milvus
from annoy import AnnoyIndex
from langchain_community.vectorstores import FAISS, Annoy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

process = psutil.Process(os.getpid())

def get_resources(row:list|None = None, time_start: float = None):
    memory = process.memory_info().rss  
    cpu = process.cpu_percent(interval=0.1) 
    timer = time.time()
    if row is None:
        return timer, memory, cpu
    else: 
        row.append((timer - time_start) * 1000) # time in MS
        row.append(timer - time_start) # time in S
        row.append(f"{memory / (1024 * 1024):.2f}")
        row.append(cpu)


LANGUAGE = ['es', 'en']
PATH = '../data'
Milvus_URI= "http://192.168.50.21:19530"

MODELS =[
    {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'str': 'all_MiniLM_L6_v2',     
        'size':int(384)
    },
    {
        'name': 'sentence-transformers/all-MiniLM-L12-v2',
        'str': 'all_MiniLM_L12_v2',
        'size':int(384)
    },
    {
        'name': 'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
        'str': 'paraphrase_xlm_r_multilingual_v1',
        'size':int(768)
    },
    {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'str': 'all_mpnet_base_v2',
        'size':int(768)
    },
    {
        'name': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
        'str': 'paraphrase_MiniLM_L6_v2',
        'size':int(384)
    },
    {
        'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'str': 'paraphrase_multilingual_MiniLM_L12_v2',
        'size':int(384)
    },
    {
        'name': 'sentence-transformers/paraphrase-distilroberta-base-v2',
        'str': 'paraphrase_distilroberta_base_v2',
        'size':int(768)
    },
]

for language in LANGUAGE:
    UNESCO_PATH = f"{PATH}/unesco-dataset-{language}.json"
    AGROVOC_PATH = f"{PATH}/agrovoc-dataset-{language}.json"
    EUROVOC_PATH = f"{PATH}/eurovoc-dataset-{language}.json"
    
    # LOAD DATASETS
    documents= []
    
    pprint('Load unesco-dataset')
    with open(UNESCO_PATH, 'r', encoding='utf-8') as f:
        unesco_dataset= json.load(f)

    pprint('Convert unesco_dataset to LangChain documents')
    for d in unesco_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'][0], 'conceptUri':d['conceptUri'], }))
    
    pprint('Load agrovoc-dataset')
    with open(AGROVOC_PATH, 'r', encoding='utf-8') as f:
        agrovoc_dataset= json.load(f)

    pprint('Convert agrovoc_dataset to LangChain documents')
    for d in agrovoc_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'][0], 'conceptUri':d['conceptUri'], }))
    
    pprint('Load eurovoc-dataset')
    with open(EUROVOC_PATH, 'r', encoding='utf-8') as f:
        eurovoc_dataset= json.load(f)

    pprint('Convert eurovoc_dataset to LangChain documents')
    for d in eurovoc_dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'][0], 'conceptUri':d['conceptUri'], }))
    
    

    pprint(f"Execution for the {language.upper()} language")
    pprint(f"The amount of documents is: {len(documents)}")

    # --------------------------------------------------------------------------------------
    # FAISS --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    for m in MODELS:
        embeddings_model= HuggingFaceEmbeddings(model_name=m['name'])
        pprint(m['name'])
        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexFlat_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources()
        index = faiss.IndexFlat(m['size'])
        get_resources(row, time_start)
        
        
        time_start, memory_start, cpu_start = get_resources()
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)
        
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        
        time.sleep(5)
        
        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexHNSW_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.IndexHNSW(m['size'])
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)
        
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        
        time.sleep(5)
        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexIVFFlat_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.index_factory(m['size'], "IVF1000_NSG64,Flat") #faiss.IndexIVFFlat(m['size'])
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        get_resources(row, time_start)
        
        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)

        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexIVFPQ_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.index_factory(m['size'], "IVF1000_NSG64,PQ2x8") #faiss.IndexIVFPQ(m['size'])
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexLSH_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.IndexLSH(m['size'], 23)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating FAISS index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/faiss_db/IndexPQ_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.index_factory(m['size'], "NSG64,PQ2x8") #faiss.IndexPQ(m['size'])
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)

        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)



    # --------------------------------------------------------------------------------------
    # MILVUS -------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    for m in MODELS:
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexFLAT_{m['str']}_{language}"
        row.append(collection_name)
        # Milvus does not have an index generator
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        time_start, memory_start, cpu_start = get_resources()
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "FLAT", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "FLAT", "metric_type": "COSINE"})
        get_resources(row, time_start)

        # Milvus does not save locally
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        # load data from database 
        time_start, memory_start, cpu_start = get_resources()
        milvus_db = Milvus(
                embeddings_model,
                connection_args={"uri": Milvus_URI},
                collection_name= collection_name,
            )
        get_resources(row, time_start)
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexIVF_FLAT_{m['str']}_{language}"
        row.append(collection_name)
        # Milvus does not have an index generator
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)
        time_start, memory_start, cpu_start = get_resources()
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"})
        get_resources(row, time_start)

        # Milvus does not save locally
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        # load data from database 
        time_start, memory_start, cpu_start = get_resources()
        milvus_db = Milvus(
                embeddings_model,
                connection_args={"uri": Milvus_URI},
                collection_name= collection_name,
            )
        get_resources(row, time_start)
        
        # saving data        
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexHNSW_{m['str']}_{language}"
        row.append(collection_name)
        # Milvus does not have an index generator
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)
        time_start, memory_start, cpu_start = get_resources()
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "HNSW", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "HNSW", "metric_type": "COSINE"})
        get_resources(row, time_start)
        # Milvus does not save locally
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        # load data from database 
        time_start, memory_start, cpu_start = get_resources()
        milvus_db = Milvus(
                embeddings_model,
                connection_args={"uri": Milvus_URI},
                collection_name= collection_name,
            )
        get_resources(row, time_start)
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexIVF_PQ_{m['str']}_{language}"
        row.append(collection_name)
        # Milvus does not have an index generator
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)
        time_start, memory_start, cpu_start = get_resources()
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={
                        "index_type": "IVF_PQ", 
                        "metric_type": "COSINE", 
                        "params": {
                            "nlist": 1024,  # Número de listas
                            "m": 8,         # Número de subcódigos
                            "nbits": 8      # Número de bits por subcódigo
                        }
                    }, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={
                                "index_type": "IVF_PQ", 
                                "metric_type": "COSINE", 
                                "params": {
                                    "nlist": 1024,  # Número de listas
                                    "m": 8,         # Número de subcódigos
                                    "nbits": 8      # Número de bits por subcódigo
                                }
                            }
                        )
        get_resources(row, time_start)
        # Milvus does not save locally
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        # load data from database 
        time_start, memory_start, cpu_start = get_resources()
        milvus_db = Milvus(
                embeddings_model,
                connection_args={"uri": Milvus_URI},
                collection_name= collection_name,
            )
        get_resources(row, time_start)
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

    # --------------------------------------------------------------------------------------
    # ANNOY --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    for m in MODELS:
        pprint('Creating ANNOY index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        folder_path=f"{PATH}/annoy_db/{metric}_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources()
        index = AnnoyIndex(
            f=m['size'], 
            metric=metric) 
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        vector_store = Annoy(
            embedding_function= embeddings_model,
            index=index,
            metric=metric,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.save_local(folder_path=folder_path)
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.load_local(folder_path=folder_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        get_resources(row, time_start)
        
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)


    # --------------------------------------------------------------------------------------
    # CHROMA -------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    for m in MODELS:
        pprint('Creating CHROMA index')
        from langchain_chroma import Chroma
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        folder_path=f"{PATH}/chroma_db/{index_name}_{m['str']}_{language}"
        row.append(folder_path)

        # Chroma does not have to calculate an index build time
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        time_start, memory_start, cpu_start = get_resources()
        vector_store = Chroma(
            collection_name= collection_name,
            embedding_function= embeddings_model,
            persist_directory= folder_path,  # Where to save data locally
            collection_metadata={"hnsw:space": "cosine"} 
        )
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model, collection_metadata={"hnsw:space": "cosine"})
        get_resources(row, time_start)

        time_start, memory_start, cpu_start = get_resources()
        db.persist()
        get_resources(row, time_start)
        
        from chromadb.config import Settings
        time_start, memory_start, cpu_start = get_resources()
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=folder_path))
        db = Chroma(client=client, embedding_function=embeddings_model)
        get_resources(row, time_start)
        
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)



    # --------------------------------------------------------------------------------------
    # WEAVIATE -----------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    for m in MODELS:
        pprint('Creating WEAVIATE index')
        import weaviate
        from langchain_weaviate.vectorstores import WeaviateVectorStore
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name=f"weaviate_index_{m['str']}_{language}"
        row.append(index_name)

        weaviate_host = '192.168.50.20'

        time_start, memory_start, cpu_start = get_resources()
        weaviate_client = weaviate.connect_to_local(
            host= weaviate_host
        )
        get_resources(row, time_start)

        # Weaviate does not have to calculate a time to create the vector space
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        time_start, memory_start, cpu_start = get_resources()
        db = WeaviateVectorStore.from_documents(
                                    documents=documents, 
                                    embedding= embeddings_model, 
                                    client=weaviate_client
                                )
        get_resources(row, time_start)

        # Weaviate does not have to calculate save_local time
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

        weaviate_client.close()

        time_start, memory_start, cpu_start = get_resources()
        weaviate_client = weaviate.connect_to_local(
            host= weaviate_host
        )
        weaviate_store = WeaviateVectorStore(
                            weaviate_client, 
                            embedding_function=embeddings_model, 
                            index_name=index_name
                        )
        weaviate_client.close()
        get_resources(row, time_start)
        
        # saving data
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        time.sleep(5)

        