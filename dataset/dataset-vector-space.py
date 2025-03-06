
import os
import json
import time
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

# construir los espacios vectoriales para cada base de dato y cada tipo de indice
def get_resources(is_start = False):
    if is_start:
        process = psutil.Process(os.getpid())  # get the current process
        memory = process.memory_info().rss  
        cpu = process.cpu_percent(interval=0.1) 
        timer = time.time()
    else:
        timer = time.time()
        process = psutil.Process(os.getpid())  # get the current process
        memory = process.memory_info().rss  
        cpu = process.cpu_percent(interval=0.1) 
    return timer, memory, cpu

def calc_resources(row:list, time_start:float, time_end:float, memory_start, memory_end, cpu_start:float, cpu_end:float):
    
    time_difference_s = (time_end - time_start)
    row.append(time_difference_s)
    row.append(time_difference_s * 10**3)
    row.append(memory_end - memory_start)
    row.append(cpu_end - cpu_start)



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
    # pprint(f"Execution time of program is: {time_difference_s} s")
    # pprint(f"Execution time of program is: {time_difference_ms} ms") 
    # pprint(f"Used memory (in bytes): {memory_end - memory_start}")
    # pprint(f"Used CPU (%): {cpu_end - cpu_start}")


    stats = []
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

        time_start, memory_start, cpu_start = get_resources(True)
        index = faiss.IndexFlat(m['size'])
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        
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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        
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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path, index_name=index_name)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)

        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        



        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexFLAT_{m['str']}_{language}"
        row.append(collection_name)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "FLAT", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "FLAT", "metric_type": "COSINE"})
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexIVF_FLAT_{m['str']}_{language}"
        row.append(collection_name)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"})
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexHNSW_{m['str']}_{language}"
        row.append(collection_name)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": Milvus_URI},
                    index_params={"index_type": "HNSW", "metric_type": "COSINE"}, #FLAT, HNSW, IVF_FLAT, IVF_PQ
                    enable_dynamic_field= True
                )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model,
                            collection_name= collection_name,
                            connection_args={"uri": Milvus_URI},
                            index_params={"index_type": "HNSW", "metric_type": "COSINE"})
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        # saving data
        stats.append(row)
        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

        # --------------------------------------------------------------------------------------
        pprint('Creating Milvus index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        collection_name=f"IndexIVF_PQ_{m['str']}_{language}"
        row.append(collection_name)

        time_start, memory_start, cpu_start = get_resources(True)
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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
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
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        # saving data
        stats.append(row)

        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        pprint('Creating ANNOY index')
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        folder_path=f"{PATH}/annoy_db/{metric}_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        index = AnnoyIndex(
            f=m['size'], 
            metric=metric) 
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = Annoy(
            embedding_function= embeddings_model,
            index=index,
            metric=metric,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        db.save_local(folder_path=folder_path)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)

        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        


        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        pprint('Creating CHROMA index')
        from langchain_chroma import Chroma
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        folder_path=f"{PATH}/chroma_db/{metric}_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        # Chroma does not have to calculate an index build time
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        vector_store = Chroma(
            collection_name= collection_name,
            embedding_function= embeddings_model,
            persist_directory= folder_path,  # Where to save data locally
            collection_metadata={"hnsw:space": "cosine"} 
        )
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = vector_store.from_documents(documents=documents, embedding=embeddings_model, collection_metadata={"hnsw:space": "cosine"})
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        # db.save_local(folder_path=folder_path)
        # Chroma does not have to calculate an index save time
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)

        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        



        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        pprint('Creating WEAVIATE index')
        import weaviate
        from langchain_weaviate.vectorstores import WeaviateVectorStore
        row = []
        # saving model's name
        row.append(m['name'])
        
        # saving index's name
        index_name ='index' 
        metric='angular' # metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        folder_path=f"{PATH}/chroma_db/{metric}_{m['str']}_{language}"
        row.append(folder_path)

        time_start, memory_start, cpu_start = get_resources(True)
        weaviate_client = weaviate.connect_to_local(
            host= '192.168.50.20'
        )
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        # Weaviate does not have to calculate a time to create the vector space
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        time_start, memory_start, cpu_start = get_resources(True)
        db = WeaviateVectorStore.from_documents(documents=documents, embedding= embeddings_model, client=weaviate_client)
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)
        
        time_start, memory_start, cpu_start = get_resources(True)
        # db.save_local(folder_path=folder_path)
        # Weaviate does not have to calculate an index save time
        time_end, memory_end, cpu_end = get_resources()
        calc_resources(row, time_start, time_end, memory_start, memory_end, cpu_start, cpu_end)

        # saving data
        stats.append(row)

        df = pd.DataFrame([row])

        with pd.ExcelWriter('../data/dataset-vector-space.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Escribir los nuevos datos al final de la hoja (por ejemplo, 'Hoja1')
            df.to_excel(writer, sheet_name='Hoja1', index=False, header=False, startrow=writer.sheets['Hoja1'].max_row)
        

        