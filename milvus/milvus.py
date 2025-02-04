import json
from pprint import pprint
from uuid import uuid4
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import time
from pymilvus import MilvusClient

language= 'es'
URI= "http://localhost:19530"
# Load dataset
def create_vector_store(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        collection_name='vocabs', 
        create_index=False):
    
    # Load Model
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)
    # embeddings= embeddings_model.embed_documents(documents)

    if create_index:
        pprint('Load dataset')
        with open('../data/unesco-parser-'+language+'.json', 'r', encoding='utf-8') as f:
            dataset= json.load(f)

        # Convert to LangChain documents
        pprint('Convert to LangChain documents')
        documents= []
        for d in dataset:
            documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
        
        # uuids = [str(uuid4()) for _ in range(len(documents))]
        pprint('Create Vector store')
        vector_store = Milvus(
                    embedding_function= embeddings_model,
                    collection_name= collection_name,
                    connection_args={"uri": URI},
                    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
                    enable_dynamic_field= True
                )
        # vector_store.add_documents(documents=documents, ids=uuids)
        # Crear índice Milvus
        start = time.time()
        pprint('Store documents in the collection')
        milvus_db = vector_store.from_documents(
                            documents=documents, 
                            embedding=embeddings_model, 
                            collection_name= collection_name,
                            connection_args={"uri": URI},
                            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
                        )
        end = time.time()
        return milvus_db, start, end
    else:
        start = time.time()
        milvus_db = Milvus(
                embeddings_model,
                connection_args={"uri": URI},
                collection_name= collection_name,
            )
        end = time.time()
        return milvus_db, start, end


# This is a sentence-transformers model: 
# It maps sentences & paragraphs to a 384 dimensional dense vector space
# and can be used for tasks like clustering or semantic search.
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# model                                                 dimensional dense vector
# sentence-transformers/all-mpnet-bembeddingsase-v2         768
# sentence-transformers/all-MiniLM-L12-v2                   384
# sentence-transformers/all-MiniLM-L6-v2                    384 (have the best results in the example)

# client = MilvusClient(URI)
# client.drop_index(collection_name='vocabs_all_MiniLM_L6_v2',index_name='vector')
# client.drop_collection(collection_name='vocabs_all_MiniLM_L6_v2')
k=10
query= "educación"
vector_store, start, end= create_vector_store(
                            model_name='sentence-transformers/all-MiniLM-L6-v2', 
                            collection_name='vocabs_all_MiniLM_L6_v2', 
                            create_index=True)

pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
pprint(f"Search time: {end - start:.4f} s")

for res in results:
    pprint(res)