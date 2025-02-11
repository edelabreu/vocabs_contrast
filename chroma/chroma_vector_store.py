from pprint import pprint
import time
import json

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

language= 'es'

def create_vector_store(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name='vocabs_all_MiniLM_L6_v2'):
    # Load Model
    embeddings_model= HuggingFaceEmbeddings(model_name=model_name)

    pprint('Load dataset')
    with open('../data/unesco-parser-'+language+'.json', 'r', encoding='utf-8') as f:
        dataset= json.load(f)

    # Convert to LangChain documents
    pprint('Convert to LangChain documents')
    documents= []
    for d in dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
    pprint('The amount of documents is '+str(len(documents)))

    pprint('Create Vector store')
    vector_store = Chroma(
        collection_name= collection_name,
        embedding_function= embeddings_model,
        persist_directory= "../data/chroma_db",  # Where to save data locally
        collection_metadata={"hnsw:space": "cosine"} 
    )
    ####
    # Available distance metrics are: 'cosine', 'l2' and 'ip'.
    # cosine: cosine
    # l2: euclidean
    # ip: max inner product

    pprint('Store documents in the collection')
    start = time.time()
    vector_store.add_documents(
        documents=documents, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    end = time.time()
    return vector_store, start, end


k=10
query= "educación"
vector_store, start, end= create_vector_store(
                            model_name='sentence-transformers/all-MiniLM-L6-v2', 
                            collection_name='vocabs_all_MiniLM_L6_v2')

pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
pprint(f"Search time: {end - start:.4f} s")

for res in results:
    pprint(res)