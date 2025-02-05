from pprint import pprint
import time
import json
import faiss

from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

language= 'es'

def create_vector_store(model_name="sentence-transformers/all-mpnet-base-v2"):
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
    # Create index FAISS
    pprint('Creating FAISS index')
    index = faiss.IndexBinaryFlat(len(embeddings_model.embed_query(documents[0].page_content)))
    
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy= DistanceStrategy.COSINE
    )
    pprint('Indexing documents to the vector store')
    start = time.time()
    faiss_db = vector_store.from_documents(documents=documents, embedding=embeddings_model)
    end = time.time()
    return faiss_db, start, end


model_name= 'sentence-transformers/all-MiniLM-L6-v2'
k=10
query= "educación"
vector_store, start, end= create_vector_store(model_name=model_name)
pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
pprint(f"Search time: {end - start:.4f} s")

for res in results:
    pprint(res)