from pprint import pprint
import time
import json
import weaviate

from langchain.schema import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
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

    pprint('Connecting to weaviate vector database')
    weaviate_client = weaviate.connect_to_local()
    
    pprint('Indexing documents to the vector store')
    start = time.time()
    db = WeaviateVectorStore.from_documents(documents=documents, embedding= embeddings_model, client=weaviate_client)
    end = time.time()

    return db, weaviate_client, start, end

model_name= 'sentence-transformers/all-MiniLM-L6-v2'
k=10
query= "educación"
vector_store, weaviate_client, start, end= create_vector_store(model_name=model_name)
pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
# Return list of documents most similar to the query text and `cosine distance` in float for each
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
weaviate_client.close()

pprint(f"Search time: {end - start:.4f} s")

for res in results:
    pprint(res)
