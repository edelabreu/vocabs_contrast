import json
import time
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

    pprint('Load dataset')
    with open('../data/unesco-parser-'+language+'.json', 'r', encoding='utf-8') as f:
        dataset= json.load(f)

    # Convert to LangChain documents
    pprint('Convert to LangChain documents')
    documents= []
    for d in dataset:
        documents.append(Document(page_content=d['str'], metadata={'conceptLabel':d['conceptLabel'], 'conceptUri':d['conceptUri'], }))
    pprint('The amount of documents is '+str(len(documents)))
    
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
k=10
query= "educación"
vector_store, start, end= create_vector_store(model_name=model_name)
pprint(f"Time for create or load vector store : {end - start:.4f} s")

start = time.time()
# TODO: Find information about `search_k` parameter and how it affects the result
results = vector_store.similarity_search_with_score(query=query,k=k)
end = time.time()
pprint(f"Search time: {end - start:.4f} s")
results.sort( key=lambda x: x[1], reverse=True)
for res in results:
    pprint(res)