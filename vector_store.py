from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import os

def setup_vector_store(file_dicts, persist_dir):
    if os.path.exists(persist_dir):
        vector_store = ChromaVectorStore.from_params(
            collection_name="rfp_docs", persist_dir=persist_dir
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        all_nodes = [c for d in file_dicts.values() for c in d["docs"]]
        vector_store = ChromaVectorStore.from_params(
            collection_name="rfp_docs", persist_dir=persist_dir
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        index.insert_nodes(all_nodes)
    return index
