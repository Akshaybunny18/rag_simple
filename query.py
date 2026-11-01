import numpy as np


def query_rag(query, model, index, chunk_store, top_k=2):
	query_embedding = model.encode([query])
	query_embedding_np = np.array(query_embedding).astype("float32")
	distances, indices = index.search(query_embedding_np, top_k)

	retrieved = [chunk_store[i] for i in indices[0]]
	return retrieved, distances
