import faiss

class FaissIndexer:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
    
    def index(self, embeddings, document_ids):
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
    
    def search(self, query_embedding, top_k):
        query_embedding = query_embedding.reshape(1, self.embedding_dim).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]
