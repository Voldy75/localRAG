from typing import List, Dict
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.storage_dir = os.path.join(os.getcwd(), ".vector_store")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_data()
    
    def _load_data(self):
        """Load existing data from disk if available."""
        data_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.embeddings = [np.array(emb) for emb in data['embeddings']]
                self.ids = data['ids']
    
    def _save_data(self):
        """Save current data to disk."""
        data_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        data = {
            'documents': self.documents,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'ids': self.ids
        }
        with open(data_file, 'w') as f:
            json.dump(data, f)
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]], ids: List[str]):
        """Add documents and their embeddings to the vector store."""
        self.documents.extend([{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in documents])
        self.embeddings.extend([np.array(emb) for emb in embeddings])
        self.ids.extend(ids)
        self._save_data()
    
    def search(self, query_embedding: List[float], n_results: int = 5):
        """Search for similar documents using the query embedding."""
        if not self.embeddings:
            return {'documents': [], 'distances': [], 'ids': []}
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        embeddings_array = np.array(self.embeddings)
        
        # Calculate cosine similarity
        cosine_scores = np.dot(query_embedding, embeddings_array.T)[0] / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embeddings_array, axis=1)
        )
        
        # Get top results
        n_results = min(n_results, len(self.embeddings))
        top_indices = np.argsort(cosine_scores)[-n_results:][::-1]
        
        return {
            'documents': [self.documents[i] for i in top_indices],
            'distances': cosine_scores[top_indices].tolist(),
            'ids': [self.ids[i] for i in top_indices]
        }
