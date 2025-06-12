from typing import List, Dict
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, collection_name: str = "documents", store_dir: str = None):
        self.collection_name = collection_name
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.storage_dir = store_dir if store_dir else os.path.join(os.getcwd(), ".vector_store")
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
    
    def add_documents(self, documents: List[Dict]):
        """Add documents and their embeddings to the vector store."""
        for doc in documents:
            self.documents.append({
                'content': doc['content'],
                'metadata': doc['metadata']
            })
            self.embeddings.append(np.array(doc['embedding']))
            # Generate an ID if not provided
            doc_id = doc.get('id', str(len(self.ids)))
            self.ids.append(doc_id)
        self._save_data()
    
    def search(self, query_embedding: List[float], n_results: int = 5, similarity_threshold: float = 0.3):
        """Search for similar documents using the query embedding."""
        if not self.embeddings:
            return {'documents': [], 'distances': [], 'ids': []}

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarities
        similarities = []
        for doc_vector in self.embeddings:
            # Normalize vectors
            query_norm = np.linalg.norm(query_vector)
            doc_norm = np.linalg.norm(doc_vector)
            
            if query_norm == 0 or doc_norm == 0:
                similarities.append(0)
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
            similarities.append(float(similarity))  # Convert to float for consistent comparison
        
        # Filter by threshold and get top N
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
        filtered_indices.sort(key=lambda i: similarities[i], reverse=True)
        
        if not filtered_indices:
            return {'documents': [], 'distances': [], 'ids': []}
            
        # Get top N results
        top_indices = filtered_indices[:n_results]
            
        return {
            'documents': [self.documents[i] for i in top_indices],
            'distances': [similarities[i] for i in top_indices],
            'ids': [self.ids[i] for i in top_indices]
        }
