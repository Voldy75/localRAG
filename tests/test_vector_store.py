import unittest
from unittest.mock import patch, MagicMock
from src.vector_store import VectorStore
import numpy as np
import os
import shutil

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_vector_store')
        os.makedirs(self.test_dir, exist_ok=True)
        self.vector_store = VectorStore(store_dir=self.test_dir)

    def tearDown(self):
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_and_search(self):
        """Test adding documents and searching"""
        # Create test documents with embeddings
        docs = [
            {
                "content": "Test document one",
                "metadata": {"source": "doc1.txt"},
                "embedding": np.random.rand(384).tolist(),  # Assuming 384-dim embeddings
                "id": "doc1"
            },
            {
                "content": "Test document two",
                "metadata": {"source": "doc2.txt"},
                "embedding": np.random.rand(384).tolist(),
                "id": "doc2"
            }
        ]

        # Add documents
        self.vector_store.add_documents(docs)

        # Search with a random query vector
        query_vector = np.random.rand(384).tolist()
        results = self.vector_store.search(query_vector)

        self.assertIsNotNone(results)
        self.assertIn('documents', results)
        self.assertIn('distances', results)
        self.assertEqual(len(results['documents']), min(len(docs), 5))  # Default top-k is 5    def test_similarity_threshold(self):
        """Test similarity threshold filtering"""
        # Create test documents with controlled similarity
        base_vector = np.ones(384) / np.sqrt(384)  # Unit vector
        
        # Create a very similar vector (small noise)
        similar_noise = np.random.rand(384) * 0.1
        similar_vector = base_vector + similar_noise
        similar_vector = similar_vector / np.linalg.norm(similar_vector)  # Normalize
        
        # Create a different vector (orthogonal to base_vector)
        diff_vector = np.random.rand(384)
        diff_vector = diff_vector - np.dot(diff_vector, base_vector) * base_vector  # Make orthogonal
        diff_vector = diff_vector / np.linalg.norm(diff_vector)  # Normalize
        
        docs = [
            {
                "content": "Similar document",
                "metadata": {"source": "similar.txt"},
                "embedding": similar_vector.tolist()
            },
            {
                "content": "Different document",
                "metadata": {"source": "different.txt"},
                "embedding": diff_vector.tolist()
            }
        ]

        self.vector_store.add_documents(docs)
        
        # Search with base_vector and high similarity threshold
        results = self.vector_store.search(base_vector.tolist(), similarity_threshold=0.8)
        
        # Should only return the similar document
        self.assertEqual(len(results['documents']), 1)
        self.assertEqual(results['documents'][0]['metadata']['source'], 'similar.txt')
        self.assertEqual(results['documents'][0]['metadata']['source'], 'similar.txt')

    def test_persistence(self):
        """Test vector store persistence"""
        # Add test documents
        docs = [{
            "content": "Test persistence",
            "metadata": {"source": "test.txt"},
            "embedding": np.random.rand(384).tolist()
        }]
        
        self.vector_store.add_documents(docs)
        
        # Create new instance with same directory
        new_store = VectorStore(store_dir=self.test_dir)
        
        # Search should work with new instance
        query_vector = np.random.rand(384).tolist()
        results = new_store.search(query_vector)
        
        self.assertIsNotNone(results)
        self.assertEqual(len(results['documents']), 1)

if __name__ == '__main__':
    unittest.main()
