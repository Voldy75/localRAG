import unittest
from fastapi.testclient import TestClient
from src.main import app
import os
import shutil

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_documents')
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_upload_document(self):
        """Test document upload endpoint"""
        # Create test document
        test_file_path = os.path.join(self.test_dir, 'test.txt')
        with open(test_file_path, 'w') as f:
            f.write("This is a test document")

        # Test file upload
        with open(test_file_path, 'rb') as f:
            response = self.client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_process_all_documents(self):
        """Test processing all documents endpoint"""
        # Create test documents
        for i in range(3):
            with open(os.path.join(self.test_dir, f'test_{i}.txt'), 'w') as f:
                f.write(f"Test document {i}")

        response = self.client.post("/process-all-documents")
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("processed_files", result)
        self.assertIn("failed_files", result)

    def test_query_endpoint(self):
        """Test query endpoint"""
        # First upload and process a test document
        test_file_path = os.path.join(self.test_dir, 'test.txt')
        with open(test_file_path, 'w') as f:
            f.write("RAG is a Retrieval Augmented Generation system")

        with open(test_file_path, 'rb') as f:
            self.client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )

        # Test querying
        response = self.client.post(
            "/query",
            json={
                "question": "What is RAG?",
                "temperature": 0.7
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        self.assertIn("sources", response.json())

    def test_chat_completion_endpoint(self):
        """Test OpenAI-compatible chat completion endpoint"""
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is RAG?"}
                ],
                "model": "deepseek-r1",
                "temperature": 0.7
            }
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("choices", result)
        self.assertIn("message", result["choices"][0])

    def test_error_handling(self):
        """Test error handling in endpoints"""
        # Test invalid file type
        with open(os.path.join(self.test_dir, 'test.xyz'), 'w') as f:
            f.write("Test content")

        with open(os.path.join(self.test_dir, 'test.xyz'), 'rb') as f:
            response = self.client.post(
                "/upload",
                files={"file": ("test.xyz", f, "text/plain")}
            )
        
        self.assertEqual(response.status_code, 400)

        # Test invalid query
        response = self.client.post(
            "/query",
            json={"invalid": "request"}
        )
        
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

if __name__ == '__main__':
    unittest.main()
