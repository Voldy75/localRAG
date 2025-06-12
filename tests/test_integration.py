import unittest
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_service import LLMService
import os
import tempfile
import shutil

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(store_dir=self.test_dir)
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_full_rag_pipeline(self):
        """Test the complete RAG pipeline from document processing to response generation"""
        # 1. Create test document
        test_doc = os.path.join(self.test_dir, 'test.txt')
        test_content = """
        RAG (Retrieval Augmented Generation) is a technique that combines retrieval-based and generation-based approaches.
        It first retrieves relevant documents from a collection, then uses them to generate contextually informed responses.
        This helps improve accuracy and provides source attribution for generated responses.
        """
        with open(test_doc, 'w') as f:
            f.write(test_content)

        # 2. Process document
        processed_docs = self.doc_processor.process_document(test_doc)
        self.assertIsNotNone(processed_docs)

        # 3. Create embeddings and store in vector store
        for doc in processed_docs if isinstance(processed_docs, list) else [processed_docs]:
            embedding = self.doc_processor.create_embeddings([doc['content']])[0]
            doc['embedding'] = embedding
            self.vector_store.add_documents([doc])

        # 4. Query the system
        test_query = "What is RAG?"
        query_embedding = self.doc_processor.create_embeddings([test_query])[0]
        
        # 5. Search for relevant documents
        search_results = self.vector_store.search(query_embedding)
        self.assertIsNotNone(search_results)
        self.assertTrue(len(search_results['documents']) > 0)

        # 6. Generate response
        response = self.llm_service.generate_response(
            test_query,
            [doc['content'] for doc in search_results['documents']],
            temperature=0.7
        )
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)

    def test_multiple_document_types(self):
        """Test processing and querying with different document types"""
        # Create test documents of different types
        test_files = {
            'text.txt': "This is a test text document.",
            'doc.docx': "This is a test Word document.",
            'document.pdf': "This is a test PDF document."
        }

        processed_docs = []
        for filename, content in test_files.items():
            file_path = os.path.join(self.test_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Process each document
            result = self.doc_processor.process_document(file_path)
            if result:
                if isinstance(result, list):
                    processed_docs.extend(result)
                else:
                    processed_docs.append(result)

        # Verify processing
        self.assertTrue(len(processed_docs) > 0)

        # Create and store embeddings
        for doc in processed_docs:
            embedding = self.doc_processor.create_embeddings([doc['content']])[0]
            doc['embedding'] = embedding
            self.vector_store.add_documents([doc])

        # Test querying
        test_queries = [
            "What is in the text document?",
            "What is in the Word document?",
            "What is in the PDF document?"
        ]

        for query in test_queries:
            query_embedding = self.doc_processor.create_embeddings([query])[0]
            results = self.vector_store.search(query_embedding)
            self.assertTrue(len(results['documents']) > 0)

    def test_error_recovery(self):
        """Test system's ability to handle and recover from errors"""
        # Test with invalid document
        invalid_doc = os.path.join(self.test_dir, 'invalid.txt')
        with open(invalid_doc, 'w') as f:
            f.write('')  # Empty file

        # Should handle empty file gracefully
        result = self.doc_processor.process_document(invalid_doc)
        self.assertIsNone(result)

        # Test with invalid query
        empty_query = ""
        query_embedding = self.doc_processor.create_embeddings([empty_query])[0]
        results = self.vector_store.search(query_embedding)
        self.assertEqual(len(results['documents']), 0)

        # Test with very large document
        large_doc = os.path.join(self.test_dir, 'large.txt')
        with open(large_doc, 'w') as f:
            f.write("test " * 10000)  # Very large document

        # Should handle large document by chunking
        result = self.doc_processor.process_document(large_doc)
        self.assertIsNotNone(result)
        if isinstance(result, list):
            self.assertTrue(len(result) > 1)  # Should be split into chunks

if __name__ == '__main__':
    unittest.main()
