import unittest
import os
from src.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.doc_processor = DocumentProcessor()
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_files')
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_supported_file_types(self):
        """Test that supported file types are correctly identified"""
        self.assertTrue(self.doc_processor.is_supported_file('test.txt'))
        self.assertTrue(self.doc_processor.is_supported_file('test.pdf'))
        self.assertTrue(self.doc_processor.is_supported_file('test.docx'))
        self.assertFalse(self.doc_processor.is_supported_file('test.xyz'))

    def test_text_file_processing(self):
        """Test processing of text files"""
        test_file = os.path.join(self.test_dir, 'test.txt')
        test_content = "This is a test document.\nIt has multiple lines.\nTesting document processing."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        result = self.doc_processor.process_document(test_file)
        self.assertIsNotNone(result)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['source'], 'test.txt')

    def test_create_embeddings(self):
        """Test creation of embeddings"""
        texts = ["This is a test document", "Another test document"]
        embeddings = self.doc_processor.create_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(emb, list) for emb in embeddings))
        self.assertTrue(all(isinstance(val, float) for emb in embeddings for val in emb))

    def test_document_splitting(self):
        """Test document splitting functionality"""
        long_text = " ".join(["Test document"] * 1000)  # Create a long document
        test_file = os.path.join(self.test_dir, 'long.txt')
        
        with open(test_file, 'w') as f:
            f.write(long_text)
        
        result = self.doc_processor.process_document(test_file)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)  # Should return list of chunks
        self.assertTrue(all(len(chunk['content'].split()) <= 512 for chunk in result))  # Check chunk size

if __name__ == '__main__':
    unittest.main()
