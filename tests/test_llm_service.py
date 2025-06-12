import unittest
from unittest.mock import patch, MagicMock
from src.llm_service import LLMService

class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.llm_service = LLMService()

    @patch('src.llm_service.requests.post')
    def test_generate_response(self, mock_post):
        """Test response generation"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a test response"
        }
        mock_post.return_value = mock_response

        question = "What is RAG?"
        context = ["RAG stands for Retrieval Augmented Generation"]
        
        response = self.llm_service.generate_response(question, context)
        
        self.assertIsNotNone(response)
        self.assertEqual(response, "This is a test response")

    @patch('src.llm_service.requests.post')
    def test_generate_response_with_temperature(self, mock_post):
        """Test response generation with different temperature settings"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response"
        }
        mock_post.return_value = mock_response

        # Test with different temperature values
        for temp in [0.0, 0.5, 1.0]:
            response = self.llm_service.generate_response(
                "Test question",
                ["Test context"],
                temperature=temp
            )
            
            # Verify temperature was passed correctly
            call_args = mock_post.call_args[1]['json']
            self.assertIn('temperature', call_args)
            self.assertEqual(call_args['temperature'], temp)

    @patch('src.llm_service.requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling"""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with self.assertRaises(Exception):
            self.llm_service.generate_response(
                "Test question",
                ["Test context"]
            )

    @patch('src.llm_service.requests.post')
    def test_context_formatting(self, mock_post):
        """Test context formatting in prompts"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response"
        }
        mock_post.return_value = mock_response

        # Test with multiple context items
        contexts = [
            "First piece of context",
            "Second piece of context",
            "Third piece of context"
        ]
        
        self.llm_service.generate_response("Test question", contexts)
        
        # Verify context was formatted correctly in the prompt
        call_args = mock_post.call_args[1]['json']
        prompt = call_args['prompt']
        
        for context in contexts:
            self.assertIn(context, prompt)

if __name__ == '__main__':
    unittest.main()
