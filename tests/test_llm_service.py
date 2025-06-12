import unittest
from unittest.mock import patch, MagicMock
from src.llm_service import LLMService

class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.llm_service = LLMService()

    def test_generate_response(self):
        """Test response generation"""
        question = "What is RAG?"
        context = ["RAG stands for Retrieval Augmented Generation"]
        
        with patch('langchain_community.llms.ollama.Ollama') as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = "This is a test response"
            mock_ollama.return_value = mock_instance

            response = self.llm_service.generate_response(question, context)
            
            self.assertIsNotNone(response)
            self.assertEqual(response, "This is a test response")
            mock_instance.invoke.assert_called_once()

    def test_generate_response_with_temperature(self):
        """Test response generation with different temperature settings"""
        with patch('langchain_community.llms.ollama.Ollama') as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = "Test response"
            mock_ollama.return_value = mock_instance

            # Test with different temperature values
            for temp in [0.0, 0.5, 1.0]:
                self.llm_service.generate_response(
                    "Test question",
                    ["Test context"],
                    temperature=temp
                )
                
                # Verify the last call's arguments
                args, kwargs = mock_instance.invoke.call_args
                self.assertEqual(kwargs.get('temperature'), temp)

    @patch('langchain_community.llms.ollama.Ollama')
    def test_error_handling(self, mock_ollama):
        """Test error handling"""
        # Mock error case
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Test error")
        mock_ollama.return_value = mock_instance

        with self.assertRaises(Exception):
            self.llm_service.generate_response(
                "Test question",
                ["Test context"]
            )

    @patch('langchain_community.llms.ollama.Ollama')
    def test_context_formatting(self, mock_ollama):
        """Test context formatting in prompts"""
        # Mock Ollama response
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "Test response"
        mock_ollama.return_value = mock_instance
        
        # Test with multiple context items
        contexts = [
            "First piece of context",
            "Second piece of context",
            "Third piece of context"
        ]
        
        self.llm_service.generate_response("Test question", contexts)
        
        # Verify the call was made and get the prompt
        mock_instance.invoke.assert_called_once()
        args, _ = mock_instance.invoke.call_args
        prompt = args[0]
        
        # Verify all contexts are in the prompt
        for context in contexts:
            self.assertIn(context, prompt)

if __name__ == '__main__':
    unittest.main()
