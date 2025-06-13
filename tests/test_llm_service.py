import unittest
from unittest.mock import patch, MagicMock
from src.llm_service import LLMService

class TestLLMService(unittest.TestCase):
    def setUp(self):
        patcher = patch('src.llm_service.Ollama')
        self.mock_ollama = patcher.start()
        self.mock_instance = MagicMock()
        self.mock_ollama.return_value = self.mock_instance
        self.llm_service = LLMService()
        self.addCleanup(patcher.stop)

    def test_generate_response(self):
        """Test response generation"""
        # Setup
        self.mock_instance.invoke.return_value = "This is a test response"
        question = "What is RAG?"
        context = ["RAG stands for Retrieval Augmented Generation"]

        # Execute
        response = self.llm_service.generate_response(question, context)

        # Verify
        self.assertIsNotNone(response)
        self.assertEqual(response, "This is a test response")
        self.mock_instance.invoke.assert_called_once()

    def test_generate_response_with_temperature(self):
        """Test response generation with different temperature settings"""
        self.mock_instance.invoke.return_value = "Test response"

        # Test with different temperature values
        for temp in [0.0, 0.5, 1.0]:
            self.mock_instance.invoke.reset_mock()

            # Execute
            self.llm_service.generate_response(
                "Test question",
                ["Test context"],
                temperature=temp
            )

            # Verify
            self.mock_instance.invoke.assert_called_once()
            _, kwargs = self.mock_instance.invoke.call_args
            self.assertEqual(kwargs.get('temperature'), temp)

    def test_error_handling(self):
        """Test error handling"""
        # Setup error case
        self.mock_instance.invoke.side_effect = Exception("Test error")

        # Execute and verify
        with self.assertRaises(Exception):
            self.llm_service.generate_response(
                "Test question",
                ["Test context"]
            )

    def test_context_formatting(self):
        """Test context formatting in prompts"""
        # Setup
        self.mock_instance.invoke.return_value = "Test response"
        contexts = [
            "First piece of context",
            "Second piece of context",
            "Third piece of context"
        ]

        # Execute
        self.llm_service.generate_response("Test question", contexts)

        # Verify
        self.mock_instance.invoke.assert_called_once()
        args, _ = self.mock_instance.invoke.call_args
        prompt = args[0]

        # Verify all contexts are in the prompt
        for context in contexts:
            self.assertIn(context, prompt)

if __name__ == '__main__':
    unittest.main()
