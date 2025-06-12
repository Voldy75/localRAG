from langchain_community.llms import Ollama
from typing import List
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name: str = "deepseek-r1"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"Initializing Ollama with base_url: {base_url}")
        try:
            self.llm = Ollama(
                model=model_name,
                base_url=base_url
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise

    def _create_prompt(self, query: str, context: List[str]) -> str:
        """Create a prompt combining the context and query."""
        context_str = "\n".join(context)
        return f"""Based on the following context, answer the question.
Context:
{context_str}

Question: {query}
Answer:"""

    def generate_response(self, query: str, context: List[str], temperature: float = 0.7) -> str:
        """Generate a response using the LLM with provided context."""
        try:
            prompt = self._create_prompt(query, context)
            logger.info(f"Generating response for query with temperature {temperature}")
            response = self.llm.invoke(
                prompt,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
