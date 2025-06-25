from langchain_community.llms import Ollama
from typing import List
import os
import logging

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
        context_str = "\n\n".join(context) if context else ""
        return f"""You are a helpful AI assistant that provides accurate and clear answers based on the given context.

Given this context:
{context_str}

Question: {query}

Instructions:
1. Answer the question using ONLY the information from the context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question accurately."
3. Be clear and concise
4. Use a natural, conversational tone
5. DO NOT mention or refer to the context in your answer
6. DO NOT make up information that's not in the context

Answer: """

    def generate_response(self, query: str, context: List[str], temperature: float = 0.7) -> str:
        """Generate a response using the LLM with provided context."""
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
            raise ValueError("Temperature must be a number between 0 and 1")

        try:
            prompt = self._create_prompt(query, context)
            logger.info(f"Generating response with temperature {temperature}")
            
            response = self.llm.invoke(
                input=prompt,
                temperature=temperature,
                options={
                    "num_ctx": 4096,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Question:", "Instructions:", "\n\n"]
                }
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
