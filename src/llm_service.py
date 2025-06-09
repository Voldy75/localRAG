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
    
    def generate_response(self, query: str, context: List[str], temperature: float = 0.7) -> str:
        """Generate a response using the LLM with provided context."""
        try:
            prompt = self._create_prompt(query, context)
            logger.info(f"Generating response for query with temperature {temperature}")
            # Using invoke instead of predict as per langchain 0.1.7+ recommendations
            return self.llm.invoke(prompt, temperature=temperature)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _create_prompt(self, query: str, context: List[str]) -> str:
        """Create a prompt combining the context and query."""
        context_str = "\n".join(context)
        return f"""You are a helpful assistant that provides accurate answers based only on the given context.
        Rules:
        1. Never include any internal thinking, analysis, or <think> tags
        2. If information is not in the context, say so clearly
        3. Format responses in clear, structured markdown
        4. Be direct and concise
        
        Context:
        {context_str}
        
        Question: {query}
        Answer: """
