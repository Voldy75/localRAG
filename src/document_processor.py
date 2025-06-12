from typing import List, Dict
import os
import logging
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.supported_formats = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self.supported_formats.keys())
    
    def load_and_split_document(self, file_path: str) -> List[Document]:
        """Load and split a document into chunks."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Load document using appropriate loader
        loader_class = self.supported_formats[file_extension]
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Clean and prepare the documents
        cleaned_documents = []
        for doc in documents:
            # Ensure we have a string content
            if not isinstance(doc.page_content, str):
                continue
                
            # Clean up the content
            content = doc.page_content.strip()
            if not content:
                continue
                
            # Add metadata
            doc.metadata.update({
                "source": os.path.basename(file_path),  # Use just the filename for cleaner output
                "file_type": file_extension,
                "processed_at": datetime.utcnow().isoformat(),
                "file_name": os.path.basename(file_path),
                "chunk_id": len(cleaned_documents)  # Add chunk ID for reference
            })
            cleaned_documents.append(doc)
        
        # Split documents if we have any clean documents
        if not cleaned_documents:
            return []
            
        split_docs = self.text_splitter.split_documents(cleaned_documents)
        return split_docs

    def create_embeddings(self, texts: List[str]):
        """Create embeddings for a list of texts."""
        return self.embeddings.embed_documents(texts)

    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file format is supported."""
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_formats

    def process_document(self, file_path: str) -> List[Dict]:
        """Process a single document and return its content and metadata."""
        try:
            # Check if file exists and is supported
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not self.is_supported_file(file_path):
                raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")

            # Load and split the document
            documents = self.load_and_split_document(file_path)
            if not documents:
                return None

            # Convert Document objects to dictionaries
            processed_docs = []
            for doc in documents:
                processed_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

            return processed_docs
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {str(e)}")
            return None
