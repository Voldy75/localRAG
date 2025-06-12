from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid
import uvicorn
import os
from datetime import datetime

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_service import LLMService

app = FastAPI(title="Local RAG System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore()
llm_service = LLMService()

# Constants
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "deepseek-r1"
    temperature: Optional[float] = 0.7

class OpenAIChoice(BaseModel):
    message: Dict[str, str]
    finish_reason: str = "stop"
    index: int = 0

class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Dict[str, int]

class Query(BaseModel):
    question: str
    context_window: Optional[int] = 2000
    temperature: Optional[float] = 0.7

class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    processed_at: Optional[str]
    size_bytes: int

class CompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all available documents in the documents directory."""
    try:
        documents = []
        for filename in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in document_processor.get_supported_formats():
                    stat = os.stat(file_path)
                    doc_info = DocumentInfo(
                        filename=filename,
                        file_type=ext.lower(),
                        processed_at=None,  # We could store this in vector store metadata
                        size_bytes=stat.st_size
                    )
                    documents.append(doc_info)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Check file extension
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in document_processor.get_supported_formats():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(document_processor.get_supported_formats())}"
            )

        # Create documents directory if it doesn't exist
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        # Save the uploaded file to documents directory
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process the document
        docs = document_processor.process_document(file_path)
        if not docs:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document {file.filename}"
            )

        # Create embeddings for each document
        for doc in docs:
            embedding = document_processor.create_embeddings([doc['content']])[0]
            doc['embedding'] = embedding
            
        # Store in vector database
        vector_store.add_documents(docs)
        
        return {
            "message": f"Document {file.filename} processed and stored successfully",
            "chunks": len(docs)
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-all-documents")
async def process_all_documents():
    """Process all documents in the documents directory."""
    try:
        # Get all supported files in the documents directory
        processed_files = []
        failed_files = []
        
        for filename in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            if not os.path.isfile(file_path):
                continue
                
            _, ext = os.path.splitext(filename)
            if ext.lower() not in document_processor.get_supported_formats():
                continue
                
            try:
                # Process the document
                documents = document_processor.load_and_split_document(file_path)
                embeddings = document_processor.create_embeddings([doc.page_content for doc in documents])
                
                # Generate unique IDs for the documents
                ids = [str(uuid.uuid4()) for _ in documents]
                
                # Store in vector database
                vector_store.add_documents(documents, embeddings, ids)
                processed_files.append(filename)
            except Exception as e:
                failed_files.append({"file": filename, "error": str(e)})
        
        return {
            "message": f"Processed {len(processed_files)} documents successfully",
            "processed_files": processed_files,
            "failed_files": failed_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query):
    """Query the RAG system."""
    try:
        # Create embedding for the query
        query_embedding = document_processor.create_embeddings([query.question])[0]
        
        # Search for relevant documents
        results = vector_store.search(query_embedding)
        
        if not results["documents"]:
            return {
                "response": "I don't have enough context to answer your question. Please try uploading relevant documents first."
            }
        
        # Generate response using LLM
        response = llm_service.generate_response(
            query.question,
            results["documents"][0],
            temperature=query.temperature
        )
        
        return {
            "response": response,
            "sources": [doc.get("metadata", {}).get("source") for doc in results["documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest):
    """OpenAI-compatible chat completion endpoint for RAG integration."""
    try:
        # Extract the last user message
        user_message = next(
            (msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"),
            None
        )
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Create embedding for the query
        query_embedding = document_processor.create_embeddings([user_message])[0]
        
        # Search for relevant documents
        results = vector_store.search(query_embedding)
        
        if not results["documents"]:
            response_text = "I don't have enough context to answer your question. Please try uploading relevant documents first."
        else:
            # Generate response using LLM with all relevant context
            context = []
            sources = set()
            
            for doc in results["documents"]:
                # Extract content and metadata
                content = doc["content"]
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "unknown source")
                similarity = results["distances"][results["documents"].index(doc)]
                
                # Only include content above similarity threshold
                if similarity >= 0.3:
                    context.append(f"Content from {source}:\n{content}")
                    sources.add(source)
            
            if not context:
                response_text = "I cannot find relevant information in the provided documents to answer your question."
            else:
                response_text = llm_service.generate_response(
                    user_message,
                    context,
                    temperature=request.temperature or 0.7
                )
                
                # Add source citations
                if sources:
                    response_text += f"\n\nSources: {', '.join(sources)}"
        
        # Add source citations
        sources = [doc.get("metadata", {}).get("source") for doc in results["documents"]]
        if sources:
            response_text += f"\n\nSources: {', '.join(sources)}"
        
        return OpenAIResponse(
            id=str(uuid.uuid4()),
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop",
                "index": 0
            }],
            usage={"total_tokens": 0}  # Placeholder as we don't track token usage
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
