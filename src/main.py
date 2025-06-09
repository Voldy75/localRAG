from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
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

        # Save the uploaded file to documents directory
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process the document
        documents = document_processor.load_and_split_document(file_path)
        embeddings = document_processor.create_embeddings([doc.page_content for doc in documents])
        
        # Generate unique IDs for the documents
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Store in vector database
        vector_store.add_documents(documents, embeddings, ids)
        
        return {
            "message": f"Document {file.filename} processed and stored successfully",
            "chunks": len(documents)
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
async def chat_completion(request: CompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
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
            # Generate response using LLM
            response_text = llm_service.generate_response(
                user_message,
                results["documents"][0],
                temperature=request.temperature
            )
        
        # Format response in OpenAI-compatible format
        source_docs = [doc.get("metadata", {}).get("source") for doc in results["documents"]]
        sources = f"\n\nSources: {', '.join(source_docs)}" if source_docs else ""
        
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text + sources
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
