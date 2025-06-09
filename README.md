# Local RAG System with Ollama

This project implements a local RAG (Retrieval-Augmented Generation) system using Ollama for LLM integration, vector storage, and FastAPI for the API interface. The system allows you to upload documents and ask questions about them, with the LLM providing contextual answers based on the document content.

## Prerequisites

### Required Software
- Docker and Docker Compose (for containerized setup)
- Git (for cloning the repository)

OR if running locally without Docker:
- Python 3.8+ 
- Ollama installed locally (https://ollama.ai)
- Virtual environment

## Project Structure

```
.
├── documents/         # Store your source documents here
├── src/              # Source code
│   ├── main.py       # FastAPI application
│   ├── document_processor.py
│   ├── vector_store.py
│   └── llm_service.py
└── requirements.txt
```

## Installation & Setup

### Option 1: Using Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd localRAG
```

2. Start the application using Docker Compose:
```bash
docker compose up -d
```

This will start three services:
- Ollama service (LLM backend) at http://localhost:11434
- Open WebUI (User interface) at http://localhost:3000
- RAG API (Document processing and querying) at http://localhost:8000

3. Wait for the services to initialize (this may take a few minutes on first run):
- The Ollama service will automatically download the required model (deepseek-r1)
- The RAG API will initialize its vector store
- The Web UI will become available

### Option 2: Local Setup (Without Docker)

1. Clone the repository and navigate to it:
```bash
git clone <repository-url>
cd localRAG
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
- Download from https://ollama.ai
- Install and start the Ollama service
- Pull the required model:
```bash
ollama pull deepseek-r1
```

5. Start the FastAPI server:
```bash
python src/main.py
```

The server will start at http://localhost:8000

## Using the Application

### 1. Document Management

The system supports various document formats:
- Text files (.txt)
- PDF files (.pdf)
- Word documents (.docx)

You have two ways to add documents:

1. Place documents directly in the `documents` folder:
   - If using Docker: Place them in the `documents` folder before starting the containers
   - If running locally: Place them in the `documents` folder at any time

2. Use the upload endpoint to add documents one at a time (see API usage below)

### 2. Processing Documents

After adding documents, you need to process them to make them available for querying:

1. Process all documents at once:
```bash
curl -X POST http://localhost:8000/process-all-documents
```

2. Or process a single uploaded document:
```bash
curl -X POST -F "file=@path/to/your/document.txt" http://localhost:8000/upload
```

### 3. Querying the System

Call the required command: 
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is BBPS and what are its key features?", "temperature": 0.7}'

Once your documents are processed, you can use the OpenWebUI interface to query your documents:

1. Open http://localhost:3000 in your browser
2. Login with default credentials (if prompted)
3. Configure the Model:
   - Click on "Settings" in the sidebar
   - Select "deepseek-r1" from the model dropdown
   - Enable the "RAG" toggle
   
4. Start Querying:
   - Type your question in the chat interface
   - The system will automatically:
     - Search through your processed documents
     - Find relevant context
     - Generate answers based on your documents
   
5. View Sources:
   - Each response will include references to the source documents
   - Click on the source to see the relevant document section

Note: The traditional API endpoints are still available for programmatic access, but the OpenWebUI interface provides a more user-friendly experience.
