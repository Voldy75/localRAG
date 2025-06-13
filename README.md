# Local RAG System with Ollama

This project implements a local RAG (Retrieval-Augmented Generation) system using Ollama for LLM integration, vector storage, and FastAPI for the API interface. The system allows you to upload documents and ask questions about them, with the LLM providing contextual answers based on the document content.

## Testing

To run the tests:
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

The coverage report will be generated in the `coverage_report` directory.

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

Once documents are processed, Run the required curl command for executing query from terminal: 
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is BBPS and what are its key features?", "temperature": 0.7}'
```
OR

Alternatively, you can also leverage UI & use the OpenWebUI interface to query your documents:

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

## Testing Requirements

The system includes a comprehensive test suite to ensure all components work correctly. Before making any changes to the system, all tests must pass.

### Test Structure

1. Unit Tests:
   - `test_document_processor.py`: Tests document processing, embedding creation, and document splitting
   - `test_vector_store.py`: Tests vector storage, similarity search, and persistence
   - `test_llm_service.py`: Tests LLM integration, response generation, and error handling
   - `test_api.py`: Tests all API endpoints and error handling

2. Integration Tests:
   - `test_integration.py`: Tests full RAG pipeline, multiple document types, and error recovery

### Running Tests

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Run the test suite:
```bash
./run_tests.sh
```

The test runner will:
- Execute all unit and integration tests
- Generate a coverage report
- Display results in the terminal
- Create an HTML coverage report in `coverage_report/`

### Test Coverage

The test suite verifies:
1. Document Processing:
   - File type validation
   - Content extraction
   - Document splitting
   - Embedding generation

2. Vector Storage:
   - Document indexing
   - Similarity search
   - Persistence
   - Threshold filtering

3. LLM Integration:
   - Response generation
   - Context handling
   - Temperature settings
   - Error handling

4. API Endpoints:
   - Document upload
   - Document processing
   - Query handling
   - Error responses

5. Integration:
   - Full RAG pipeline
   - Multi-document handling
   - System recovery
   - Large document processing

### Required Test Pass Criteria

Before deploying any changes:
1. All unit tests must pass
2. All integration tests must pass
3. Code coverage should meet minimum thresholds:
   - Overall coverage: 80%
   - Critical components (document_processor.py, vector_store.py): 90%
   - API endpoints: 100%

### Adding New Tests

When adding new features:
1. Create corresponding unit tests
2. Update integration tests if needed
3. Verify coverage requirements are met
4. Run the full test suite before committing changes
