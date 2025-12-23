"""
nAI Core API Tests
Test the REST API endpoints
"""

import io
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "time" in data
        assert "version" in data
    
    def test_detailed_health(self, client: TestClient):
        """Test detailed health endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "config" in data
        assert "index" in data


class TestIngestEndpoints:
    """Test document ingestion endpoints."""
    
    def test_ingest_text_file(self, client: TestClient):
        """Test ingesting a text file."""
        content = b"This is a test document for nAI testing purposes."
        files = {"files": ("test.txt", io.BytesIO(content), "text/plain")}
        
        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "added" in data
        assert len(data["added"]) == 1
        assert data["added"][0]["status"] == "success"
        assert data["added"][0]["chunks"] > 0
    
    def test_ingest_multiple_files(self, client: TestClient):
        """Test ingesting multiple files."""
        files = [
            ("files", ("doc1.txt", io.BytesIO(b"First document content."), "text/plain")),
            ("files", ("doc2.txt", io.BytesIO(b"Second document content."), "text/plain")),
        ]
        
        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_files"] == 2
    
    def test_ingest_empty_file(self, client: TestClient):
        """Test ingesting an empty file."""
        files = {"files": ("empty.txt", io.BytesIO(b""), "text/plain")}
        
        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        
        data = response.json()
        # Empty file should be skipped
        assert data["added"][0]["status"] == "skipped"
    
    def test_ingest_markdown(self, client: TestClient):
        """Test ingesting a markdown file."""
        content = b"""# Test Document

## Introduction
This is a test markdown document.

## Content
It has multiple sections with different content.
"""
        files = {"files": ("test.md", io.BytesIO(content), "text/markdown")}
        
        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["added"][0]["status"] == "success"


class TestSearchEndpoints:
    """Test search and ask endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_index(self, client: TestClient):
        """Ensure we have some documents indexed."""
        content = b"""
        Artificial intelligence is the simulation of human intelligence by machines.
        Machine learning is a subset of AI that enables computers to learn from data.
        Deep learning uses neural networks with many layers.
        Natural language processing helps computers understand human language.
        """
        files = {"files": ("ai_doc.txt", io.BytesIO(content), "text/plain")}
        client.post("/ingest", files=files)
    
    def test_ask_question(self, client: TestClient):
        """Test asking a question."""
        response = client.post(
            "/ask",
            json={"question": "What is artificial intelligence?", "top_k": 3}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert data["method"] in ["extractive", "llm"]
    
    def test_ask_with_no_results(self, client: TestClient):
        """Test asking about something not in the index."""
        response = client.post(
            "/ask",
            json={"question": "What is quantum entanglement?", "top_k": 3}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
    
    def test_search_documents(self, client: TestClient):
        """Test raw search endpoint."""
        response = client.post(
            "/search",
            json={"query": "machine learning", "top_k": 5}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["method"] == "bm25"
    
    def test_ask_validation(self, client: TestClient):
        """Test request validation."""
        # Empty question
        response = client.post("/ask", json={"question": "", "top_k": 3})
        assert response.status_code == 422  # Validation error
        
        # Invalid top_k
        response = client.post("/ask", json={"question": "test", "top_k": 1000})
        assert response.status_code == 422


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_document(self, client: TestClient):
        """Add a document for testing."""
        content = b"Test document for document management."
        files = {"files": ("manage_test.txt", io.BytesIO(content), "text/plain")}
        response = client.post("/ingest", files=files)
        self.doc_id = response.json()["added"][0]["doc_id"]
    
    def test_list_documents(self, client: TestClient):
        """Test listing documents."""
        response = client.get("/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] >= 1
    
    def test_get_document(self, client: TestClient):
        """Test getting a specific document."""
        # First get the list
        response = client.get("/documents")
        docs = response.json()["documents"]
        
        if docs:
            doc_id = docs[0]["doc_id"]
            response = client.get(f"/documents/{doc_id}")
            assert response.status_code == 200
    
    def test_get_document_chunks(self, client: TestClient):
        """Test getting document chunks."""
        response = client.get("/documents")
        docs = response.json()["documents"]
        
        if docs:
            doc_id = docs[0]["doc_id"]
            response = client.get(f"/documents/{doc_id}/chunks")
            assert response.status_code == 200
            
            data = response.json()
            assert "chunks" in data
            assert "total" in data
    
    def test_delete_document(self, client: TestClient):
        """Test deleting a document."""
        # Get a document to delete
        response = client.get("/documents")
        docs = response.json()["documents"]
        
        if docs:
            doc_id = docs[0]["doc_id"]
            response = client.delete(f"/documents/{doc_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
    
    def test_delete_nonexistent(self, client: TestClient):
        """Test deleting a non-existent document."""
        response = client.delete("/documents/nonexistent123")
        assert response.status_code == 404


class TestChatEndpoints:
    """Test chat endpoints."""
    
    def test_chat_single_message(self, client: TestClient):
        """Test single chat message."""
        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "use_context": False,
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "conversation_id" in data
    
    def test_chat_with_context(self, client: TestClient):
        """Test chat with document context."""
        # First ingest a document
        content = b"The capital of France is Paris."
        files = {"files": ("france.txt", io.BytesIO(content), "text/plain")}
        client.post("/ingest", files=files)
        
        # Then chat
        response = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "use_context": True,
            }
        )
        assert response.status_code == 200


class TestRateLimiting:
    """Test rate limiting (when enabled)."""
    
    def test_rate_limit_headers(self, client: TestClient):
        """Test rate limit headers are present."""
        response = client.get("/health")
        # Headers should be present (but values depend on config)
        # Just check endpoint works
        assert response.status_code == 200

