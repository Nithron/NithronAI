"""
nAI Core Test Configuration
Pytest fixtures and configuration
"""

import os
import sys
import tempfile
import pytest
from typing import Generator

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "index"), exist_ok=True)
        yield tmpdir


@pytest.fixture(scope="session")
def app(test_data_dir):
    """Create test application."""
    # Set test environment
    os.environ["NAI_DEBUG"] = "true"
    os.environ["NAI_DATA_DIR"] = test_data_dir
    os.environ["NAI_LLM_ENABLED"] = "false"
    os.environ["NAI_EMBEDDINGS_ENABLED"] = "false"
    os.environ["NAI_AUTH_ENABLED"] = "false"
    os.environ["NAI_RATE_LIMIT_ENABLED"] = "false"
    
    # Clear cached settings
    from app.config import get_settings
    get_settings.cache_clear()
    
    from app.main import create_app
    return create_app()


@pytest.fixture
def client(app) -> Generator:
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_text_file(test_data_dir) -> str:
    """Create a sample text file for testing."""
    content = """
    This is a sample document for testing nAI.
    
    It contains multiple paragraphs with different topics.
    The first topic is about artificial intelligence and machine learning.
    
    The second paragraph discusses document retrieval and search algorithms.
    BM25 is a popular ranking function used in information retrieval.
    
    Finally, we talk about natural language processing and embeddings.
    Vector representations help capture semantic meaning of text.
    """
    
    filepath = os.path.join(test_data_dir, "sample_test.txt")
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create minimal PDF bytes for testing."""
    # Minimal valid PDF
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test PDF content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""

