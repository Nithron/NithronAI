"""
nAI Core Services Tests
Test the business logic layer
"""

import os
import tempfile
import pytest

from app.utils.text import (
    normalize_text,
    chunk_text,
    chunk_text_semantic,
    tokenize,
    highlight_matches,
)
from app.utils.extractors import (
    extract_text,
    extract_markdown,
    get_file_hash,
)


class TestTextUtils:
    """Test text processing utilities."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        text = "Hello\r\nWorld\r\n"
        result = normalize_text(text)
        assert result == "Hello\nWorld"
    
    def test_normalize_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\n\n\nTest"
        result = normalize_text(text)
        assert "    " not in result
        assert "\n\n\n" not in result
    
    def test_normalize_text_empty(self):
        """Test empty text normalization."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "A" * 500 + " " + "B" * 500 + " " + "C" * 500
        chunks = chunk_text(text, max_chars=600, overlap=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 700  # Some tolerance
    
    def test_chunk_text_short(self):
        """Test chunking short text."""
        text = "Short text"
        chunks = chunk_text(text, max_chars=1000, overlap=100)
        
        assert len(chunks) == 1
        assert chunks[0] == "Short text"
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        assert chunk_text("", max_chars=1000, overlap=100) == []
    
    def test_chunk_text_semantic(self):
        """Test semantic chunking."""
        text = "First sentence. Second sentence. Third sentence. " * 20
        chunks = chunk_text_semantic(text, max_chars=200, min_chars=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            # Check that chunks don't end mid-sentence (mostly)
            assert chunk.strip()
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello, World! This is a test."
        tokens = tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert all(t == t.lower() for t in tokens)
    
    def test_tokenize_special_chars(self):
        """Test tokenization with special characters."""
        text = "user-agent HTTP/1.1 status_code"
        tokens = tokenize(text)
        
        assert "user-agent" in tokens
        assert "status_code" in tokens
    
    def test_highlight_matches(self):
        """Test match highlighting."""
        text = "This is a long text about artificial intelligence and machine learning."
        query = "artificial intelligence"
        
        result = highlight_matches(text, query, max_length=100)
        
        assert "artificial" in result.lower() or "intelligence" in result.lower()
        assert len(result) <= 110  # Some tolerance for ellipsis


class TestExtractors:
    """Test text extraction utilities."""
    
    def test_extract_markdown(self):
        """Test markdown extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Title\n\n## Section\n\nContent here.\n\n```python\ncode\n```")
            f.flush()
            
            text, meta = extract_markdown(f.name)
            
            assert "Title" in text
            assert "Content here" in text
            assert meta["format"] == "markdown"
            assert meta.get("sections", 0) > 0
        
        os.unlink(f.name)
    
    def test_extract_text_file(self):
        """Test plain text extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Plain text content.\nMultiple lines.\n")
            f.flush()
            
            text, meta = extract_text(f.name)
            
            assert "Plain text content" in text
            assert meta["extension"] == ".txt"
        
        os.unlink(f.name)
    
    def test_file_hash(self):
        """Test file hash generation."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"Test content")
            f.flush()
            
            hash1 = get_file_hash(f.name)
            hash2 = get_file_hash(f.name)
            
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
        
        os.unlink(f.name)
    
    def test_file_hash_different(self):
        """Test that different files have different hashes."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f1:
            f1.write(b"Content 1")
            f1.flush()
            hash1 = get_file_hash(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f2:
            f2.write(b"Content 2")
            f2.flush()
            hash2 = get_file_hash(f2.name)
        
        assert hash1 != hash2
        
        os.unlink(f1.name)
        os.unlink(f2.name)


class TestRetrieverService:
    """Test retriever service."""
    
    def test_bm25_search(self):
        """Test BM25 search."""
        from app.services.retriever import RetrieverService
        from app.config import get_settings
        
        retriever = RetrieverService(get_settings())
        
        records = [
            {"doc_id": "1", "doc_path": "a.txt", "chunk_id": 0, "text": "Machine learning is great"},
            {"doc_id": "2", "doc_path": "b.txt", "chunk_id": 0, "text": "Deep learning uses neural networks"},
            {"doc_id": "3", "doc_path": "c.txt", "chunk_id": 0, "text": "Python is a programming language"},
        ]
        
        results = retriever.search_bm25("machine learning", records, top_k=2)
        
        assert len(results) <= 2
        # ML related docs should rank higher
        assert any("machine" in r.text.lower() or "learning" in r.text.lower() for r in results)
    
    def test_bm25_empty_index(self):
        """Test BM25 with empty index."""
        from app.services.retriever import RetrieverService
        from app.config import get_settings
        
        retriever = RetrieverService(get_settings())
        results = retriever.search_bm25("test query", [], top_k=5)
        
        assert results == []

