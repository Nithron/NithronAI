"""
nAI Core Text Extractors
Extract text from various file formats
"""

import hashlib
import re
from typing import Tuple, Dict, Any
from pathlib import Path


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_pdf(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF file.
    Returns (text, metadata).
    """
    from pypdf import PdfReader
    
    metadata: Dict[str, Any] = {"pages": 0, "format": "pdf"}
    
    try:
        reader = PdfReader(file_path)
        metadata["pages"] = len(reader.pages)
        
        # Extract document metadata if available
        if reader.metadata:
            if reader.metadata.title:
                metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                metadata["author"] = reader.metadata.author
            if reader.metadata.subject:
                metadata["subject"] = reader.metadata.subject
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[Page {i + 1}]\n{text}")
        
        return "\n\n".join(pages_text), metadata
    
    except Exception as e:
        metadata["error"] = str(e)
        return "", metadata


def extract_markdown(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from Markdown file.
    Preserves structure for better chunking.
    """
    metadata: Dict[str, Any] = {"format": "markdown"}
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Extract title from first H1 if present
        h1_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            metadata["title"] = h1_match.group(1).strip()
        
        # Count headers
        headers = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
        metadata["sections"] = len(headers)
        
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        metadata["code_blocks"] = len(code_blocks)
        
        return content, metadata
    
    except Exception as e:
        metadata["error"] = str(e)
        return "", metadata


def extract_html(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from HTML file.
    Strips tags and extracts readable content.
    """
    metadata: Dict[str, Any] = {"format": "html"}
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            metadata["title"] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
        
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', content, flags=re.IGNORECASE)
        
        # Remove HTML comments
        content = re.sub(r'<!--[\s\S]*?-->', '', content)
        
        # Replace block elements with newlines
        content = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', content, flags=re.IGNORECASE)
        
        # Remove remaining tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode HTML entities
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        
        # Clean up whitespace
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip(), metadata
    
    except Exception as e:
        metadata["error"] = str(e)
        return "", metadata


def extract_text_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from plain text file.
    """
    metadata: Dict[str, Any] = {"format": "text"}
    
    try:
        # Try UTF-8 first, then fall back to other encodings
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        content = ""
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                metadata["encoding"] = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            metadata["encoding"] = "utf-8 (lossy)"
        
        # Count lines
        metadata["lines"] = content.count('\n') + 1
        
        return content, metadata
    
    except Exception as e:
        metadata["error"] = str(e)
        return "", metadata


def extract_text(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from file based on extension.
    Returns (text, metadata).
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    # Get file stats
    stat = path.stat()
    base_metadata = {
        "filename": path.name,
        "extension": ext,
        "size_bytes": stat.st_size,
    }
    
    # Extract based on type
    if ext == ".pdf":
        text, meta = extract_pdf(file_path)
    elif ext in (".md", ".markdown"):
        text, meta = extract_markdown(file_path)
    elif ext in (".html", ".htm"):
        text, meta = extract_html(file_path)
    elif ext in (".txt", ".text", ".rst", ".log", ".csv", ".json", ".yaml", ".yml", ".xml"):
        text, meta = extract_text_file(file_path)
    else:
        # Try as plain text
        text, meta = extract_text_file(file_path)
        meta["format"] = "unknown"
    
    # Merge metadata
    base_metadata.update(meta)
    base_metadata["char_count"] = len(text)
    
    return text, base_metadata

