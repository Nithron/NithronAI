"""
nAI Core Utilities
Text processing, extraction, and helper functions
"""

from .text import (
    normalize_text,
    chunk_text,
    chunk_text_semantic,
    extract_sentences,
    tokenize,
    highlight_matches,
)

from .extractors import (
    extract_text,
    extract_pdf,
    extract_markdown,
    extract_html,
    extract_text_file,
    get_file_hash,
)

from .logging import (
    setup_logging,
    get_logger,
    log_request,
)

__all__ = [
    # Text
    "normalize_text",
    "chunk_text",
    "chunk_text_semantic",
    "extract_sentences",
    "tokenize",
    "highlight_matches",
    # Extractors
    "extract_text",
    "extract_pdf",
    "extract_markdown",
    "extract_html",
    "extract_text_file",
    "get_file_hash",
    # Logging
    "setup_logging",
    "get_logger",
    "log_request",
]

