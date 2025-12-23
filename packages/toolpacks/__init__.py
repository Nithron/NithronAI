"""
nAI Toolpacks
=============

Specialized extractors and processors for different content types.

Toolpacks:
- PDFPack: PDF extraction with OCR support
- WebPack: Web scraping and URL ingestion
- EmailPack: Email parsing (IMAP, MBOX, EML)
- CodePack: Code file parsing with syntax awareness
"""

__version__ = "0.1.0"

from .pdf_pack import PDFExtractor, PDFPage, extract_pdf_with_ocr
from .web_pack import WebScraper, URLContent, scrape_url
from .email_pack import EmailParser, ParsedEmail, parse_email_file
from .code_pack import CodeExtractor, CodeFile, extract_code_structure

__all__ = [
    # PDF
    "PDFExtractor",
    "PDFPage",
    "extract_pdf_with_ocr",
    # Web
    "WebScraper",
    "URLContent",
    "scrape_url",
    # Email
    "EmailParser",
    "ParsedEmail",
    "parse_email_file",
    # Code
    "CodeExtractor",
    "CodeFile",
    "extract_code_structure",
]

