"""
nAI Toolpacks - PDF Pack
========================

PDF extraction with OCR support for scanned documents.
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, BinaryIO
from pathlib import Path


@dataclass
class PDFPage:
    """A single page from a PDF."""
    page_number: int
    text: str
    has_images: bool = False
    image_count: int = 0
    tables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFDocument:
    """Extracted PDF document."""
    pages: List[PDFPage]
    total_pages: int
    metadata: Dict[str, Any]
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.text}"
            for p in self.pages
            if p.text.strip()
        )
    
    @property
    def has_ocr_content(self) -> bool:
        """Check if document contains OCR'd content."""
        return any(p.metadata.get("ocr_applied", False) for p in self.pages)


class PDFExtractor:
    """
    Extract text and structure from PDF documents.
    
    Supports:
    - Native text extraction (pypdf)
    - OCR for scanned documents (pytesseract)
    - Table extraction (basic)
    - Image extraction
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_language: str = "eng",
        ocr_dpi: int = 300,
        extract_images: bool = False,
        extract_tables: bool = True
    ):
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.extract_images = extract_images
        self.extract_tables = extract_tables
    
    def _check_ocr_available(self) -> bool:
        """Check if OCR dependencies are available."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            return False
    
    def _extract_text_native(self, pdf_path: str) -> PDFDocument:
        """Extract text using pypdf (native text)."""
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        pages = []
        
        # Extract document metadata
        doc_metadata = {}
        if reader.metadata:
            if reader.metadata.title:
                doc_metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                doc_metadata["author"] = reader.metadata.author
            if reader.metadata.subject:
                doc_metadata["subject"] = reader.metadata.subject
            if reader.metadata.creator:
                doc_metadata["creator"] = reader.metadata.creator
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            
            # Check for images
            image_count = 0
            if hasattr(page, 'images'):
                try:
                    image_count = len(page.images)
                except Exception:
                    pass
            
            pages.append(PDFPage(
                page_number=i + 1,
                text=text.strip(),
                has_images=image_count > 0,
                image_count=image_count,
                metadata={"native_extraction": True},
            ))
        
        return PDFDocument(
            pages=pages,
            total_pages=len(reader.pages),
            metadata=doc_metadata,
        )
    
    def _ocr_page(self, image) -> str:
        """Perform OCR on a single page image."""
        import pytesseract
        
        # Preprocess image for better OCR
        try:
            from PIL import ImageFilter, ImageEnhance
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
        except Exception:
            pass
        
        # Run OCR
        text = pytesseract.image_to_string(
            image,
            lang=self.ocr_language,
            config='--psm 1'  # Automatic page segmentation with OSD
        )
        
        return text.strip()
    
    def _extract_with_ocr(self, pdf_path: str) -> PDFDocument:
        """Extract text using OCR."""
        from pdf2image import convert_from_path
        
        # First get native extraction for metadata
        native_doc = self._extract_text_native(pdf_path)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.ocr_dpi)
        
        pages = []
        for i, image in enumerate(images):
            native_text = native_doc.pages[i].text if i < len(native_doc.pages) else ""
            
            # Decide whether to use OCR
            # Use OCR if native text is very short (likely scanned)
            needs_ocr = len(native_text.strip()) < 100
            
            if needs_ocr and self.enable_ocr:
                ocr_text = self._ocr_page(image)
                text = ocr_text if len(ocr_text) > len(native_text) else native_text
                ocr_applied = len(ocr_text) > len(native_text)
            else:
                text = native_text
                ocr_applied = False
            
            pages.append(PDFPage(
                page_number=i + 1,
                text=text,
                has_images=native_doc.pages[i].has_images if i < len(native_doc.pages) else True,
                metadata={
                    "ocr_applied": ocr_applied,
                    "native_text_length": len(native_text),
                },
            ))
        
        return PDFDocument(
            pages=pages,
            total_pages=len(images),
            metadata=native_doc.metadata,
        )
    
    def extract(self, pdf_path: str) -> PDFDocument:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            PDFDocument with extracted content
        """
        # Start with native extraction
        doc = self._extract_text_native(pdf_path)
        
        # Check if OCR is needed (document has little native text)
        total_native_text = sum(len(p.text) for p in doc.pages)
        avg_chars_per_page = total_native_text / max(1, len(doc.pages))
        
        # If average is less than 100 chars per page, try OCR
        if avg_chars_per_page < 100 and self.enable_ocr and self._check_ocr_available():
            try:
                doc = self._extract_with_ocr(pdf_path)
            except Exception as e:
                # Fall back to native extraction
                doc.metadata["ocr_error"] = str(e)
        
        return doc
    
    def extract_from_bytes(self, pdf_bytes: bytes) -> PDFDocument:
        """Extract from bytes (e.g., uploaded file)."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        try:
            return self.extract(tmp_path)
        finally:
            os.unlink(tmp_path)


def extract_pdf_with_ocr(
    pdf_path: str,
    enable_ocr: bool = True,
    language: str = "eng"
) -> str:
    """
    Convenience function to extract text from PDF.
    
    Args:
        pdf_path: Path to PDF file
        enable_ocr: Enable OCR for scanned documents
        language: OCR language code
    
    Returns:
        Extracted text as string
    """
    extractor = PDFExtractor(enable_ocr=enable_ocr, ocr_language=language)
    doc = extractor.extract(pdf_path)
    return doc.full_text

