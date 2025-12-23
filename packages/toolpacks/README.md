# nAI Toolpacks

Specialized content extractors for different file types and sources.

## Features

- **PDF Pack**: PDF extraction with OCR support for scanned documents
- **Web Pack**: Web scraping with JavaScript rendering
- **Email Pack**: Parse EML, MBOX files and IMAP connections
- **Code Pack**: Code analysis with syntax awareness

## Installation

```bash
pip install nai-toolpacks

# With PDF support (requires Tesseract for OCR)
pip install nai-toolpacks[pdf]

# With web scraping
pip install nai-toolpacks[web]

# Everything
pip install nai-toolpacks[all]
```

For OCR support, install Tesseract:
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### PDF Extraction

```python
from toolpacks import PDFExtractor, extract_pdf_with_ocr

# Full extraction with metadata
extractor = PDFExtractor(enable_ocr=True, ocr_language="eng")
doc = extractor.extract("document.pdf")

print(f"Pages: {doc.total_pages}")
print(f"Has OCR: {doc.has_ocr_content}")
print(doc.full_text)

# Quick extraction
text = extract_pdf_with_ocr("scanned.pdf")
```

### Web Scraping

```python
from toolpacks import WebScraper, scrape_url

# Full scraping with metadata
scraper = WebScraper(
    timeout=30,
    extract_links=True,
    extract_images=True
)
content = scraper.scrape("https://example.com")

print(f"Title: {content.title}")
print(f"Words: {content.word_count}")
print(content.text)

# With JavaScript rendering
content = scraper.scrape_with_js("https://spa-app.com")

# Quick extraction
text = scrape_url("https://example.com")
```

### Email Parsing

```python
from toolpacks import EmailParser, parse_email_file

# Parse single email
parser = EmailParser(extract_attachments=True)
email = parser.parse_eml("message.eml")

print(f"From: {email.from_address}")
print(f"Subject: {email.subject}")
print(f"Attachments: {len(email.attachments)}")
print(email.body_text)

# Parse MBOX (multiple emails)
emails = parser.parse_mbox("mailbox.mbox")

# Fetch from IMAP
emails = parser.fetch_imap(
    host="imap.gmail.com",
    username="user@gmail.com",
    password="app-password",
    folder="INBOX",
    limit=100
)

# Quick extraction
text = parse_email_file("message.eml")
```

### Code Analysis

```python
from toolpacks import CodeExtractor, extract_code_structure

# Full analysis
extractor = CodeExtractor(extract_docstrings=True)
code = extractor.extract("main.py")

print(f"Language: {code.language}")
print(f"Lines: {code.metadata['lines']}")
print(f"Functions: {len([s for s in code.symbols if s.kind == 'function'])}")

for symbol in code.symbols:
    print(f"  {symbol.kind}: {symbol.signature}")

# Get summary
print(code.summary)

# Get searchable text
print(code.searchable_text)

# Quick summary
summary = extract_code_structure("main.py")
```

## Supported Formats

### PDF Pack
- Native PDF text extraction
- OCR for scanned documents
- Multi-page support
- Metadata extraction (title, author, etc.)

### Web Pack
- Static HTML pages
- JavaScript-rendered pages (Playwright)
- Link and image extraction
- Metadata (title, description, OG tags)

### Email Pack
- EML files
- MBOX files
- IMAP connections
- Attachments extraction

### Code Pack
| Language | Imports | Classes | Functions | Docstrings |
|----------|---------|---------|-----------|------------|
| Python | ✅ | ✅ | ✅ | ✅ |
| JavaScript/TypeScript | ✅ | ✅ | ✅ | - |
| Go | ✅ | ✅ | ✅ | - |
| Rust | ✅ | ✅ | ✅ | - |
| Java | ✅ | ✅ | ✅ | - |
| Others | - | - | Basic | - |

## License

AGPL-3.0-only
