"""
nAI Toolpacks - Code Pack
=========================

Code file parsing with syntax awareness.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from pathlib import Path


@dataclass
class CodeSymbol:
    """A code symbol (function, class, etc.)."""
    name: str
    kind: str  # function, class, method, variable, etc.
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    parent: Optional[str] = None


@dataclass
class CodeFile:
    """Parsed code file."""
    path: str
    language: str
    content: str
    symbols: List[CodeSymbol]
    imports: List[str]
    metadata: Dict[str, Any]
    
    @property
    def summary(self) -> str:
        """Generate a summary of the code file."""
        parts = [f"# {Path(self.path).name} ({self.language})"]
        
        if self.imports:
            parts.append(f"\n## Imports\n- " + "\n- ".join(self.imports[:10]))
            if len(self.imports) > 10:
                parts.append(f"... and {len(self.imports) - 10} more")
        
        classes = [s for s in self.symbols if s.kind == 'class']
        if classes:
            parts.append("\n## Classes")
            for cls in classes:
                doc = f": {cls.docstring[:100]}..." if cls.docstring else ""
                parts.append(f"- `{cls.name}`{doc}")
        
        functions = [s for s in self.symbols if s.kind == 'function']
        if functions:
            parts.append("\n## Functions")
            for func in functions[:20]:
                parts.append(f"- `{func.signature}`")
            if len(functions) > 20:
                parts.append(f"... and {len(functions) - 20} more")
        
        return "\n".join(parts)
    
    @property
    def searchable_text(self) -> str:
        """Get searchable text representation."""
        parts = [
            f"File: {self.path}",
            f"Language: {self.language}",
            "",
        ]
        
        for symbol in self.symbols:
            parts.append(f"{symbol.kind}: {symbol.signature}")
            if symbol.docstring:
                parts.append(f"  {symbol.docstring}")
        
        parts.append("")
        parts.append(self.content)
        
        return "\n".join(parts)


class CodeExtractor:
    """
    Extract structure and content from code files.
    
    Supports:
    - Python
    - JavaScript/TypeScript
    - Go
    - Rust
    - Java
    - C/C++
    - And basic support for other languages
    """
    
    # Language detection by extension
    EXTENSION_MAP = {
        '.py': 'python',
        '.pyi': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.R': 'r',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.md': 'markdown',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
    }
    
    def __init__(
        self,
        extract_docstrings: bool = True,
        max_file_size: int = 1024 * 1024  # 1MB
    ):
        self.extract_docstrings = extract_docstrings
        self.max_file_size = max_file_size
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(ext, 'unknown')
    
    def extract(self, file_path: str) -> CodeFile:
        """Extract structure from a code file."""
        path = Path(file_path)
        
        # Check file size
        if path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes")
        
        # Read content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = path.read_text(encoding='latin-1')
        
        language = self.detect_language(file_path)
        
        # Extract based on language
        if language == 'python':
            symbols, imports = self._extract_python(content)
        elif language in ('javascript', 'typescript'):
            symbols, imports = self._extract_javascript(content)
        elif language == 'go':
            symbols, imports = self._extract_go(content)
        elif language == 'rust':
            symbols, imports = self._extract_rust(content)
        elif language == 'java':
            symbols, imports = self._extract_java(content)
        else:
            symbols, imports = self._extract_generic(content)
        
        return CodeFile(
            path=str(file_path),
            language=language,
            content=content,
            symbols=symbols,
            imports=imports,
            metadata={
                "lines": content.count('\n') + 1,
                "size_bytes": len(content.encode('utf-8')),
            },
        )
    
    def _extract_python(self, content: str) -> tuple:
        """Extract Python code structure."""
        symbols = []
        imports = []
        lines = content.split('\n')
        
        # Imports
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)
        
        # Classes and functions using regex
        class_pattern = r'^class\s+(\w+)(?:\([^)]*\))?\s*:'
        func_pattern = r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*[^:]+)?\s*:'
        
        current_class = None
        
        for i, line in enumerate(lines):
            # Class detection
            class_match = re.match(class_pattern, line)
            if class_match:
                name = class_match.group(1)
                docstring = self._extract_python_docstring(lines, i + 1)
                
                # Find class end (simplified)
                end_line = self._find_block_end(lines, i)
                
                symbols.append(CodeSymbol(
                    name=name,
                    kind='class',
                    line_start=i + 1,
                    line_end=end_line,
                    signature=f"class {name}",
                    docstring=docstring,
                ))
                current_class = name
            
            # Function detection
            func_match = re.match(func_pattern, line)
            if func_match:
                indent = func_match.group(1)
                name = func_match.group(2)
                params = func_match.group(3)
                
                docstring = self._extract_python_docstring(lines, i + 1)
                
                is_method = len(indent) > 0
                kind = 'method' if is_method else 'function'
                parent = current_class if is_method else None
                
                symbols.append(CodeSymbol(
                    name=name,
                    kind=kind,
                    line_start=i + 1,
                    line_end=i + 1,  # Simplified
                    signature=f"def {name}({params})",
                    docstring=docstring,
                    parent=parent,
                ))
        
        return symbols, imports
    
    def _extract_python_docstring(self, lines: List[str], start: int) -> Optional[str]:
        """Extract Python docstring."""
        if start >= len(lines):
            return None
        
        line = lines[start].strip()
        
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            if line.endswith(quote) and len(line) > 6:
                return line[3:-3].strip()
            
            # Multi-line docstring
            docstring_lines = [line[3:]]
            for i in range(start + 1, min(start + 20, len(lines))):
                if quote in lines[i]:
                    idx = lines[i].index(quote)
                    docstring_lines.append(lines[i][:idx])
                    break
                docstring_lines.append(lines[i])
            
            return '\n'.join(docstring_lines).strip()
        
        return None
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find the end of an indented block."""
        if start >= len(lines):
            return start
        
        # Get the indentation of the block start
        start_indent = len(lines[start]) - len(lines[start].lstrip())
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            indent = len(line) - len(line.lstrip())
            if indent <= start_indent:
                return i
        
        return len(lines)
    
    def _extract_javascript(self, content: str) -> tuple:
        """Extract JavaScript/TypeScript structure."""
        symbols = []
        imports = []
        lines = content.split('\n')
        
        # Imports
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('require('):
                imports.append(stripped)
        
        # Functions and classes
        func_patterns = [
            r'function\s+(\w+)\s*\(([^)]*)\)',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>',
            r'(\w+)\s*:\s*(?:async\s*)?function\s*\(([^)]*)\)',
        ]
        
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    params = match.group(2) if len(match.groups()) > 1 else ''
                    symbols.append(CodeSymbol(
                        name=name,
                        kind='function',
                        line_start=i + 1,
                        line_end=i + 1,
                        signature=f"{name}({params})",
                    ))
                    break
            
            class_match = re.search(class_pattern, line)
            if class_match:
                symbols.append(CodeSymbol(
                    name=class_match.group(1),
                    kind='class',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"class {class_match.group(1)}",
                ))
        
        return symbols, imports
    
    def _extract_go(self, content: str) -> tuple:
        """Extract Go structure."""
        symbols = []
        imports = []
        lines = content.split('\n')
        
        # Imports
        in_import = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ('):
                in_import = True
            elif in_import:
                if stripped == ')':
                    in_import = False
                elif stripped:
                    imports.append(stripped.strip('"'))
            elif stripped.startswith('import '):
                imports.append(stripped.replace('import ', '').strip('"'))
        
        # Functions and types
        func_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)'
        type_pattern = r'type\s+(\w+)\s+(struct|interface)'
        
        for i, line in enumerate(lines):
            func_match = re.search(func_pattern, line)
            if func_match:
                symbols.append(CodeSymbol(
                    name=func_match.group(1),
                    kind='function',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"func {func_match.group(1)}({func_match.group(2)})",
                ))
            
            type_match = re.search(type_pattern, line)
            if type_match:
                symbols.append(CodeSymbol(
                    name=type_match.group(1),
                    kind=type_match.group(2),
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"type {type_match.group(1)} {type_match.group(2)}",
                ))
        
        return symbols, imports
    
    def _extract_rust(self, content: str) -> tuple:
        """Extract Rust structure."""
        symbols = []
        imports = []
        lines = content.split('\n')
        
        # Use statements
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('use '):
                imports.append(stripped)
        
        # Functions and structs
        func_pattern = r'(?:pub\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\(([^)]*)\)'
        struct_pattern = r'(?:pub\s+)?struct\s+(\w+)'
        impl_pattern = r'impl(?:<[^>]+>)?\s+(\w+)'
        
        for i, line in enumerate(lines):
            func_match = re.search(func_pattern, line)
            if func_match:
                symbols.append(CodeSymbol(
                    name=func_match.group(1),
                    kind='function',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"fn {func_match.group(1)}({func_match.group(2)})",
                ))
            
            struct_match = re.search(struct_pattern, line)
            if struct_match:
                symbols.append(CodeSymbol(
                    name=struct_match.group(1),
                    kind='struct',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"struct {struct_match.group(1)}",
                ))
        
        return symbols, imports
    
    def _extract_java(self, content: str) -> tuple:
        """Extract Java structure."""
        symbols = []
        imports = []
        lines = content.split('\n')
        
        # Imports
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(stripped.replace('import ', '').rstrip(';'))
        
        # Classes and methods
        class_pattern = r'(?:public\s+)?class\s+(\w+)'
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
        
        for i, line in enumerate(lines):
            class_match = re.search(class_pattern, line)
            if class_match:
                symbols.append(CodeSymbol(
                    name=class_match.group(1),
                    kind='class',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"class {class_match.group(1)}",
                ))
            
            method_match = re.search(method_pattern, line)
            if method_match:
                symbols.append(CodeSymbol(
                    name=method_match.group(1),
                    kind='method',
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"{method_match.group(1)}({method_match.group(2)})",
                ))
        
        return symbols, imports
    
    def _extract_generic(self, content: str) -> tuple:
        """Generic extraction for unknown languages."""
        symbols = []
        imports = []
        
        # Try to find function-like patterns
        func_patterns = [
            r'(?:function|func|def|fn)\s+(\w+)',
            r'class\s+(\w+)',
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    symbols.append(CodeSymbol(
                        name=match.group(1),
                        kind='symbol',
                        line_start=i + 1,
                        line_end=i + 1,
                        signature=line.strip()[:80],
                    ))
                    break
        
        return symbols, imports


def extract_code_structure(file_path: str) -> str:
    """
    Convenience function to extract code structure summary.
    
    Args:
        file_path: Path to code file
    
    Returns:
        Summary string
    """
    extractor = CodeExtractor()
    code_file = extractor.extract(file_path)
    return code_file.summary

