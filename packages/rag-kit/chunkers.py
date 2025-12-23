"""
nAI RAG Kit - Chunkers
======================

Text chunking strategies for document processing.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0


class BaseChunker(ABC):
    """Base class for all chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text.strip()


class CharacterChunker(BaseChunker):
    """
    Simple character-based chunker with overlap.
    
    Fast but may break mid-word or mid-sentence.
    Best for: Quick processing where semantic boundaries don't matter.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separator: str = " "
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        text = self._normalize_text(text)
        if not text:
            return []
        
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            
            # Try to break at separator
            if end < len(text):
                last_sep = text.rfind(self.separator, start + self.chunk_size // 2, end)
                if last_sep > start:
                    end = last_sep + len(self.separator)
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    metadata=metadata or {},
                    start_char=start,
                    end_char=end,
                ))
                index += 1
            
            start = end - self.chunk_overlap if end < len(text) else end
            if start <= chunks[-1].start_char if chunks else 0:
                start = end
        
        return chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-aware chunker that respects sentence boundaries.
    
    Uses NLTK for sentence tokenization.
    Best for: General-purpose chunking with better semantic coherence.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap_sentences: int = 1
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
        self._init_nltk()
    
    def _init_nltk(self):
        """Initialize NLTK sentence tokenizer."""
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            return re.split(r'(?<=[.!?])\s+', text)
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        text = self._normalize_text(text)
        if not text:
            return []
        
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return [Chunk(text=text, index=0, metadata=metadata or {})]
        
        chunks = []
        current_sentences: List[str] = []
        current_length = 0
        index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            # Handle very long sentences
            if sentence_len > self.max_chunk_size:
                if current_sentences:
                    chunks.append(Chunk(
                        text=" ".join(current_sentences),
                        index=index,
                        metadata=metadata or {},
                    ))
                    index += 1
                    current_sentences = []
                    current_length = 0
                
                # Split long sentence
                char_chunker = CharacterChunker(self.max_chunk_size, 50)
                for sub_chunk in char_chunker.chunk(sentence):
                    chunks.append(Chunk(
                        text=sub_chunk.text,
                        index=index,
                        metadata=metadata or {},
                    ))
                    index += 1
                continue
            
            # Check if adding this sentence exceeds limit
            if current_length + sentence_len + 1 > self.max_chunk_size and current_sentences:
                chunks.append(Chunk(
                    text=" ".join(current_sentences),
                    index=index,
                    metadata=metadata or {},
                ))
                index += 1
                
                # Overlap
                if self.overlap_sentences > 0 and len(current_sentences) >= self.overlap_sentences:
                    current_sentences = current_sentences[-self.overlap_sentences:]
                    current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
                else:
                    current_sentences = []
                    current_length = 0
            
            current_sentences.append(sentence)
            current_length += sentence_len + 1
        
        # Handle remaining sentences
        if current_sentences:
            final_text = " ".join(current_sentences)
            if len(final_text) >= self.min_chunk_size or not chunks:
                chunks.append(Chunk(
                    text=final_text,
                    index=index,
                    metadata=metadata or {},
                ))
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1] = Chunk(
                    text=chunks[-1].text + " " + final_text,
                    index=chunks[-1].index,
                    metadata=chunks[-1].metadata,
                )
        
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive chunker that tries multiple separators in order.
    
    Inspired by LangChain's RecursiveCharacterTextSplitter.
    Best for: Documents with clear structure (headers, paragraphs).
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator, keeping the separator."""
        if separator == "":
            return list(text)
        return text.split(separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into larger chunks."""
        merged = []
        current = []
        current_len = 0
        
        for split in splits:
            split_len = len(split) + len(separator)
            
            if current_len + split_len > self.chunk_size:
                if current:
                    merged.append(separator.join(current))
                current = [split]
                current_len = len(split)
            else:
                current.append(split)
                current_len += split_len
        
        if current:
            merged.append(separator.join(current))
        
        return merged
    
    def _recursive_split(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = self._split_by_separator(text, separator)
        
        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recursively split with next separator
                result.extend(self._recursive_split(split, remaining_separators))
        
        # Merge small chunks
        return self._merge_splits(result, separator)
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        text = self._normalize_text(text)
        if not text:
            return []
        
        splits = self._recursive_split(text, self.separators)
        
        chunks = []
        for i, split in enumerate(splits):
            if split.strip():
                chunks.append(Chunk(
                    text=split.strip(),
                    index=i,
                    metadata=metadata or {},
                ))
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that groups text by topic similarity.
    
    Uses sentence embeddings to detect topic boundaries.
    Best for: Long documents where semantic coherence is critical.
    
    Requires: sentence-transformers
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._model = None
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def _compute_similarity(self, emb1, emb2) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        text = self._normalize_text(text)
        if not text:
            return []
        
        # First split into sentences
        sentence_chunker = SentenceChunker(max_chunk_size=500, min_chunk_size=50)
        sentences = sentence_chunker.chunk(text)
        
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for each sentence
        model = self._get_model()
        texts = [s.text for s in sentences]
        embeddings = model.encode(texts)
        
        # Find semantic boundaries
        chunks = []
        current_sentences = [sentences[0]]
        current_length = len(sentences[0].text)
        
        for i in range(1, len(sentences)):
            similarity = self._compute_similarity(embeddings[i-1], embeddings[i])
            sentence_len = len(sentences[i].text)
            
            # Start new chunk if topic changed or size limit reached
            if (similarity < self.similarity_threshold or 
                current_length + sentence_len > self.max_chunk_size):
                chunks.append(Chunk(
                    text=" ".join(s.text for s in current_sentences),
                    index=len(chunks),
                    metadata=metadata or {},
                ))
                current_sentences = []
                current_length = 0
            
            current_sentences.append(sentences[i])
            current_length += sentence_len + 1
        
        # Handle remaining sentences
        if current_sentences:
            chunks.append(Chunk(
                text=" ".join(s.text for s in current_sentences),
                index=len(chunks),
                metadata=metadata or {},
            ))
        
        return chunks

