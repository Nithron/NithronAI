"""
nAI Core Answerer Service
Answer generation with extractive fallback and LLM support
"""

from typing import List, Dict, Any, Optional, Tuple
import os

from ..config import Settings, get_settings
from ..models.schemas import Citation, AskResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


# LLM prompts
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents.

Instructions:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say so.
2. Be concise and accurate.
3. When referencing information, mention which source document it comes from.
4. If you're uncertain, express that uncertainty.
5. Format your response clearly with proper structure when appropriate.

Context documents will be provided in the following format:
[Source: filename, Chunk #N]
<content>
"""

QA_PROMPT_TEMPLATE = """Based on the following context documents, please answer the question.

{context}

Question: {question}

Provide a clear, accurate answer based only on the information in the context above. If the context doesn't contain enough information to answer the question fully, say so and provide what information is available."""


class AnswererService:
    """
    Service for answer generation.
    Supports extractive answers (no LLM) and LLM-powered generation.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm_client = None
        self._llm_available: Optional[bool] = None
    
    def _init_llm_client(self) -> bool:
        """Initialize the LLM client if enabled and available."""
        if not self.settings.llm_enabled:
            return False
        
        if self._llm_available is not None:
            return self._llm_available
        
        try:
            # Use LiteLLM for unified interface
            import litellm
            
            # Configure based on provider
            if self.settings.llm_provider == "ollama":
                litellm.api_base = self.settings.llm_base_url
            elif self.settings.llm_api_key:
                # For OpenAI, Anthropic, etc.
                os.environ["OPENAI_API_KEY"] = self.settings.llm_api_key
            
            self._llm_available = True
            logger.info(f"LLM initialized: {self.settings.llm_provider}/{self.settings.llm_model}")
            return True
        
        except ImportError:
            logger.warning("LiteLLM not installed, falling back to extractive answers")
            self._llm_available = False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._llm_available = False
            return False
    
    def _format_context(self, citations: List[Citation]) -> str:
        """Format citations as context for LLM."""
        context_parts = []
        
        for i, citation in enumerate(citations):
            filename = os.path.basename(citation.doc_path)
            context_parts.append(
                f"[Source: {filename}, Chunk #{citation.chunk_id}]\n"
                f"{citation.text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def compose_extractive_answer(
        self,
        question: str,
        citations: List[Citation],
        max_excerpt_length: int = 500
    ) -> str:
        """
        Generate an extractive answer by selecting and formatting relevant passages.
        No LLM required.
        """
        if not citations:
            return "I couldn't find any relevant information in the indexed documents for your question."
        
        # Select top passages
        top_citations = citations[:3]
        
        # Format excerpts
        excerpts = []
        for i, cit in enumerate(top_citations, 1):
            text = cit.text[:max_excerpt_length]
            if len(cit.text) > max_excerpt_length:
                # Try to end at a sentence
                last_period = text.rfind('.')
                if last_period > max_excerpt_length // 2:
                    text = text[:last_period + 1]
                else:
                    text = text + "..."
            
            filename = os.path.basename(cit.doc_path)
            excerpts.append(
                f"**[{i}] From: {filename}** (relevance: {cit.score:.2f})\n"
                f"> {text}"
            )
        
        answer = f"""Based on your indexed documents, here are the most relevant excerpts:

{chr(10).join(excerpts)}

---
*This is an extractive answer based on BM25 retrieval. For better answers, enable LLM generation by setting `NAI_LLM_ENABLED=true`.*
"""
        
        return answer.strip()
    
    async def compose_llm_answer(
        self,
        question: str,
        citations: List[Citation]
    ) -> Tuple[str, Optional[int]]:
        """
        Generate an answer using LLM.
        Returns (answer, tokens_used).
        """
        import litellm
        
        context = self._format_context(citations)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Determine model name based on provider
        if self.settings.llm_provider == "ollama":
            model = f"ollama/{self.settings.llm_model}"
        elif self.settings.llm_provider == "anthropic":
            model = f"anthropic/{self.settings.llm_model}"
        else:
            model = self.settings.llm_model
        
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                api_base=self.settings.llm_base_url if self.settings.llm_provider == "ollama" else None,
            )
            
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None
            
            # Add citation references
            if citations:
                sources = "\n\n**Sources:**\n"
                for i, cit in enumerate(citations[:5], 1):
                    filename = os.path.basename(cit.doc_path)
                    sources += f"- [{i}] {filename} (chunk {cit.chunk_id})\n"
                answer += sources
            
            return answer, tokens
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall back to extractive
            return self.compose_extractive_answer(question, citations), None
    
    async def answer(
        self,
        question: str,
        citations: List[Citation],
        use_llm: bool = True
    ) -> AskResponse:
        """
        Generate an answer for the question based on retrieved citations.
        
        Args:
            question: User's question
            citations: Retrieved relevant passages
            use_llm: Whether to use LLM (if available)
        
        Returns:
            AskResponse with answer and metadata
        """
        # Check if LLM should be used
        llm_ready = use_llm and self._init_llm_client() and self._llm_available
        
        if llm_ready:
            try:
                answer, tokens = await self.compose_llm_answer(question, citations)
                return AskResponse(
                    answer=answer,
                    citations=citations,
                    method="llm",
                    model=f"{self.settings.llm_provider}/{self.settings.llm_model}",
                    tokens_used=tokens,
                )
            except Exception as e:
                logger.warning(f"LLM failed, falling back to extractive: {e}")
        
        # Extractive fallback
        answer = self.compose_extractive_answer(question, citations)
        return AskResponse(
            answer=answer,
            citations=citations,
            method="extractive",
            model=None,
        )
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available."""
        return self._init_llm_client()


# Singleton instance
_answerer_instance: Optional[AnswererService] = None


def get_answerer() -> AnswererService:
    """Get or create the answerer service singleton."""
    global _answerer_instance
    if _answerer_instance is None:
        _answerer_instance = AnswererService(get_settings())
    return _answerer_instance

