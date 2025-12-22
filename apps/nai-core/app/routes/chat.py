"""
nAI Core Chat Routes
Multi-turn conversation endpoints
"""

import uuid
from typing import Optional, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..config import Settings, get_config
from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    Citation,
)
from ..services.indexer import IndexerService, get_indexer
from ..services.retriever import RetrieverService, get_retriever
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..services.answerer import AnswererService, get_answerer
from ..utils.logging import get_logger

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)

# In-memory conversation store (replace with Redis/DB in production)
_conversations: Dict[str, List[ChatMessage]] = {}


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    conversation_id: Optional[str] = None,
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    retriever: RetrieverService = Depends(get_retriever),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    answerer: AnswererService = Depends(get_answerer),
) -> ChatResponse:
    """
    Multi-turn chat endpoint.
    
    Maintains conversation context and retrieves relevant documents
    based on the full conversation history.
    """
    from datetime import datetime
    
    # Get or create conversation
    if conversation_id and conversation_id in _conversations:
        history = _conversations[conversation_id]
    else:
        conversation_id = str(uuid.uuid4())
        history = []
    
    # Add new messages to history
    for msg in request.messages:
        if msg.timestamp is None:
            msg.timestamp = datetime.utcnow().isoformat() + "Z"
        history.append(msg)
    
    # Get the latest user message
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    citations: List[Citation] = []
    
    # Retrieve context if enabled
    if request.use_context:
        records = indexer.load_index()
        
        if records:
            # Build context query from recent messages
            context_query = " ".join(
                msg.content for msg in history[-5:]
                if msg.role == "user"
            )
            
            if settings.embeddings_enabled and embeddings.is_available():
                embedding_results = embeddings.search(context_query, top_k=request.top_k)
                results = retriever.search_hybrid(
                    context_query,
                    records,
                    top_k=request.top_k,
                    embedding_results=embedding_results,
                )
            else:
                results = retriever.search_bm25(context_query, records, top_k=request.top_k)
            
            citations = retriever.results_to_citations(results)
    
    # Generate response
    if settings.llm_enabled and answerer.is_llm_available():
        # Use LLM with conversation context
        answer = await _generate_chat_response(
            history,
            citations,
            answerer,
            settings,
        )
    else:
        # Extractive fallback
        answer = answerer.compose_extractive_answer(user_message, citations)
    
    # Create response message
    response_message = ChatMessage(
        role="assistant",
        content=answer,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
    
    # Update history
    history.append(response_message)
    _conversations[conversation_id] = history[-20:]  # Keep last 20 messages
    
    return ChatResponse(
        message=response_message,
        citations=citations,
        conversation_id=conversation_id,
    )


async def _generate_chat_response(
    history: List[ChatMessage],
    citations: List[Citation],
    answerer: AnswererService,
    settings: Settings,
) -> str:
    """Generate chat response using LLM with conversation history."""
    import os
    
    try:
        import litellm
        
        # Build context from citations
        context = ""
        if citations:
            context_parts = []
            for cit in citations[:5]:
                filename = os.path.basename(cit.doc_path)
                context_parts.append(f"[{filename}]: {cit.text[:500]}")
            context = "\n\n".join(context_parts)
        
        # Build messages for LLM
        system_content = """You are a helpful AI assistant that answers questions based on provided documents and conversation history.

When relevant context is provided, use it to answer accurately. If the context doesn't contain enough information, say so clearly.

Be conversational and helpful. Reference previous parts of the conversation when relevant."""
        
        if context:
            system_content += f"\n\nRelevant document context:\n{context}"
        
        messages = [{"role": "system", "content": system_content}]
        
        # Add conversation history
        for msg in history:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        # Determine model
        if settings.llm_provider == "ollama":
            model = f"ollama/{settings.llm_model}"
        elif settings.llm_provider == "anthropic":
            model = f"anthropic/{settings.llm_model}"
        else:
            model = settings.llm_model
        
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_base=settings.llm_base_url if settings.llm_provider == "ollama" else None,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Chat LLM failed: {e}")
        # Fallback
        return "I apologize, but I'm having trouble generating a response right now. Please try again."


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
):
    """Get conversation history by ID."""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": [msg.model_dump() for msg in _conversations[conversation_id]],
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
):
    """Delete a conversation."""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del _conversations[conversation_id]
    return {"success": True, "message": "Conversation deleted"}

