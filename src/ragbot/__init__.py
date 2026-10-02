"""ragbot — a grounded RAG chatbot with tool calling and citations.

    from ragbot import RagChatbot
    bot = RagChatbot()
    bot.ingest_path("corpus")
    print(bot.chat("How long do refunds take?").text)
"""
from .chat import Answer, Citation, RagChatbot, SYSTEM_PROMPT
from .chunking import Chunk, chunk_document
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder, cosine, tokenize
from .llm import (
    AnthropicLLM, CallableLLM, EchoContextLLM, LLM, LLMResponse,
    Message, OpenAILLM, ScriptedLLM, parse_tool_calls,
)
from .retriever import Hit, HybridRetriever, VectorStore, lexical_overlap
from .tools import Tool, ToolError, ToolRegistry, calculator, make_calculator_tool, make_lookup_tool

__version__ = "0.1.0"
__all__ = [
    "RagChatbot", "Answer", "Citation", "SYSTEM_PROMPT",
    "Chunk", "chunk_document",
    "Embedder", "HashingEmbedder", "OpenAIEmbedder", "cosine", "tokenize",
    "LLM", "Message", "LLMResponse", "ScriptedLLM", "EchoContextLLM",
    "OpenAILLM", "AnthropicLLM", "CallableLLM", "parse_tool_calls",
    "HybridRetriever", "VectorStore", "Hit", "lexical_overlap",
    "Tool", "ToolRegistry", "ToolError", "calculator",
    "make_calculator_tool", "make_lookup_tool",
    "__version__",
]
