"""
Prompt Service for RAG Prompt Management

Provides prompt template management, context injection,
and prompt building for RAG applications.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""

    name: str
    template: str
    description: str = ""
    required_variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.required_variables:
            pattern = r"\{(\w+)\}"
            self.required_variables = re.findall(pattern, self.template)

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        missing = set(self.required_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)


class PromptService:
    """
    Service for managing prompts and building context-aware prompts.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to document knowledge.
Answer questions accurately based on the provided context. If information is not
available in the context, acknowledge this honestly."""

    DEFAULT_TEMPLATES = {
        "rag_qa": PromptTemplate(
            name="rag_qa",
            template="""Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:""",
            description="Question answering with retrieved context",
        ),
        "rag_summarize": PromptTemplate(
            name="rag_summarize",
            template="""Summarize the following content concisely:

{context}

Summary:""",
            description="Summarize retrieved content",
        ),
        "document_analysis": PromptTemplate(
            name="document_analysis",
            template="""Analyze this document and extract key information:

Document: {filename}
Content:
{context}

Provide:
1. Summary
2. Key Points
3. Important Entities
4. Document Type""",
            description="Comprehensive document analysis",
        ),
    }

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.templates: Dict[str, PromptTemplate] = dict(self.DEFAULT_TEMPLATES)
        self.is_initialized = True
        logger.info("PromptService initialized")

    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom prompt template."""
        self.templates[template.name] = template
        logger.info(f"Registered template: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def build_prompt(self, template_name: str, **variables) -> str:
        """Build a prompt using a named template."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        return template.format(**variables)

    def build_rag_prompt(
        self, question: str, context: str, template_name: str = "rag_qa"
    ) -> str:
        """Build a RAG prompt with context injection."""
        return self.build_prompt(template_name, question=question, context=context)

    def build_chat_messages(
        self,
        user_message: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for LLM API.

        Returns list of messages in chat format:
        [{"role": "system", "content": "..."}, ...]
        """
        messages = []

        # System message
        system = system_prompt or self.system_prompt
        if context:
            system += f"\n\nRelevant Context:\n{context}"
        messages.append({"role": "system", "content": system})

        # Add history
        if history:
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def format_context(
        self, chunks: List[str], separator: str = "\n\n---\n\n", max_length: int = 4000
    ) -> str:
        """Format multiple chunks into a single context string."""
        if not chunks:
            return ""

        combined = separator.join(chunks)

        if len(combined) > max_length:
            combined = combined[:max_length] + "..."

        return combined

    def format_history(
        self, messages: List[Dict[str, str]], max_messages: int = 10
    ) -> str:
        """Format conversation history as a string."""
        if not messages:
            return "No previous conversation."

        recent = messages[-max_messages:]
        formatted = []

        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def list_templates(self) -> List[Dict[str, str]]:
        """List available templates."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "variables": t.required_variables,
            }
            for t in self.templates.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "template_count": len(self.templates),
            "templates": list(self.templates.keys()),
        }
