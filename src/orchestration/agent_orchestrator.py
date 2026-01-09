"""
Multi-Agent Orchestration for RAG Pipeline

Coordinates multiple specialized agents for document processing,
retrieval, reasoning, and response generation.

This module implements Task 4 of the AI Developer Roadmap:
- Multi-agent workflow orchestration
- Logical integration points
- Agent coordination and communication
- Self-reflection and improvement patterns
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of agents in the orchestration system."""

    ROUTER = "router"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    REFLECTOR = "reflector"
    EXTRACTOR = "extractor"


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentState:
    """State passed between agents in the workflow."""

    workflow_id: str
    query: str
    context: List[str] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_step: str = ""
    response: Optional[str] = None
    confidence: float = 0.0
    errors: List[str] = field(default_factory=list)

    # Execution tracking
    steps_executed: List[str] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_context(self, context: str) -> None:
        """Add context to the state."""
        self.context.append(context)

    def add_step(self, step_name: str, execution_time: float) -> None:
        """Record a step execution."""
        self.steps_executed.append(step_name)
        self.execution_times[step_name] = execution_time


@dataclass
class AgentResult:
    """Result from an agent execution."""

    agent_type: AgentType
    success: bool
    output: Any
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["agent_type"] = self.agent_type.value
        return result


@dataclass
class WorkflowResult:
    """Final result from workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    response: Optional[str]
    confidence: float
    total_time: float
    steps_completed: List[str]
    agent_results: List[AgentResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "response": self.response,
            "confidence": self.confidence,
            "total_time": self.total_time,
            "steps_completed": self.steps_completed,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "metadata": self.metadata,
        }
        return result


class Agent(ABC):
    """Base class for all agents in the orchestration system."""

    def __init__(self, agent_type: AgentType, name: Optional[str] = None):
        self.agent_type = agent_type
        self.name = name or agent_type.value
        self.is_initialized = False

    @abstractmethod
    def execute(self, state: AgentState) -> AgentResult:
        """Execute the agent's task."""
        pass

    def initialize(self) -> None:
        """Initialize the agent (override if needed)."""
        self.is_initialized = True

    def cleanup(self) -> None:
        """Cleanup agent resources (override if needed)."""
        pass


class RouterAgent(Agent):
    """Agent for query classification and routing."""

    QUERY_PATTERNS = {
        "factual": ["what is", "who is", "when", "where", "define"],
        "analytical": ["analyze", "compare", "explain why", "evaluate"],
        "procedural": ["how to", "steps to", "process for"],
        "summarization": ["summarize", "overview", "brief"],
    }

    def __init__(self):
        super().__init__(AgentType.ROUTER, "Query Router")

    def execute(self, state: AgentState) -> AgentResult:
        """Classify and route the query."""
        start_time = datetime.now()

        try:
            query_lower = state.query.lower()
            query_type = "general"
            confidence = 0.5

            for q_type, patterns in self.QUERY_PATTERNS.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        query_type = q_type
                        confidence = 0.9
                        break
                if confidence > 0.5:
                    break

            # Determine routing
            routing = self._determine_routing(query_type)

            state.metadata["query_type"] = query_type
            state.metadata["routing"] = routing
            state.confidence = confidence

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "query_type": query_type,
                    "routing": routing,
                    "confidence": confidence,
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )

    def _determine_routing(self, query_type: str) -> List[str]:
        """Determine agent routing based on query type."""
        base_route = ["retriever", "reranker", "synthesizer"]

        if query_type == "analytical":
            return ["retriever", "reranker", "synthesizer", "validator", "reflector"]
        elif query_type == "summarization":
            return ["retriever", "synthesizer"]

        return base_route


class RetrieverAgent(Agent):
    """Agent for document retrieval."""

    def __init__(self, retrieval_service=None):
        super().__init__(AgentType.RETRIEVER, "Document Retriever")
        self.retrieval_service = retrieval_service

    def execute(self, state: AgentState) -> AgentResult:
        """Retrieve relevant documents."""
        start_time = datetime.now()

        try:
            # Simulate retrieval (in production, use actual retrieval service)
            retrieved_docs = []

            if self.retrieval_service:
                # Use actual retrieval service
                result = self.retrieval_service.retrieve(
                    query=state.query,
                    top_k=5,
                )
                retrieved_docs = result.results if hasattr(result, "results") else []
            else:
                # Mock retrieval for demonstration
                retrieved_docs = [
                    {
                        "id": "doc_1",
                        "content": f"Relevant content for query: {state.query[:50]}...",
                        "score": 0.95,
                        "metadata": {"source": "knowledge_base"},
                    },
                    {
                        "id": "doc_2",
                        "content": "Additional context and supporting information.",
                        "score": 0.87,
                        "metadata": {"source": "documents"},
                    },
                ]

            state.retrieved_documents = retrieved_docs
            state.context = [doc.get("content", "") for doc in retrieved_docs]

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "documents_retrieved": len(retrieved_docs),
                    "avg_score": sum(d.get("score", 0) for d in retrieved_docs)
                    / len(retrieved_docs)
                    if retrieved_docs
                    else 0,
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )


class RerankerAgent(Agent):
    """Agent for reranking retrieved documents."""

    def __init__(self):
        super().__init__(AgentType.RERANKER, "Document Reranker")

    def execute(self, state: AgentState) -> AgentResult:
        """Rerank retrieved documents."""
        start_time = datetime.now()

        try:
            # Simple reranking based on score (in production, use cross-encoder)
            docs = state.retrieved_documents

            # Sort by relevance score
            reranked = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)

            # Filter low-quality results
            threshold = 0.5
            filtered = [d for d in reranked if d.get("score", 0) >= threshold]

            state.retrieved_documents = filtered
            state.context = [doc.get("content", "") for doc in filtered]

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "original_count": len(docs),
                    "reranked_count": len(filtered),
                    "filtered_out": len(docs) - len(filtered),
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )


class SynthesizerAgent(Agent):
    """Agent for response synthesis."""

    def __init__(self, llm_service=None):
        super().__init__(AgentType.SYNTHESIZER, "Response Synthesizer")
        self.llm_service = llm_service

    def execute(self, state: AgentState) -> AgentResult:
        """Synthesize response from context."""
        start_time = datetime.now()

        try:
            context = "\n\n".join(state.context)
            query = state.query

            if self.llm_service:
                # Use actual LLM service
                response = self.llm_service.generate(
                    prompt=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                )
            else:
                # Mock response for demonstration
                response = (
                    f"Based on the retrieved context about '{query}', "
                    f"the relevant information indicates that this topic "
                    f"involves multiple aspects covered in the knowledge base. "
                    f"The documents provide comprehensive information "
                    f"addressing the query requirements."
                )

            state.response = response
            state.confidence = 0.85

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "response_length": len(response),
                    "context_used": len(state.context),
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )


class ValidatorAgent(Agent):
    """Agent for response validation."""

    def __init__(self):
        super().__init__(AgentType.VALIDATOR, "Response Validator")

    def execute(self, state: AgentState) -> AgentResult:
        """Validate the generated response."""
        start_time = datetime.now()

        try:
            response = state.response or ""

            # Validation checks
            validations = {
                "has_content": len(response) > 10,
                "not_too_short": len(response) > 50,
                "not_too_long": len(response) < 5000,
                "has_context_reference": any(
                    c[:20] in response for c in state.context if c
                ),
            }

            is_valid = all(validations.values())
            validation_score = sum(validations.values()) / len(validations)

            state.metadata["validation"] = validations
            state.metadata["validation_score"] = validation_score

            # Adjust confidence based on validation
            if is_valid:
                state.confidence = min(1.0, state.confidence + 0.1)
            else:
                state.confidence = max(0.0, state.confidence - 0.2)

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "is_valid": is_valid,
                    "validation_score": validation_score,
                    "validations": validations,
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )


class ReflectorAgent(Agent):
    """Agent for self-reflection and improvement."""

    def __init__(self):
        super().__init__(AgentType.REFLECTOR, "Self-Reflector")
        self.reflection_history: List[Dict[str, Any]] = []

    def execute(self, state: AgentState) -> AgentResult:
        """Reflect on the response and suggest improvements."""
        start_time = datetime.now()

        try:
            response = state.response or ""
            validation = state.metadata.get("validation", {})

            # Reflection analysis
            reflections = []
            improvements = []

            if not validation.get("has_context_reference", True):
                reflections.append("Response may not be well-grounded in context")
                improvements.append(
                    "Consider adding specific references to source documents"
                )

            if len(response) < 100:
                reflections.append("Response is relatively brief")
                improvements.append("Consider expanding with more details")

            if state.confidence < 0.7:
                reflections.append("Low confidence in response quality")
                improvements.append("Additional context retrieval may help")

            # Store reflection
            reflection_record = {
                "workflow_id": state.workflow_id,
                "reflections": reflections,
                "improvements": improvements,
                "confidence_before": state.confidence,
            }
            self.reflection_history.append(reflection_record)

            state.metadata["reflections"] = reflections
            state.metadata["improvements"] = improvements

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                output={
                    "reflections_count": len(reflections),
                    "improvements_suggested": len(improvements),
                    "should_iterate": len(improvements) > 0 and state.confidence < 0.7,
                },
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
            )


class AgentOrchestrator:
    """
    Multi-agent workflow orchestration.

    This class implements the core functionality for Task 4:
    Model Integration of the AI Developer Roadmap.

    Agents:
    - Router: Query classification and routing
    - Retriever: Vector and graph search
    - Reranker: Result relevance scoring
    - Synthesizer: Response generation
    - Validator: Output quality checking
    - Reflector: Self-improvement feedback

    Example:
        orchestrator = AgentOrchestrator()
        result = orchestrator.process_query("What is machine learning?")
        print(result.response)
    """

    def __init__(
        self,
        retrieval_service=None,
        llm_service=None,
        max_iterations: int = 3,
    ):
        """
        Initialize the agent orchestrator.

        Args:
            retrieval_service: Optional retrieval service for documents
            llm_service: Optional LLM service for generation
            max_iterations: Maximum iterations for reflection loop
        """
        self.max_iterations = max_iterations

        # Initialize agents
        self.agents: Dict[str, Agent] = {
            "router": RouterAgent(),
            "retriever": RetrieverAgent(retrieval_service),
            "reranker": RerankerAgent(),
            "synthesizer": SynthesizerAgent(llm_service),
            "validator": ValidatorAgent(),
            "reflector": ReflectorAgent(),
        }

        # Default workflow
        self.default_workflow = [
            "router",
            "retriever",
            "reranker",
            "synthesizer",
            "validator",
        ]

        # Execution history
        self.execution_history: List[WorkflowResult] = []

        logger.info(
            "AgentOrchestrator initialized with agents: "
            + ", ".join(self.agents.keys())
        )

    def _generate_workflow_id(self) -> str:
        """Generate unique workflow ID."""
        return f"wf_{uuid.uuid4().hex[:12]}"

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        workflow: Optional[List[str]] = None,
    ) -> WorkflowResult:
        """
        Process a query through the agent workflow.

        Args:
            query: User query to process
            session_id: Optional session ID for context
            history: Optional conversation history
            workflow: Optional custom workflow (agent names)

        Returns:
            WorkflowResult with response and metadata
        """
        start_time = datetime.now()
        workflow_id = self._generate_workflow_id()

        # Initialize state
        state = AgentState(
            workflow_id=workflow_id,
            query=query,
            history=history or [],
            metadata={
                "session_id": session_id,
                "started_at": start_time.isoformat(),
            },
        )

        agent_results: List[AgentResult] = []

        try:
            # Determine workflow
            workflow_agents = workflow or self.default_workflow

            # Execute routing first to potentially modify workflow
            router_result = self.agents["router"].execute(state)
            agent_results.append(router_result)
            state.add_step("router", router_result.execution_time)

            if router_result.success and router_result.output:
                routing = router_result.output.get("routing", [])
                if routing:
                    workflow_agents = routing

            # Execute remaining agents
            for agent_name in workflow_agents:
                if agent_name == "router":
                    continue  # Already executed

                if agent_name not in self.agents:
                    logger.warning(f"Unknown agent: {agent_name}")
                    continue

                state.current_step = agent_name
                agent = self.agents[agent_name]
                result = agent.execute(state)
                agent_results.append(result)
                state.add_step(agent_name, result.execution_time)

                if not result.success:
                    state.errors.append(f"{agent_name}: {result.error}")
                    logger.warning(f"Agent {agent_name} failed: {result.error}")

            # Determine final status
            status = WorkflowStatus.COMPLETED
            if state.errors:
                status = (
                    WorkflowStatus.FAILED
                    if not state.response
                    else WorkflowStatus.COMPLETED
                )

            total_time = (datetime.now() - start_time).total_seconds()

            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                status=status,
                response=state.response,
                confidence=state.confidence,
                total_time=total_time,
                steps_completed=state.steps_executed,
                agent_results=agent_results,
                metadata={
                    **state.metadata,
                    "errors": state.errors,
                    "execution_times": state.execution_times,
                },
            )

            self.execution_history.append(workflow_result)

            logger.info(
                f"Workflow {workflow_id} completed in {total_time:.3f}s with status {status.value}"
            )
            return workflow_result

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()

            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                response=None,
                confidence=0.0,
                total_time=total_time,
                steps_completed=state.steps_executed,
                agent_results=agent_results,
                metadata={"error": str(e)},
            )

            self.execution_history.append(workflow_result)
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return workflow_result

    async def process_query_async(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> WorkflowResult:
        """Async version of process_query."""
        # Run in thread pool for compatibility with sync agents
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process_query(query, session_id, history),
        )

    def get_agent_chain(self, query_type: str) -> List[str]:
        """Get agent chain based on query type."""
        chains = {
            "factual": ["router", "retriever", "reranker", "synthesizer"],
            "analytical": [
                "router",
                "retriever",
                "reranker",
                "synthesizer",
                "validator",
                "reflector",
            ],
            "summarization": ["router", "retriever", "synthesizer"],
            "general": self.default_workflow,
        }
        return chains.get(query_type, self.default_workflow)

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register a custom agent."""
        self.agents[name] = agent
        logger.info(f"Registered custom agent: {name}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        if not self.execution_history:
            return {"total_workflows": 0}

        completed = sum(
            1 for r in self.execution_history if r.status == WorkflowStatus.COMPLETED
        )
        failed = sum(
            1 for r in self.execution_history if r.status == WorkflowStatus.FAILED
        )
        avg_time = sum(r.total_time for r in self.execution_history) / len(
            self.execution_history
        )
        avg_confidence = sum(r.confidence for r in self.execution_history) / len(
            self.execution_history
        )

        return {
            "total_workflows": len(self.execution_history),
            "completed": completed,
            "failed": failed,
            "success_rate": completed / len(self.execution_history),
            "avg_execution_time": avg_time,
            "avg_confidence": avg_confidence,
            "agents_available": list(self.agents.keys()),
        }

    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics per agent."""
        performance: Dict[str, Dict[str, Any]] = {}

        for result in self.execution_history:
            for agent_result in result.agent_results:
                agent_name = agent_result.agent_type.value
                if agent_name not in performance:
                    performance[agent_name] = {
                        "executions": 0,
                        "successes": 0,
                        "total_time": 0.0,
                    }

                performance[agent_name]["executions"] += 1
                if agent_result.success:
                    performance[agent_name]["successes"] += 1
                performance[agent_name]["total_time"] += agent_result.execution_time

        # Calculate averages
        for agent_name, stats in performance.items():
            if stats["executions"] > 0:
                stats["success_rate"] = stats["successes"] / stats["executions"]
                stats["avg_time"] = stats["total_time"] / stats["executions"]

        return performance

    def visualize_workflow(self, workflow_result: WorkflowResult) -> str:
        """Generate Mermaid diagram for workflow visualization."""
        lines = ["```mermaid", "graph LR"]

        # Add nodes for each step
        for i, step in enumerate(workflow_result.steps_completed):
            status = "✓" if workflow_result.agent_results[i].success else "✗"
            time = f"{workflow_result.agent_results[i].execution_time:.3f}s"
            lines.append(f'    {step}["{step}\\n{status} {time}"]')

        # Add edges
        for i in range(len(workflow_result.steps_completed) - 1):
            current = workflow_result.steps_completed[i]
            next_step = workflow_result.steps_completed[i + 1]
            lines.append(f"    {current} --> {next_step}")

        # Style based on success
        for i, step in enumerate(workflow_result.steps_completed):
            if workflow_result.agent_results[i].success:
                lines.append(f"    style {step} fill:#90EE90")
            else:
                lines.append(f"    style {step} fill:#FFB6C1")

        lines.append("```")
        return "\n".join(lines)

    def cleanup(self) -> None:
        """Cleanup all agents."""
        for agent in self.agents.values():
            agent.cleanup()
        logger.info("Orchestrator cleanup complete")
