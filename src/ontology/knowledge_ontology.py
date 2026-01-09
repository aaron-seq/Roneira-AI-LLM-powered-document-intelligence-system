"""
Knowledge Ontology for Document Intelligence

Defines domain-specific semantic structures, entity relationships,
and conceptual hierarchies for intelligent document processing.

This module implements Task 3 of the AI Developer Roadmap:
- Ontology schema definition
- Entity and concept management
- Relationship mapping and hierarchies
- Integration with RAG pipeline
- Visualization and export capabilities
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of ontology nodes."""

    ENTITY = "entity"
    CONCEPT = "concept"
    CATEGORY = "category"
    ATTRIBUTE = "attribute"
    ACTION = "action"


class RelationType(str, Enum):
    """Types of relationships between nodes."""

    IS_A = "is_a"
    PART_OF = "part_of"
    RELATES_TO = "relates_to"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    INSTANCE_OF = "instance_of"


@dataclass
class OntologyNode:
    """Represents a node in the knowledge ontology."""

    node_id: str
    name: str
    node_type: NodeType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["node_type"] = self.node_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyNode":
        data["node_type"] = NodeType(data["node_type"])
        return cls(**data)


@dataclass
class OntologyRelation:
    """Represents a relationship between two nodes."""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["relation_type"] = self.relation_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyRelation":
        data["relation_type"] = RelationType(data["relation_type"])
        return cls(**data)


@dataclass
class OntologyStatistics:
    """Statistics about the ontology."""

    total_nodes: int = 0
    total_relations: int = 0
    nodes_by_type: Dict[str, int] = field(default_factory=dict)
    relations_by_type: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    avg_relations_per_node: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeOntology:
    """
    Domain ontology for document intelligence systems.

    This class implements the core functionality for Task 3:
    Prompt & Ontology Creation of the AI Developer Roadmap.

    Features:
    - Entity and concept definition with hierarchies
    - Multiple relationship types (is_a, part_of, relates_to, etc.)
    - Path finding and traversal algorithms
    - Integration points for RAG pipeline context enhancement
    - RDF/OWL export for interoperability
    - DOT/Mermaid visualization export

    Example:
        ontology = KnowledgeOntology("document_intelligence")

        # Add nodes
        doc_node = ontology.add_entity("document", "A written or electronic file")
        pdf_node = ontology.add_entity("pdf", "Portable Document Format file")

        # Add relationships
        ontology.add_relation(pdf_node.node_id, doc_node.node_id, RelationType.IS_A)

        # Query
        related = ontology.find_related_concepts("document", depth=2)
        hierarchy = ontology.get_hierarchy("document")
    """

    # Core document intelligence domain concepts
    CORE_CONCEPTS = {
        "document": {
            "type": NodeType.CATEGORY,
            "description": "Base category for all document types",
            "children": ["pdf", "image", "text", "spreadsheet"],
        },
        "entity": {
            "type": NodeType.CATEGORY,
            "description": "Named entities extracted from documents",
            "children": ["person", "organization", "location", "date", "money"],
        },
        "action": {
            "type": NodeType.CATEGORY,
            "description": "Actions that can be performed on documents",
            "children": ["extract", "analyze", "summarize", "translate", "classify"],
        },
    }

    def __init__(self, name: str = "default", auto_init_core: bool = True):
        """
        Initialize the knowledge ontology.

        Args:
            name: Name identifier for this ontology
            auto_init_core: Whether to initialize core document concepts
        """
        self.name = name
        self.nodes: Dict[str, OntologyNode] = {}
        self.relations: Dict[str, OntologyRelation] = {}

        # Index structures for efficient lookup
        self._name_index: Dict[str, str] = {}  # name -> node_id
        self._parent_index: Dict[str, List[str]] = defaultdict(
            list
        )  # parent_id -> [child_ids]
        self._relations_from: Dict[str, List[str]] = defaultdict(
            list
        )  # source_id -> [relation_ids]
        self._relations_to: Dict[str, List[str]] = defaultdict(
            list
        )  # target_id -> [relation_ids]

        self.created_at = datetime.now().isoformat()

        if auto_init_core:
            self._initialize_core_concepts()

        logger.info(f"Initialized KnowledgeOntology: {name}")

    def _generate_node_id(self, name: str) -> str:
        """Generate unique node ID."""
        base = name.lower().replace(" ", "_")[:20]
        short_uuid = str(uuid.uuid4())[:8]
        return f"{base}_{short_uuid}"

    def _generate_relation_id(self) -> str:
        """Generate unique relation ID."""
        return f"rel_{uuid.uuid4().hex[:12]}"

    def _initialize_core_concepts(self) -> None:
        """Initialize core domain concepts."""
        for concept_name, concept_data in self.CORE_CONCEPTS.items():
            node = self.add_node(
                name=concept_name,
                node_type=concept_data["type"],
                description=concept_data["description"],
            )

            # Add children
            for child_name in concept_data.get("children", []):
                child = self.add_node(
                    name=child_name,
                    node_type=NodeType.CONCEPT,
                    description=f"{child_name.capitalize()} type",
                    parent_id=node.node_id,
                )
                self.add_relation(
                    child.node_id,
                    node.node_id,
                    RelationType.IS_A,
                )

        logger.info("Initialized core ontology concepts")

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> OntologyNode:
        """
        Add a new node to the ontology.

        Args:
            name: Node name
            node_type: Type of node
            description: Node description
            properties: Additional properties
            parent_id: Parent node ID for hierarchies
            aliases: Alternative names for this concept

        Returns:
            Created OntologyNode
        """
        node_id = self._generate_node_id(name)

        node = OntologyNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            description=description,
            properties=properties or {},
            parent_id=parent_id,
            aliases=aliases or [],
        )

        self.nodes[node_id] = node
        self._name_index[name.lower()] = node_id

        if parent_id:
            self._parent_index[parent_id].append(node_id)

        # Index aliases
        for alias in node.aliases:
            self._name_index[alias.lower()] = node_id

        logger.debug(f"Added node: {name} ({node_type.value})")
        return node

    def add_entity(
        self,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> OntologyNode:
        """Add an entity node."""
        return self.add_node(name, NodeType.ENTITY, description, properties)

    def add_concept(
        self,
        name: str,
        description: str = "",
        parent_name: Optional[str] = None,
    ) -> OntologyNode:
        """Add a concept node with optional parent."""
        parent_id = None
        if parent_name:
            parent_id = self._name_index.get(parent_name.lower())
        return self.add_node(name, NodeType.CONCEPT, description, parent_id=parent_id)

    def add_category(
        self,
        name: str,
        description: str = "",
    ) -> OntologyNode:
        """Add a category node."""
        return self.add_node(name, NodeType.CATEGORY, description)

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Optional[OntologyNode]:
        """Get node by name or alias."""
        node_id = self._name_index.get(name.lower())
        if node_id:
            return self.nodes.get(node_id)
        return None

    def update_node(
        self,
        node_id: str,
        **kwargs,
    ) -> Optional[OntologyNode]:
        """Update node properties."""
        node = self.nodes.get(node_id)
        if not node:
            return None

        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)

        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its relations."""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Remove from name index
        if node.name.lower() in self._name_index:
            del self._name_index[node.name.lower()]

        # Remove related relations
        for rel_id in list(self._relations_from.get(node_id, [])):
            self.remove_relation(rel_id)
        for rel_id in list(self._relations_to.get(node_id, [])):
            self.remove_relation(rel_id)

        # Remove from parent index
        if node.parent_id and node_id in self._parent_index.get(node.parent_id, []):
            self._parent_index[node.parent_id].remove(node_id)

        del self.nodes[node_id]
        logger.debug(f"Removed node: {node_id}")
        return True

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 1.0,
        bidirectional: bool = False,
        properties: Optional[Dict[str, Any]] = None,
    ) -> OntologyRelation:
        """
        Add a relationship between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            confidence: Confidence score (0-1)
            bidirectional: Whether relation applies both ways
            properties: Additional relation properties

        Returns:
            Created OntologyRelation
        """
        relation_id = self._generate_relation_id()

        relation = OntologyRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            bidirectional=bidirectional,
            properties=properties or {},
        )

        self.relations[relation_id] = relation
        self._relations_from[source_id].append(relation_id)
        self._relations_to[target_id].append(relation_id)

        if bidirectional:
            self._relations_from[target_id].append(relation_id)
            self._relations_to[source_id].append(relation_id)

        logger.debug(
            f"Added relation: {source_id} --{relation_type.value}--> {target_id}"
        )
        return relation

    def get_relation(self, relation_id: str) -> Optional[OntologyRelation]:
        """Get relation by ID."""
        return self.relations.get(relation_id)

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation."""
        if relation_id not in self.relations:
            return False

        relation = self.relations[relation_id]

        # Remove from indices
        if relation_id in self._relations_from.get(relation.source_id, []):
            self._relations_from[relation.source_id].remove(relation_id)
        if relation_id in self._relations_to.get(relation.target_id, []):
            self._relations_to[relation.target_id].remove(relation_id)

        del self.relations[relation_id]
        return True

    def get_relations_from(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[OntologyRelation]:
        """Get all relations originating from a node."""
        relations = []
        for rel_id in self._relations_from.get(node_id, []):
            rel = self.relations.get(rel_id)
            if rel:
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)
        return relations

    def get_relations_to(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[OntologyRelation]:
        """Get all relations pointing to a node."""
        relations = []
        for rel_id in self._relations_to.get(node_id, []):
            rel = self.relations.get(rel_id)
            if rel:
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)
        return relations

    def get_children(self, node_id: str) -> List[OntologyNode]:
        """Get direct children of a node."""
        child_ids = self._parent_index.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_hierarchy(self, root_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get hierarchical tree structure starting from a node.

        Args:
            root_id: Root node ID
            max_depth: Maximum depth to traverse

        Returns:
            Nested dictionary representing the hierarchy
        """
        root = self.get_node(root_id) or self.get_node_by_name(root_id)
        if not root:
            return {}

        def build_tree(node: OntologyNode, depth: int) -> Dict[str, Any]:
            if depth >= max_depth:
                return {"name": node.name, "truncated": True}

            children = self.get_children(node.node_id)

            return {
                "id": node.node_id,
                "name": node.name,
                "type": node.node_type.value,
                "description": node.description,
                "children": [build_tree(child, depth + 1) for child in children],
            }

        return build_tree(root, 0)

    def find_related_concepts(
        self,
        concept: str,
        depth: int = 2,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[OntologyNode]:
        """
        Find concepts related to a given concept.

        Args:
            concept: Concept name or ID
            depth: How many hops to traverse
            relation_types: Filter by relation types

        Returns:
            List of related OntologyNode objects
        """
        start_node = self.get_node(concept) or self.get_node_by_name(concept)
        if not start_node:
            return []

        visited: Set[str] = {start_node.node_id}
        result: List[OntologyNode] = []
        frontier: List[tuple] = [(start_node.node_id, 0)]

        while frontier:
            current_id, current_depth = frontier.pop(0)

            if current_depth >= depth:
                continue

            # Get outgoing relations
            for relation in self.get_relations_from(current_id):
                if relation_types and relation.relation_type not in relation_types:
                    continue
                if relation.target_id not in visited:
                    visited.add(relation.target_id)
                    target_node = self.nodes.get(relation.target_id)
                    if target_node:
                        result.append(target_node)
                        frontier.append((relation.target_id, current_depth + 1))

            # Get incoming relations
            for relation in self.get_relations_to(current_id):
                if relation_types and relation.relation_type not in relation_types:
                    continue
                if relation.source_id not in visited:
                    visited.add(relation.source_id)
                    source_node = self.nodes.get(relation.source_id)
                    if source_node:
                        result.append(source_node)
                        frontier.append((relation.source_id, current_depth + 1))

        return result

    def find_path(
        self,
        source: str,
        target: str,
    ) -> Optional[List[OntologyNode]]:
        """Find shortest path between two nodes."""
        source_node = self.get_node(source) or self.get_node_by_name(source)
        target_node = self.get_node(target) or self.get_node_by_name(target)

        if not source_node or not target_node:
            return None

        # BFS to find path
        visited: Set[str] = {source_node.node_id}
        queue: List[List[str]] = [[source_node.node_id]]

        while queue:
            path = queue.pop(0)
            current_id = path[-1]

            if current_id == target_node.node_id:
                return [self.nodes[nid] for nid in path]

            # Explore neighbors
            for relation in self.get_relations_from(current_id):
                if relation.target_id not in visited:
                    visited.add(relation.target_id)
                    queue.append(path + [relation.target_id])

            for relation in self.get_relations_to(current_id):
                if relation.source_id not in visited:
                    visited.add(relation.source_id)
                    queue.append(path + [relation.source_id])

        return None

    def get_context_for_concept(
        self,
        concept: str,
        include_related: bool = True,
        depth: int = 1,
    ) -> str:
        """
        Generate context string for RAG pipeline enhancement.

        Args:
            concept: Concept to get context for
            include_related: Whether to include related concepts
            depth: Depth for related concept search

        Returns:
            Context string for RAG augmentation
        """
        node = self.get_node(concept) or self.get_node_by_name(concept)
        if not node:
            return ""

        context_parts = [
            f"{node.name}: {node.description}",
        ]

        # Add properties
        for key, value in node.properties.items():
            context_parts.append(f"  - {key}: {value}")

        if include_related:
            related = self.find_related_concepts(concept, depth=depth)
            if related:
                context_parts.append("\nRelated concepts:")
                for rel_node in related[:10]:  # Limit for context size
                    relations = self.get_relations_from(node.node_id)
                    rel_types = [
                        r.relation_type.value
                        for r in relations
                        if r.target_id == rel_node.node_id
                    ]
                    rel_str = ", ".join(rel_types) if rel_types else "related"
                    context_parts.append(
                        f"  - {rel_node.name} ({rel_str}): {rel_node.description[:100]}"
                    )

        return "\n".join(context_parts)

    def get_statistics(self) -> OntologyStatistics:
        """Calculate ontology statistics."""
        stats = OntologyStatistics(
            total_nodes=len(self.nodes),
            total_relations=len(self.relations),
        )

        # Count by type
        for node in self.nodes.values():
            type_key = node.node_type.value
            stats.nodes_by_type[type_key] = stats.nodes_by_type.get(type_key, 0) + 1

        for relation in self.relations.values():
            type_key = relation.relation_type.value
            stats.relations_by_type[type_key] = (
                stats.relations_by_type.get(type_key, 0) + 1
            )

        # Average relations per node
        if stats.total_nodes > 0:
            stats.avg_relations_per_node = stats.total_relations / stats.total_nodes

        # Calculate max depth
        stats.max_depth = self._calculate_max_depth()

        return stats

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of ontology hierarchy."""
        max_depth = 0

        # Find root nodes (no parent)
        roots = [n for n in self.nodes.values() if n.parent_id is None]

        def get_depth(node_id: str, current_depth: int) -> int:
            children = self._parent_index.get(node_id, [])
            if not children:
                return current_depth
            return max(get_depth(child_id, current_depth + 1) for child_id in children)

        for root in roots:
            depth = get_depth(root.node_id, 1)
            max_depth = max(max_depth, depth)

        return max_depth

    def export_json(self, output_path: str) -> str:
        """Export ontology to JSON file."""
        data = {
            "name": self.name,
            "created_at": self.created_at,
            "statistics": self.get_statistics().to_dict(),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported ontology to {output_path}")
        return output_path

    def export_rdf(self) -> str:
        """
        Export ontology in RDF/Turtle format.

        Returns:
            RDF Turtle string
        """
        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            f"@prefix : <http://example.org/{self.name}#> .",
            "",
        ]

        # Export nodes as classes/individuals
        for node in self.nodes.values():
            node_uri = f":{node.name.replace(' ', '_')}"
            lines.append(f"{node_uri} a owl:Class ;")
            lines.append(f'    rdfs:label "{node.name}" ;')
            lines.append(f'    rdfs:comment "{node.description}" .')
            lines.append("")

        # Export relations
        for relation in self.relations.values():
            source = self.nodes.get(relation.source_id)
            target = self.nodes.get(relation.target_id)
            if source and target:
                source_uri = f":{source.name.replace(' ', '_')}"
                target_uri = f":{target.name.replace(' ', '_')}"
                rel_type = relation.relation_type.value.replace("_", "")
                lines.append(f"{source_uri} :{rel_type} {target_uri} .")

        return "\n".join(lines)

    def visualize_dot(self) -> str:
        """
        Generate DOT format for Graphviz visualization.

        Returns:
            DOT format string
        """
        lines = [
            "digraph Ontology {",
            "    rankdir=TB;",
            "    node [shape=box, style=rounded];",
            "",
        ]

        # Color scheme by node type
        colors = {
            NodeType.CATEGORY: "lightblue",
            NodeType.CONCEPT: "lightgreen",
            NodeType.ENTITY: "lightyellow",
            NodeType.ATTRIBUTE: "lightpink",
            NodeType.ACTION: "lightsalmon",
        }

        # Add nodes
        for node in self.nodes.values():
            color = colors.get(node.node_type, "white")
            label = f"{node.name}\\n({node.node_type.value})"
            lines.append(
                f'    "{node.node_id}" [label="{label}", fillcolor={color}, style=filled];'
            )

        lines.append("")

        # Add edges
        for relation in self.relations.values():
            style = "solid"
            if relation.confidence < 0.8:
                style = "dashed"
            lines.append(
                f'    "{relation.source_id}" -> "{relation.target_id}" '
                f'[label="{relation.relation_type.value}", style={style}];'
            )

        lines.append("}")
        return "\n".join(lines)

    def visualize_mermaid(self) -> str:
        """
        Generate Mermaid diagram for visualization.

        Returns:
            Mermaid diagram string
        """
        lines = ["```mermaid", "graph TD"]

        # Add nodes with styling
        for node in self.nodes.values():
            safe_id = node.node_id.replace("-", "_")
            if node.node_type == NodeType.CATEGORY:
                lines.append(f'    {safe_id}["{node.name}"]')
            elif node.node_type == NodeType.CONCEPT:
                lines.append(f'    {safe_id}("{node.name}")')
            else:
                lines.append(f'    {safe_id}{{"{node.name}"}}')

        # Add edges
        arrow_types = {
            RelationType.IS_A: "-->",
            RelationType.PART_OF: "-.->",
            RelationType.RELATES_TO: "---",
            RelationType.CONTAINS: "==>",
        }

        for relation in self.relations.values():
            source_id = relation.source_id.replace("-", "_")
            target_id = relation.target_id.replace("-", "_")
            arrow = arrow_types.get(relation.relation_type, "-->")
            lines.append(
                f"    {source_id} {arrow}|{relation.relation_type.value}| {target_id}"
            )

        lines.append("```")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, input_path: str) -> "KnowledgeOntology":
        """Load ontology from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ontology = cls(name=data.get("name", "imported"), auto_init_core=False)

        # Load nodes
        for node_data in data.get("nodes", []):
            node = OntologyNode.from_dict(node_data)
            ontology.nodes[node.node_id] = node
            ontology._name_index[node.name.lower()] = node.node_id
            if node.parent_id:
                ontology._parent_index[node.parent_id].append(node.node_id)

        # Load relations
        for rel_data in data.get("relations", []):
            relation = OntologyRelation.from_dict(rel_data)
            ontology.relations[relation.relation_id] = relation
            ontology._relations_from[relation.source_id].append(relation.relation_id)
            ontology._relations_to[relation.target_id].append(relation.relation_id)

        logger.info(f"Loaded ontology from {input_path}")
        return ontology
