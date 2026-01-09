"""
Tests for Ontology Module

Unit tests for Task 3: Prompt & Ontology Creation
"""

import pytest
import json
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontology import (
    KnowledgeOntology,
    OntologyNode,
    OntologyRelation,
    NodeType,
    RelationType,
)


class TestKnowledgeOntology:
    """Tests for KnowledgeOntology class."""

    def test_ontology_initialization(self):
        """Test ontology initializes with core concepts."""
        ontology = KnowledgeOntology("test")

        assert ontology.name == "test"
        assert len(ontology.nodes) > 0  # Core concepts initialized

    def test_ontology_initialization_without_core(self):
        """Test ontology without core concepts."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        assert ontology.name == "test"
        assert len(ontology.nodes) == 0

    def test_add_node(self):
        """Test adding a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        node = ontology.add_node(
            name="Test Concept",
            node_type=NodeType.CONCEPT,
            description="A test concept",
        )

        assert node.name == "Test Concept"
        assert node.node_type == NodeType.CONCEPT
        assert node.node_id in ontology.nodes

    def test_add_entity(self):
        """Test adding an entity."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        entity = ontology.add_entity(
            name="Machine Learning",
            description="ML subset of AI",
        )

        assert entity.node_type == NodeType.ENTITY

    def test_add_concept(self):
        """Test adding a concept with parent."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        parent = ontology.add_category("Technology", "Tech category")
        child = ontology.add_concept(
            "AI", "Artificial Intelligence", parent_name="Technology"
        )

        assert child.parent_id == parent.node_id

    def test_get_node(self):
        """Test getting node by ID."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        node = ontology.add_entity("Test", "Description")
        retrieved = ontology.get_node(node.node_id)

        assert retrieved == node
        assert ontology.get_node("nonexistent") is None

    def test_get_node_by_name(self):
        """Test getting node by name."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ontology.add_entity("Machine Learning", "ML")
        node = ontology.get_node_by_name("machine learning")  # Case insensitive

        assert node is not None
        assert node.name == "Machine Learning"

    def test_update_node(self):
        """Test updating a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        node = ontology.add_entity("Test", "Old description")
        updated = ontology.update_node(node.node_id, description="New description")

        assert updated.description == "New description"

    def test_remove_node(self):
        """Test removing a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        node = ontology.add_entity("Test", "Description")
        assert ontology.remove_node(node.node_id)
        assert node.node_id not in ontology.nodes
        assert not ontology.remove_node("nonexistent")

    def test_add_relation(self):
        """Test adding a relation."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("Machine Learning", "ML")
        ai = ontology.add_entity("Artificial Intelligence", "AI")

        relation = ontology.add_relation(
            source_id=ml.node_id,
            target_id=ai.node_id,
            relation_type=RelationType.IS_A,
        )

        assert relation.source_id == ml.node_id
        assert relation.target_id == ai.node_id
        assert relation.relation_type == RelationType.IS_A

    def test_get_relations_from(self):
        """Test getting relations from a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("Machine Learning", "ML")
        ai = ontology.add_entity("Artificial Intelligence", "AI")
        ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)

        relations = ontology.get_relations_from(ml.node_id)

        assert len(relations) == 1
        assert relations[0].target_id == ai.node_id

    def test_get_relations_to(self):
        """Test getting relations to a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("Machine Learning", "ML")
        ai = ontology.add_entity("Artificial Intelligence", "AI")
        ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)

        relations = ontology.get_relations_to(ai.node_id)

        assert len(relations) == 1
        assert relations[0].source_id == ml.node_id

    def test_get_children(self):
        """Test getting children of a node."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        parent = ontology.add_category("Technology", "Tech")
        child1 = ontology.add_concept("AI", "AI", parent_name="Technology")
        child2 = ontology.add_concept("ML", "ML", parent_name="Technology")

        children = ontology.get_children(parent.node_id)

        assert len(children) == 2

    def test_get_hierarchy(self):
        """Test getting hierarchy tree."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        root = ontology.add_category("Root", "Root node")
        child = ontology.add_node("Child", NodeType.CONCEPT, parent_id=root.node_id)

        hierarchy = ontology.get_hierarchy(root.node_id)

        assert hierarchy["name"] == "Root"
        assert len(hierarchy["children"]) == 1

    def test_find_related_concepts(self):
        """Test finding related concepts."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("Machine Learning", "ML")
        ai = ontology.add_entity("Artificial Intelligence", "AI")
        dl = ontology.add_entity("Deep Learning", "DL")

        ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)
        ontology.add_relation(dl.node_id, ml.node_id, RelationType.IS_A)

        related = ontology.find_related_concepts("Machine Learning", depth=2)

        assert len(related) >= 2  # AI and DL

    def test_find_path(self):
        """Test finding path between nodes."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        a = ontology.add_entity("A", "Node A")
        b = ontology.add_entity("B", "Node B")
        c = ontology.add_entity("C", "Node C")

        ontology.add_relation(a.node_id, b.node_id, RelationType.RELATES_TO)
        ontology.add_relation(b.node_id, c.node_id, RelationType.RELATES_TO)

        path = ontology.find_path("A", "C")

        assert path is not None
        assert len(path) == 3

    def test_get_context_for_concept(self):
        """Test generating RAG context."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity(
            "Machine Learning",
            "ML is a subset of AI enabling systems to learn from data",
        )

        context = ontology.get_context_for_concept("Machine Learning")

        assert "Machine Learning" in context
        assert "subset of AI" in context

    def test_get_statistics(self):
        """Test getting ontology statistics."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ontology.add_entity("E1", "Entity 1")
        ontology.add_concept("C1", "Concept 1")

        stats = ontology.get_statistics()

        assert stats.total_nodes == 2
        assert NodeType.ENTITY.value in stats.nodes_by_type
        assert NodeType.CONCEPT.value in stats.nodes_by_type

    def test_export_json(self, tmp_path):
        """Test exporting to JSON."""
        ontology = KnowledgeOntology("test", auto_init_core=False)
        ontology.add_entity("Test", "Description")

        output_path = str(tmp_path / "ontology.json")
        ontology.export_json(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["name"] == "test"
        assert len(data["nodes"]) == 1

    def test_export_rdf(self):
        """Test exporting to RDF format."""
        ontology = KnowledgeOntology("test", auto_init_core=False)
        ontology.add_entity("Machine Learning", "ML")

        rdf = ontology.export_rdf()

        assert "@prefix" in rdf
        assert "Machine_Learning" in rdf

    def test_visualize_dot(self):
        """Test DOT visualization export."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("ML", "Machine Learning")
        ai = ontology.add_entity("AI", "Artificial Intelligence")
        ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)

        dot = ontology.visualize_dot()

        assert "digraph" in dot
        assert "is_a" in dot

    def test_visualize_mermaid(self):
        """Test Mermaid visualization export."""
        ontology = KnowledgeOntology("test", auto_init_core=False)

        ml = ontology.add_entity("ML", "Machine Learning")
        ai = ontology.add_entity("AI", "Artificial Intelligence")
        ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)

        mermaid = ontology.visualize_mermaid()

        assert "mermaid" in mermaid
        assert "graph TD" in mermaid

    def test_from_json(self, tmp_path):
        """Test loading from JSON."""
        # Create and export
        ontology = KnowledgeOntology("test", auto_init_core=False)
        node = ontology.add_entity("Test", "Description")

        output_path = str(tmp_path / "ontology.json")
        ontology.export_json(output_path)

        # Load
        loaded = KnowledgeOntology.from_json(output_path)

        assert loaded.name == "test"
        assert len(loaded.nodes) == 1


class TestOntologyNode:
    """Tests for OntologyNode dataclass."""

    def test_node_creation(self):
        """Test creating a node."""
        node = OntologyNode(
            node_id="test_001",
            name="Test Node",
            node_type=NodeType.ENTITY,
            description="A test node",
        )

        assert node.node_id == "test_001"
        assert node.node_type == NodeType.ENTITY

    def test_to_dict(self):
        """Test converting to dictionary."""
        node = OntologyNode(
            node_id="test_001",
            name="Test Node",
            node_type=NodeType.CONCEPT,
            description="A test node",
        )

        d = node.to_dict()

        assert d["node_type"] == "concept"

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "node_id": "test_001",
            "name": "Test Node",
            "node_type": "entity",
            "description": "A test node",
            "properties": {},
            "parent_id": None,
            "aliases": [],
            "created_at": "2024-01-01",
        }

        node = OntologyNode.from_dict(d)

        assert node.node_type == NodeType.ENTITY


class TestOntologyRelation:
    """Tests for OntologyRelation dataclass."""

    def test_relation_creation(self):
        """Test creating a relation."""
        relation = OntologyRelation(
            relation_id="rel_001",
            source_id="src_001",
            target_id="tgt_001",
            relation_type=RelationType.IS_A,
        )

        assert relation.relation_type == RelationType.IS_A
        assert relation.confidence == 1.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        relation = OntologyRelation(
            relation_id="rel_001",
            source_id="src_001",
            target_id="tgt_001",
            relation_type=RelationType.PART_OF,
        )

        d = relation.to_dict()

        assert d["relation_type"] == "part_of"
