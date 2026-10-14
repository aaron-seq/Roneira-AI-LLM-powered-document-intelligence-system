"""
Ontology Package

Provides knowledge ontology creation, management,
and visualization for AI document intelligence systems.
"""

from .knowledge_ontology import (
    KnowledgeOntology,
    NodeType,
    OntologyNode,
    OntologyRelation,
    RelationType,
)

__all__ = [
    "KnowledgeOntology",
    "NodeType",
    "OntologyNode",
    "OntologyRelation",
    "RelationType",
]
