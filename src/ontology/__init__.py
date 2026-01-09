"""
Ontology Package

Provides knowledge ontology creation, management,
and visualization for AI document intelligence systems.
"""

from .knowledge_ontology import (
    KnowledgeOntology,
    OntologyNode,
    OntologyRelation,
    NodeType,
    RelationType,
)

__all__ = [
    "KnowledgeOntology",
    "OntologyNode",
    "OntologyRelation",
    "NodeType",
    "RelationType",
]
