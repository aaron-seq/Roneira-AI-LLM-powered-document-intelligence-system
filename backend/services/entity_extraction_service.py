"""
Entity Extraction and Relationship Mapping Service.

Provides named entity recognition (NER) and relationship
extraction between entities in documents.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CUSTOM = "CUSTOM"


@dataclass
class Entity:
    """An extracted entity.
    
    Attributes:
        text: Entity text as found in document.
        normalized: Normalized/canonical form.
        entity_type: Type of entity.
        confidence: Extraction confidence (0-1).
        start_pos: Start position in original text.
        end_pos: End position in original text.
        metadata: Additional entity metadata.
    """
    text: str
    normalized: str
    entity_type: EntityType
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "normalized": self.normalized,
            "type": self.entity_type.value,
            "confidence": self.confidence,
            "position": {"start": self.start_pos, "end": self.end_pos},
            "metadata": self.metadata,
        }


@dataclass
class Relationship:
    """A relationship between two entities.
    
    Attributes:
        subject: Subject entity.
        relation: Relationship type/label.
        object: Object entity.
        confidence: Extraction confidence.
        context: Text snippet where relationship was found.
    """
    subject: Entity
    relation: str
    object: Entity
    confidence: float = 1.0
    context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject.to_dict(),
            "relation": self.relation,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
            "context": self.context,
        }


class PatternEntityExtractor:
    """Regex-based entity extraction."""
    
    def __init__(self):
        """Initialize pattern extractor."""
        self._patterns = {
            EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            EntityType.PHONE: r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            EntityType.URL: r'https?://[^\s<>"{}|\\^`\[\]]+',
            EntityType.DATE: r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
            EntityType.MONEY: r'[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?|euros?)',
            EntityType.PERCENT: r'\b\d+(?:\.\d+)?%\b',
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities using patterns.
        
        Args:
            text: Text to extract from.
            
        Returns:
            List of extracted entities.
        """
        entities = []
        
        for entity_type, pattern in self._patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    normalized=self._normalize(match.group(), entity_type),
                    entity_type=entity_type,
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
        
        return entities
    
    def _normalize(self, text: str, entity_type: EntityType) -> str:
        """Normalize entity text."""
        if entity_type == EntityType.EMAIL:
            return text.lower()
        if entity_type == EntityType.PHONE:
            return re.sub(r'[^\d+]', '', text)
        if entity_type == EntityType.URL:
            return text.lower()
        return text


class NEREntityExtractor:
    """NER-based entity extraction using ML models."""
    
    def __init__(self, use_spacy: bool = True):
        """Initialize NER extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NER.
        """
        self._nlp = None
        self._use_spacy = use_spacy
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Load NER model."""
        if self._initialized:
            return True
        
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model...")
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
            
            self._initialized = True
            logger.info("NER model loaded successfully")
            return True
            
        except ImportError:
            logger.warning("spaCy not installed. Using pattern-based extraction only.")
            return False
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            return False
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities using NER.
        
        Args:
            text: Text to extract from.
            
        Returns:
            List of extracted entities.
        """
        if not self._nlp:
            return []
        
        doc = self._nlp(text[:100000])  # Limit text size
        entities = []
        
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
        }
        
        for ent in doc.ents:
            if ent.label_ in type_mapping:
                entity = Entity(
                    text=ent.text,
                    normalized=ent.text.strip(),
                    entity_type=type_mapping[ent.label_],
                    confidence=0.85,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                )
                entities.append(entity)
        
        return entities


class RelationshipExtractor:
    """Extract relationships between entities."""
    
    def __init__(self):
        """Initialize relationship extractor."""
        self._relation_patterns = [
            (r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)\s+(?:of|for|at)\s+(\w+)', 'is_X_of'),
            (r'(\w+)\s+(?:works?|worked)\s+(?:at|for)\s+(\w+)', 'works_at'),
            (r'(\w+)\s+(?:founded|started|created|established)\s+(\w+)', 'founded'),
            (r'(\w+)\s+(?:acquired|bought|purchased)\s+(\w+)', 'acquired'),
            (r'(\w+)\s+(?:merged|combined)\s+with\s+(\w+)', 'merged_with'),
            (r'(\w+)\s+(?:is|was)\s+(?:located|based|headquartered)\s+(?:in|at)\s+(\w+)', 'located_in'),
        ]
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships between entities.
        
        Args:
            text: Original text.
            entities: Extracted entities.
            
        Returns:
            List of relationships.
        """
        relationships = []
        
        # Create entity lookup by position
        entity_positions = {}
        for ent in entities:
            entity_positions[(ent.start_pos, ent.end_pos)] = ent
        
        # Extract co-occurring entities in sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sent in sentences:
            sent_entities = [
                e for e in entities
                if e.text.lower() in sent.lower()
            ]
            
            # Find entity pairs
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    relation = self._detect_relation(sent, ent1, ent2)
                    if relation:
                        relationships.append(Relationship(
                            subject=ent1,
                            relation=relation,
                            object=ent2,
                            confidence=0.7,
                            context=sent[:200]
                        ))
        
        return relationships
    
    def _detect_relation(
        self,
        context: str,
        entity1: Entity,
        entity2: Entity
    ) -> Optional[str]:
        """Detect relationship type between two entities."""
        context_lower = context.lower()
        
        # Check for pattern matches
        for pattern, relation_type in self._relation_patterns:
            if re.search(pattern, context_lower):
                return relation_type
        
        # Default relationship based on entity types
        if entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.ORGANIZATION:
            return "associated_with"
        
        if entity1.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION) and entity2.entity_type == EntityType.LOCATION:
            return "related_to_location"
        
        return None


class EntityExtractionService:
    """Main entity extraction and relationship mapping service.
    
    Combines multiple extraction methods for comprehensive
    entity and relationship extraction.
    """
    
    def __init__(self, use_ner: bool = True):
        """Initialize entity extraction service.
        
        Args:
            use_ner: Whether to use NER model.
        """
        self.pattern_extractor = PatternEntityExtractor()
        self.ner_extractor = NEREntityExtractor() if use_ner else None
        self.relationship_extractor = RelationshipExtractor()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize extraction models."""
        if self._initialized:
            return
        
        if self.ner_extractor:
            await self.ner_extractor.initialize()
        
        self._initialized = True
    
    async def extract_entities(
        self,
        text: str,
        include_patterns: bool = True,
        include_ner: bool = True
    ) -> List[Entity]:
        """Extract all entities from text.
        
        Args:
            text: Text to extract from.
            include_patterns: Include pattern-based extraction.
            include_ner: Include NER extraction.
            
        Returns:
            List of deduplicated entities.
        """
        await self.initialize()
        
        all_entities = []
        
        if include_patterns:
            all_entities.extend(self.pattern_extractor.extract(text))
        
        if include_ner and self.ner_extractor:
            all_entities.extend(self.ner_extractor.extract(text))
        
        # Deduplicate
        return self._deduplicate_entities(all_entities)
    
    async def extract_with_relationships(
        self,
        text: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and their relationships.
        
        Args:
            text: Text to process.
            
        Returns:
            Tuple of (entities, relationships).
        """
        entities = await self.extract_entities(text)
        relationships = self.relationship_extractor.extract_relationships(text, entities)
        
        return entities, relationships
    
    def build_entity_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """Build entity relationship graph.
        
        Args:
            entities: Extracted entities.
            relationships: Extracted relationships.
            
        Returns:
            Graph structure with nodes and edges.
        """
        nodes = []
        edges = []
        
        # Create nodes
        entity_map = {}
        for i, entity in enumerate(entities):
            node_id = f"entity_{i}"
            entity_map[id(entity)] = node_id
            nodes.append({
                "id": node_id,
                "label": entity.normalized,
                "type": entity.entity_type.value,
                "confidence": entity.confidence
            })
        
        # Create edges
        for rel in relationships:
            subject_id = entity_map.get(id(rel.subject))
            object_id = entity_map.get(id(rel.object))
            
            if subject_id and object_id:
                edges.append({
                    "source": subject_id,
                    "target": object_id,
                    "relation": rel.relation,
                    "confidence": rel.confidence
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def _deduplicate_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.normalized.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def get_entity_summary(
        self,
        entities: List[Entity]
    ) -> Dict[str, List[str]]:
        """Get summary of entities by type.
        
        Args:
            entities: List of entities.
            
        Returns:
            Dict mapping entity types to lists of entity texts.
        """
        summary = defaultdict(list)
        
        for entity in entities:
            summary[entity.entity_type.value].append(entity.normalized)
        
        return dict(summary)


# Global instance
_entity_service: Optional[EntityExtractionService] = None


async def get_entity_service() -> EntityExtractionService:
    """Get the global entity extraction service."""
    global _entity_service
    if _entity_service is None:
        _entity_service = EntityExtractionService()
        await _entity_service.initialize()
    return _entity_service
