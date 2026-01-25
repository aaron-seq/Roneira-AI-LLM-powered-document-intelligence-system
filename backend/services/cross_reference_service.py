"""
Multi-Document Cross-Referencing Service.

Provides cross-document analysis to find relationships,
shared entities, and contextual connections between documents.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentReference:
    """A reference or connection between documents.
    
    Attributes:
        source_doc_id: Source document ID.
        target_doc_id: Target document ID.
        reference_type: Type of reference (entity, topic, citation).
        confidence: Confidence score (0-1).
        context: Context snippet where reference was found.
        shared_entities: Entities shared between documents.
    """
    source_doc_id: str
    target_doc_id: str
    reference_type: str
    confidence: float
    context: str = ""
    shared_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_doc_id": self.source_doc_id,
            "target_doc_id": self.target_doc_id,
            "reference_type": self.reference_type,
            "confidence": self.confidence,
            "context": self.context,
            "shared_entities": self.shared_entities,
        }


@dataclass
class DocumentCluster:
    """A cluster of related documents.
    
    Attributes:
        cluster_id: Unique cluster identifier.
        document_ids: Set of document IDs in the cluster.
        primary_topic: Primary topic of the cluster.
        keywords: Keywords describing the cluster.
        centroid: Cluster centroid embedding.
    """
    cluster_id: str
    document_ids: Set[str] = field(default_factory=set)
    primary_topic: str = ""
    keywords: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None


class CrossReferenceService:
    """Service for multi-document cross-referencing.
    
    Provides:
    - Entity-based document linking
    - Topic-based document clustering
    - Citation and reference extraction
    - Semantic similarity connections
    """
    
    def __init__(
        self,
        vector_store_service=None,
        embedding_service=None,
        similarity_threshold: float = 0.75
    ):
        """Initialize cross-reference service.
        
        Args:
            vector_store_service: Vector store for embeddings.
            embedding_service: Service for generating embeddings.
            similarity_threshold: Minimum similarity for linking.
        """
        self._vector_store = vector_store_service
        self._embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        
        # Document metadata cache
        self._doc_entities: Dict[str, Set[str]] = defaultdict(set)
        self._doc_topics: Dict[str, List[str]] = defaultdict(list)
        self._doc_embeddings: Dict[str, List[float]] = {}
        
        # Reference cache
        self._references: List[DocumentReference] = []
        self._clusters: Dict[str, DocumentCluster] = {}
    
    def register_document(
        self,
        document_id: str,
        entities: List[str],
        topics: List[str],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Register a document for cross-referencing.
        
        Args:
            document_id: Document identifier.
            entities: Extracted entities from document.
            topics: Topics/keywords from document.
            embedding: Optional document embedding.
        """
        self._doc_entities[document_id] = set(entities)
        self._doc_topics[document_id] = topics
        
        if embedding:
            self._doc_embeddings[document_id] = embedding
        
        logger.debug(
            f"Registered document {document_id} with "
            f"{len(entities)} entities, {len(topics)} topics"
        )
    
    def find_entity_links(
        self,
        document_id: str,
        min_shared: int = 2
    ) -> List[DocumentReference]:
        """Find documents linked by shared entities.
        
        Args:
            document_id: Source document to find links for.
            min_shared: Minimum shared entities required.
            
        Returns:
            List of document references.
        """
        if document_id not in self._doc_entities:
            return []
        
        source_entities = self._doc_entities[document_id]
        references = []
        
        for other_id, other_entities in self._doc_entities.items():
            if other_id == document_id:
                continue
            
            shared = source_entities & other_entities
            if len(shared) >= min_shared:
                # Calculate confidence based on overlap ratio
                overlap_ratio = len(shared) / min(
                    len(source_entities), len(other_entities)
                )
                
                ref = DocumentReference(
                    source_doc_id=document_id,
                    target_doc_id=other_id,
                    reference_type="entity",
                    confidence=min(overlap_ratio, 1.0),
                    shared_entities=list(shared)
                )
                references.append(ref)
        
        # Sort by confidence
        references.sort(key=lambda r: r.confidence, reverse=True)
        return references
    
    def find_topic_links(
        self,
        document_id: str,
        min_shared: int = 1
    ) -> List[DocumentReference]:
        """Find documents linked by shared topics.
        
        Args:
            document_id: Source document.
            min_shared: Minimum shared topics required.
            
        Returns:
            List of document references.
        """
        if document_id not in self._doc_topics:
            return []
        
        source_topics = set(self._doc_topics[document_id])
        references = []
        
        for other_id, other_topics in self._doc_topics.items():
            if other_id == document_id:
                continue
            
            other_set = set(other_topics)
            shared = source_topics & other_set
            
            if len(shared) >= min_shared:
                overlap_ratio = len(shared) / min(
                    len(source_topics), len(other_set)
                )
                
                ref = DocumentReference(
                    source_doc_id=document_id,
                    target_doc_id=other_id,
                    reference_type="topic",
                    confidence=min(overlap_ratio, 1.0),
                    shared_entities=list(shared)
                )
                references.append(ref)
        
        references.sort(key=lambda r: r.confidence, reverse=True)
        return references
    
    def find_semantic_links(
        self,
        document_id: str,
        top_k: int = 5
    ) -> List[DocumentReference]:
        """Find semantically similar documents.
        
        Args:
            document_id: Source document.
            top_k: Maximum number of links.
            
        Returns:
            List of document references.
        """
        if document_id not in self._doc_embeddings:
            return []
        
        source_embedding = self._doc_embeddings[document_id]
        references = []
        
        for other_id, other_embedding in self._doc_embeddings.items():
            if other_id == document_id:
                continue
            
            similarity = self._cosine_similarity(source_embedding, other_embedding)
            
            if similarity >= self.similarity_threshold:
                ref = DocumentReference(
                    source_doc_id=document_id,
                    target_doc_id=other_id,
                    reference_type="semantic",
                    confidence=similarity
                )
                references.append(ref)
        
        references.sort(key=lambda r: r.confidence, reverse=True)
        return references[:top_k]
    
    def find_all_links(
        self,
        document_id: str,
        include_entity: bool = True,
        include_topic: bool = True,
        include_semantic: bool = True
    ) -> List[DocumentReference]:
        """Find all types of links for a document.
        
        Args:
            document_id: Source document.
            include_entity: Include entity-based links.
            include_topic: Include topic-based links.
            include_semantic: Include semantic links.
            
        Returns:
            Combined list of references.
        """
        all_refs = []
        
        if include_entity:
            all_refs.extend(self.find_entity_links(document_id))
        
        if include_topic:
            all_refs.extend(self.find_topic_links(document_id))
        
        if include_semantic:
            all_refs.extend(self.find_semantic_links(document_id))
        
        # Deduplicate and merge references to same target
        merged = self._merge_references(all_refs)
        return merged
    
    def build_document_graph(self) -> Dict[str, Any]:
        """Build a graph of document relationships.
        
        Returns:
            Graph structure with nodes and edges.
        """
        nodes = []
        edges = []
        
        # Create nodes for all documents
        for doc_id in self._doc_entities.keys():
            nodes.append({
                "id": doc_id,
                "entities_count": len(self._doc_entities.get(doc_id, [])),
                "topics": self._doc_topics.get(doc_id, [])[:5]
            })
        
        # Create edges for all relationships
        processed = set()
        for doc_id in self._doc_entities.keys():
            refs = self.find_all_links(doc_id)
            for ref in refs:
                edge_key = tuple(sorted([ref.source_doc_id, ref.target_doc_id]))
                if edge_key not in processed:
                    edges.append({
                        "source": ref.source_doc_id,
                        "target": ref.target_doc_id,
                        "type": ref.reference_type,
                        "weight": ref.confidence
                    })
                    processed.add(edge_key)
        
        return {"nodes": nodes, "edges": edges}
    
    def cluster_documents(
        self,
        n_clusters: int = 5
    ) -> List[DocumentCluster]:
        """Cluster documents by semantic similarity.
        
        Args:
            n_clusters: Number of clusters.
            
        Returns:
            List of document clusters.
        """
        if len(self._doc_embeddings) < n_clusters:
            n_clusters = max(1, len(self._doc_embeddings))
        
        # Simple k-means clustering
        doc_ids = list(self._doc_embeddings.keys())
        embeddings = [self._doc_embeddings[d] for d in doc_ids]
        
        if not embeddings:
            return []
        
        clusters = self._simple_kmeans(embeddings, n_clusters)
        
        result = []
        for i, doc_indices in enumerate(clusters):
            cluster_doc_ids = {doc_ids[j] for j in doc_indices}
            
            # Extract common topics
            all_topics = []
            for doc_id in cluster_doc_ids:
                all_topics.extend(self._doc_topics.get(doc_id, []))
            
            from collections import Counter
            topic_counts = Counter(all_topics)
            top_keywords = [t for t, _ in topic_counts.most_common(5)]
            
            cluster = DocumentCluster(
                cluster_id=f"cluster_{i}",
                document_ids=cluster_doc_ids,
                primary_topic=top_keywords[0] if top_keywords else "",
                keywords=top_keywords
            )
            result.append(cluster)
        
        return result
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _merge_references(
        self,
        references: List[DocumentReference]
    ) -> List[DocumentReference]:
        """Merge references to same target document."""
        merged = {}
        
        for ref in references:
            key = ref.target_doc_id
            if key not in merged:
                merged[key] = ref
            else:
                # Combine reference types
                existing = merged[key]
                if ref.reference_type not in existing.reference_type:
                    existing.reference_type += f"+{ref.reference_type}"
                existing.confidence = max(existing.confidence, ref.confidence)
                existing.shared_entities = list(
                    set(existing.shared_entities) | set(ref.shared_entities)
                )
        
        return sorted(merged.values(), key=lambda r: r.confidence, reverse=True)
    
    def _simple_kmeans(
        self,
        embeddings: List[List[float]],
        k: int,
        max_iter: int = 100
    ) -> List[List[int]]:
        """Simple k-means clustering implementation."""
        import numpy as np
        import random
        
        data = np.array(embeddings)
        n = len(data)
        
        if n == 0:
            return []
        
        # Initialize centroids randomly
        indices = random.sample(range(n), min(k, n))
        centroids = data[indices]
        
        clusters = [[] for _ in range(k)]
        
        for _ in range(max_iter):
            # Assign points to clusters
            new_clusters = [[] for _ in range(k)]
            for i, point in enumerate(data):
                distances = [np.linalg.norm(point - c) for c in centroids]
                cluster_idx = np.argmin(distances)
                new_clusters[cluster_idx].append(i)
            
            # Check for convergence
            if new_clusters == clusters:
                break
            
            clusters = new_clusters
            
            # Update centroids
            for j in range(k):
                if clusters[j]:
                    centroids[j] = np.mean(data[clusters[j]], axis=0)
        
        return clusters
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cross-reference statistics."""
        return {
            "registered_documents": len(self._doc_entities),
            "documents_with_embeddings": len(self._doc_embeddings),
            "total_entities": sum(len(e) for e in self._doc_entities.values()),
            "total_topics": sum(len(t) for t in self._doc_topics.values()),
        }


# Global service instance
_cross_ref_service: Optional[CrossReferenceService] = None


def get_cross_reference_service() -> CrossReferenceService:
    """Get the global cross-reference service."""
    global _cross_ref_service
    if _cross_ref_service is None:
        _cross_ref_service = CrossReferenceService()
    return _cross_ref_service
