"""
Data Source Discovery & Catalog System

Comprehensive data source identification, profiling, and cataloging
for AI system grounding, training, and validation.

This module implements Task 1 of the AI Developer Roadmap:
- Data source discovery across multiple formats
- Automated schema detection
- Data quality assessment
- Comprehensive catalog generation
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import mimetypes

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Types of data sources supported by the catalog."""

    STRUCTURED = "structured"
    SEMI_STRUCTURED = "semi_structured"
    UNSTRUCTURED = "unstructured"


class DataFormat(str, Enum):
    """Supported data formats."""

    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    XLSX = "xlsx"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class DataSource:
    """Represents a discovered data source with metadata."""

    source_id: str
    name: str
    source_type: SourceType
    format: DataFormat
    location: str
    size_bytes: int
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    utility_description: str = ""
    tags: List[str] = field(default_factory=list)
    lineage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["source_type"] = self.source_type.value
        result["format"] = self.format.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSource":
        """Create from dictionary."""
        data["source_type"] = SourceType(data["source_type"])
        data["format"] = DataFormat(data["format"])
        return cls(**data)


@dataclass
class CatalogStatistics:
    """Statistics about the data catalog."""

    total_sources: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_format: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    avg_quality_score: float = 0.0
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataSourceCatalog:
    """
    Comprehensive catalog for data source discovery and management.

    This class implements the core functionality for Task 1:
    Data Source Discovery of the AI Developer Roadmap.

    Features:
    - Multi-source discovery (PDF, Web, API, Database)
    - Automated format and schema detection
    - Data quality profiling
    - Source relationship mapping
    - Catalog export and visualization

    Example:
        catalog = DataSourceCatalog()
        sources = catalog.discover_sources(["./data/documents"])
        report = catalog.generate_catalog_report()
        catalog.export_catalog("./catalog.json")
    """

    # Format detection mapping
    FORMAT_EXTENSIONS = {
        ".pdf": DataFormat.PDF,
        ".json": DataFormat.JSON,
        ".csv": DataFormat.CSV,
        ".txt": DataFormat.TXT,
        ".html": DataFormat.HTML,
        ".htm": DataFormat.HTML,
        ".md": DataFormat.MARKDOWN,
        ".markdown": DataFormat.MARKDOWN,
        ".docx": DataFormat.DOCX,
        ".xlsx": DataFormat.XLSX,
        ".xls": DataFormat.XLSX,
        ".xml": DataFormat.XML,
    }

    # Type classification by format
    TYPE_BY_FORMAT = {
        DataFormat.CSV: SourceType.STRUCTURED,
        DataFormat.XLSX: SourceType.STRUCTURED,
        DataFormat.JSON: SourceType.SEMI_STRUCTURED,
        DataFormat.XML: SourceType.SEMI_STRUCTURED,
        DataFormat.HTML: SourceType.SEMI_STRUCTURED,
        DataFormat.PDF: SourceType.UNSTRUCTURED,
        DataFormat.TXT: SourceType.UNSTRUCTURED,
        DataFormat.MARKDOWN: SourceType.UNSTRUCTURED,
        DataFormat.DOCX: SourceType.UNSTRUCTURED,
    }

    def __init__(self, catalog_name: str = "default"):
        """
        Initialize the data source catalog.

        Args:
            catalog_name: Name identifier for this catalog
        """
        self.catalog_name = catalog_name
        self.sources: Dict[str, DataSource] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self._profiler = None
        logger.info(f"Initialized DataSourceCatalog: {catalog_name}")

    def _generate_source_id(self, location: str) -> str:
        """Generate unique source ID from location."""
        return hashlib.md5(location.encode()).hexdigest()[:12]

    def _detect_format(self, file_path: str) -> DataFormat:
        """Detect data format from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.FORMAT_EXTENSIONS.get(ext, DataFormat.UNKNOWN)

    def _detect_type(self, format: DataFormat) -> SourceType:
        """Determine source type based on format."""
        return self.TYPE_BY_FORMAT.get(format, SourceType.UNSTRUCTURED)

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata."""
        path = Path(file_path)
        stat = path.stat()
        return {
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def discover_sources(
        self,
        paths: List[str],
        recursive: bool = True,
        include_hidden: bool = False,
    ) -> List[DataSource]:
        """
        Discover and catalog data sources from specified paths.

        Args:
            paths: List of file or directory paths to scan
            recursive: Whether to scan subdirectories
            include_hidden: Whether to include hidden files

        Returns:
            List of discovered DataSource objects
        """
        discovered = []

        for path_str in paths:
            path = Path(path_str)

            if not path.exists():
                logger.warning(f"Path does not exist: {path_str}")
                continue

            if path.is_file():
                source = self._catalog_file(path)
                if source:
                    discovered.append(source)
            elif path.is_dir():
                pattern = "**/*" if recursive else "*"
                for file_path in path.glob(pattern):
                    if file_path.is_file():
                        if not include_hidden and file_path.name.startswith("."):
                            continue
                        source = self._catalog_file(file_path)
                        if source:
                            discovered.append(source)

        logger.info(f"Discovered {len(discovered)} data sources")
        return discovered

    def _catalog_file(self, file_path: Path) -> Optional[DataSource]:
        """Catalog a single file."""
        try:
            format = self._detect_format(str(file_path))
            if format == DataFormat.UNKNOWN:
                return None

            metadata = self._get_file_metadata(str(file_path))
            source_id = self._generate_source_id(str(file_path.absolute()))

            source = DataSource(
                source_id=source_id,
                name=file_path.name,
                source_type=self._detect_type(format),
                format=format,
                location=str(file_path.absolute()),
                size_bytes=metadata["size_bytes"],
                created_at=metadata["created_at"],
                modified_at=metadata["modified_at"],
                utility_description=f"Document: {file_path.name}",
            )

            # Add to catalog
            self.sources[source_id] = source
            return source

        except Exception as e:
            logger.error(f"Error cataloging {file_path}: {e}")
            return None

    def add_source(self, source: DataSource) -> None:
        """Manually add a data source to the catalog."""
        self.sources[source.source_id] = source
        logger.info(f"Added source: {source.name}")

    def remove_source(self, source_id: str) -> bool:
        """Remove a source from the catalog."""
        if source_id in self.sources:
            del self.sources[source_id]
            logger.info(f"Removed source: {source_id}")
            return True
        return False

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """Get a specific source by ID."""
        return self.sources.get(source_id)

    def search_sources(
        self,
        query: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        format: Optional[DataFormat] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[DataSource]:
        """
        Search sources with filters.

        Args:
            query: Text search in name/description
            source_type: Filter by source type
            format: Filter by data format
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes

        Returns:
            Filtered list of DataSource objects
        """
        results = list(self.sources.values())

        if query:
            query_lower = query.lower()
            results = [
                s
                for s in results
                if query_lower in s.name.lower()
                or query_lower in s.utility_description.lower()
            ]

        if source_type:
            results = [s for s in results if s.source_type == source_type]

        if format:
            results = [s for s in results if s.format == format]

        if min_size is not None:
            results = [s for s in results if s.size_bytes >= min_size]

        if max_size is not None:
            results = [s for s in results if s.size_bytes <= max_size]

        return results

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Define a relationship between two sources."""
        self.relationships.append(
            {
                "source": source_id,
                "target": target_id,
                "type": relationship_type,
                "metadata": metadata or {},
            }
        )

    def get_statistics(self) -> CatalogStatistics:
        """Calculate catalog statistics."""
        by_type: Dict[str, int] = {}
        by_format: Dict[str, int] = {}
        total_size = 0
        quality_scores = []

        for source in self.sources.values():
            # Count by type
            type_key = source.source_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            # Count by format
            format_key = source.format.value
            by_format[format_key] = by_format.get(format_key, 0) + 1

            # Sum sizes
            total_size += source.size_bytes

            # Collect quality scores
            if "overall" in source.quality_metrics:
                quality_scores.append(source.quality_metrics["overall"])

        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        return CatalogStatistics(
            total_sources=len(self.sources),
            by_type=by_type,
            by_format=by_format,
            total_size_bytes=total_size,
            avg_quality_score=avg_quality,
            last_updated=datetime.now().isoformat(),
        )

    def generate_catalog_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive catalog report.

        Returns:
            Dictionary containing full catalog report with:
            - Catalog metadata
            - Statistics
            - All sources
            - Relationships
        """
        stats = self.get_statistics()

        return {
            "catalog_name": self.catalog_name,
            "created_at": self.created_at,
            "statistics": stats.to_dict(),
            "sources": [s.to_dict() for s in self.sources.values()],
            "relationships": self.relationships,
            "summary": {
                "total_documents": stats.total_sources,
                "total_size_mb": round(stats.total_size_bytes / (1024 * 1024), 2),
                "types_distribution": stats.by_type,
                "formats_distribution": stats.by_format,
            },
        }

    def export_catalog(
        self,
        output_path: str,
        format: Literal["json", "csv"] = "json",
    ) -> str:
        """
        Export catalog to file.

        Args:
            output_path: Path for output file
            format: Export format (json or csv)

        Returns:
            Path to exported file
        """
        report = self.generate_catalog_report()

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if self.sources:
                    writer = csv.DictWriter(
                        f, fieldnames=list(self.sources.values())[0].to_dict().keys()
                    )
                    writer.writeheader()
                    for source in self.sources.values():
                        writer.writerow(source.to_dict())

        logger.info(f"Exported catalog to: {output_path}")
        return output_path

    def visualize_catalog(self) -> str:
        """
        Generate visualization of the catalog structure.

        Returns:
            Mermaid diagram string for visualization
        """
        stats = self.get_statistics()

        mermaid = ["```mermaid", "pie showData"]
        mermaid.append('    title "Data Sources by Type"')

        for type_name, count in stats.by_type.items():
            mermaid.append(f'    "{type_name}": {count}')

        mermaid.append("```")
        return "\n".join(mermaid)

    def generate_lineage_graph(self) -> str:
        """
        Generate data lineage graph in DOT format.

        Returns:
            DOT format string for graph visualization
        """
        lines = ["digraph DataLineage {", "    rankdir=LR;"]

        # Add source nodes
        for source in self.sources.values():
            label = f"{source.name}\\n({source.format.value})"
            shape = "box" if source.source_type == SourceType.STRUCTURED else "ellipse"
            lines.append(f'    "{source.source_id}" [label="{label}" shape={shape}];')

        # Add relationships
        for rel in self.relationships:
            lines.append(
                f'    "{rel["source"]}" -> "{rel["target"]}" [label="{rel["type"]}"];'
            )

        lines.append("}")
        return "\n".join(lines)
