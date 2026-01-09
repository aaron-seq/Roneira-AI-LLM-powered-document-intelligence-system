"""
Tests for Data Catalog Module

Unit tests for Task 1: Data Source Discovery
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_catalog import (
    DataSourceCatalog,
    DataSource,
    SourceType,
    DataProfiler,
    DataQualityReport,
)
from src.data_catalog.data_source_catalog import DataFormat


class TestDataSourceCatalog:
    """Tests for DataSourceCatalog class."""

    def test_catalog_initialization(self):
        """Test catalog initializes correctly."""
        catalog = DataSourceCatalog("test_catalog")
        assert catalog.catalog_name == "test_catalog"
        assert len(catalog.sources) == 0

    def test_discover_sources_from_directory(self, tmp_path):
        """Test discovering sources from a directory."""
        # Create test files
        (tmp_path / "test.json").write_text('{"key": "value"}')
        (tmp_path / "data.csv").write_text("col1,col2\n1,2\n3,4")
        (tmp_path / "document.txt").write_text("Sample text content")

        catalog = DataSourceCatalog()
        sources = catalog.discover_sources([str(tmp_path)])

        assert len(sources) == 3
        assert any(s.format == DataFormat.JSON for s in sources)
        assert any(s.format == DataFormat.CSV for s in sources)
        assert any(s.format == DataFormat.TXT for s in sources)

    def test_discover_sources_with_recursive(self, tmp_path):
        """Test recursive discovery."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.json").write_text("{}")
        (tmp_path / "root.json").write_text("{}")

        catalog = DataSourceCatalog()

        # Non-recursive
        sources_flat = catalog.discover_sources([str(tmp_path)], recursive=False)
        assert len(sources_flat) == 1

        catalog = DataSourceCatalog()

        # Recursive
        sources_recursive = catalog.discover_sources([str(tmp_path)], recursive=True)
        assert len(sources_recursive) == 2

    def test_format_detection(self):
        """Test format detection by extension."""
        catalog = DataSourceCatalog()

        assert catalog._detect_format("test.pdf") == DataFormat.PDF
        assert catalog._detect_format("data.json") == DataFormat.JSON
        assert catalog._detect_format("table.csv") == DataFormat.CSV
        assert catalog._detect_format("doc.docx") == DataFormat.DOCX
        assert catalog._detect_format("unknown.xyz") == DataFormat.UNKNOWN

    def test_type_detection(self):
        """Test source type detection."""
        catalog = DataSourceCatalog()

        assert catalog._detect_type(DataFormat.CSV) == SourceType.STRUCTURED
        assert catalog._detect_type(DataFormat.JSON) == SourceType.SEMI_STRUCTURED
        assert catalog._detect_type(DataFormat.PDF) == SourceType.UNSTRUCTURED

    def test_add_source_manually(self):
        """Test adding a source manually."""
        catalog = DataSourceCatalog()

        source = DataSource(
            source_id="test_001",
            name="Test Source",
            source_type=SourceType.STRUCTURED,
            format=DataFormat.CSV,
            location="/path/to/file.csv",
            size_bytes=1024,
        )

        catalog.add_source(source)
        assert "test_001" in catalog.sources
        assert catalog.get_source("test_001") == source

    def test_remove_source(self):
        """Test removing a source."""
        catalog = DataSourceCatalog()

        source = DataSource(
            source_id="test_001",
            name="Test Source",
            source_type=SourceType.STRUCTURED,
            format=DataFormat.CSV,
            location="/path/to/file.csv",
            size_bytes=1024,
        )

        catalog.add_source(source)
        assert catalog.remove_source("test_001")
        assert "test_001" not in catalog.sources
        assert not catalog.remove_source("nonexistent")

    def test_search_sources(self, tmp_path):
        """Test searching sources with filters."""
        # Create test files
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        (tmp_path / "report.pdf").touch()
        (tmp_path / "config.json").write_text("{}")

        catalog = DataSourceCatalog()
        catalog.discover_sources([str(tmp_path)])

        # Search by type
        structured = catalog.search_sources(source_type=SourceType.STRUCTURED)
        assert len(structured) == 1

        # Search by format
        json_sources = catalog.search_sources(format=DataFormat.JSON)
        assert len(json_sources) == 1

        # Search by query
        data_sources = catalog.search_sources(query="data")
        assert len(data_sources) == 1

    def test_get_statistics(self, tmp_path):
        """Test calculating statistics."""
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        (tmp_path / "doc.pdf").touch()

        catalog = DataSourceCatalog()
        catalog.discover_sources([str(tmp_path)])

        stats = catalog.get_statistics()

        assert stats.total_sources == 2
        assert SourceType.STRUCTURED.value in stats.by_type
        assert SourceType.UNSTRUCTURED.value in stats.by_type

    def test_generate_catalog_report(self, tmp_path):
        """Test generating catalog report."""
        (tmp_path / "test.json").write_text('{"data": 1}')

        catalog = DataSourceCatalog("test")
        catalog.discover_sources([str(tmp_path)])

        report = catalog.generate_catalog_report()

        assert report["catalog_name"] == "test"
        assert "statistics" in report
        assert "sources" in report
        assert len(report["sources"]) == 1

    def test_export_catalog_json(self, tmp_path):
        """Test exporting catalog to JSON."""
        (tmp_path / "data.csv").write_text("a,b")

        catalog = DataSourceCatalog()
        catalog.discover_sources([str(tmp_path)])

        output_path = str(tmp_path / "catalog.json")
        catalog.export_catalog(output_path, format="json")

        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "sources" in data

    def test_add_relationship(self):
        """Test adding relationships between sources."""
        catalog = DataSourceCatalog()

        catalog.add_relationship(
            source_id="src_1",
            target_id="src_2",
            relationship_type="derived_from",
            metadata={"confidence": 0.9},
        )

        assert len(catalog.relationships) == 1
        assert catalog.relationships[0]["type"] == "derived_from"


class TestDataProfiler:
    """Tests for DataProfiler class."""

    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        profiler = DataProfiler()
        assert profiler.profiles_cache == {}

    def test_profile_csv_file(self, tmp_path):
        """Test profiling a CSV file."""
        csv_content = "name,age,email\nJohn,30,john@test.com\nJane,25,jane@test.com\n,28,bob@test.com"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        profiler = DataProfiler()
        report = profiler.profile_file(str(csv_file))

        assert report.row_count == 3
        assert report.column_count == 3
        assert report.completeness < 1.0  # Has one null
        assert len(report.column_profiles) == 3

    def test_profile_json_file(self, tmp_path):
        """Test profiling a JSON file."""
        json_content = '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'
        json_file = tmp_path / "test.json"
        json_file.write_text(json_content)

        profiler = DataProfiler()
        report = profiler.profile_file(str(json_file))

        assert report.row_count == 2
        assert report.column_count == 2
        assert report.validity == 1.0  # Valid JSON

    def test_profile_text_file(self, tmp_path):
        """Test profiling a text file."""
        text_content = (
            "This is a test document.\nIt has multiple lines.\nAnd some words."
        )
        text_file = tmp_path / "test.txt"
        text_file.write_text(text_content)

        profiler = DataProfiler()
        report = profiler.profile_file(str(text_file))

        assert report.row_count == 3  # Lines
        assert "word_count" in report.statistics
        assert report.statistics["word_count"] > 0

    def test_column_type_inference(self):
        """Test column type inference."""
        profiler = DataProfiler()

        assert profiler._infer_type(["1", "2", "3.5"]) == "numeric"
        assert profiler._infer_type(["true", "false", "true"]) == "boolean"
        assert profiler._infer_type(["2024-01-01", "2024-02-01"]) == "date"
        assert profiler._infer_type(["hello", "world"]) == "string"

    def test_quality_grade(self):
        """Test quality score to grade conversion."""
        profiler = DataProfiler()

        assert profiler._grade_quality(0.95) == "A"
        assert profiler._grade_quality(0.85) == "B"
        assert profiler._grade_quality(0.75) == "C"
        assert profiler._grade_quality(0.65) == "D"
        assert profiler._grade_quality(0.55) == "F"

    def test_timeliness_assessment(self, tmp_path):
        """Test timeliness assessment."""
        test_file = tmp_path / "recent.txt"
        test_file.write_text("content")

        profiler = DataProfiler()
        timeliness = profiler._assess_timeliness(test_file)

        assert "current" in timeliness or "recent" in timeliness

    def test_overall_quality_score(self, tmp_path):
        """Test overall quality score calculation."""
        csv_content = "a,b,c\n1,2,3\n4,5,6"
        csv_file = tmp_path / "complete.csv"
        csv_file.write_text(csv_content)

        profiler = DataProfiler()
        report = profiler.profile_file(str(csv_file))

        assert 0 <= report.overall_score <= 1

    def test_compare_profiles(self, tmp_path):
        """Test comparing two profiles."""
        (tmp_path / "file1.csv").write_text("a,b\n1,2")
        (tmp_path / "file2.csv").write_text("a\n1\n2\n3")

        profiler = DataProfiler()
        report1 = profiler.profile_file(str(tmp_path / "file1.csv"))
        report2 = profiler.profile_file(str(tmp_path / "file2.csv"))

        comparison = profiler.compare_profiles(report1, report2)

        assert "source1" in comparison
        assert "source2" in comparison
        assert "quality_comparison" in comparison

    def test_generate_summary_report(self, tmp_path):
        """Test generating summary across multiple files."""
        (tmp_path / "file1.csv").write_text("a,b\n1,2")
        (tmp_path / "file2.json").write_text('{"key": "value"}')

        profiler = DataProfiler()
        report1 = profiler.profile_file(str(tmp_path / "file1.csv"))
        report2 = profiler.profile_file(str(tmp_path / "file2.json"))

        summary = profiler.generate_summary_report([report1, report2])

        assert summary["total_sources"] == 2
        assert "average_quality" in summary


class TestDataSource:
    """Tests for DataSource dataclass."""

    def test_dataclass_creation(self):
        """Test creating DataSource."""
        source = DataSource(
            source_id="test_001",
            name="Test",
            source_type=SourceType.STRUCTURED,
            format=DataFormat.CSV,
            location="/path/to/file",
            size_bytes=1024,
        )

        assert source.source_id == "test_001"
        assert source.source_type == SourceType.STRUCTURED

    def test_to_dict(self):
        """Test converting to dictionary."""
        source = DataSource(
            source_id="test_001",
            name="Test",
            source_type=SourceType.STRUCTURED,
            format=DataFormat.CSV,
            location="/path/to/file",
            size_bytes=1024,
        )

        d = source.to_dict()

        assert d["source_type"] == "structured"
        assert d["format"] == "csv"

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "source_id": "test_001",
            "name": "Test",
            "source_type": "structured",
            "format": "csv",
            "location": "/path/to/file",
            "size_bytes": 1024,
        }

        source = DataSource.from_dict(d)

        assert source.source_type == SourceType.STRUCTURED
        assert source.format == DataFormat.CSV
