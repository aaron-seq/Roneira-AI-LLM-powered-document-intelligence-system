"""
Metrics Visualization Dashboard

Generates interactive charts and comparison tables
for evaluation results visualization and screenshot capture.

This module completes Task 6 by providing visualization capabilities
for metrics dashboards and proof-of-work documentation.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Types of charts supported."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    RADAR = "radar"
    HEATMAP = "heatmap"


@dataclass
class ChartConfig:
    """Configuration for a chart."""

    title: str
    chart_type: ChartType
    data: Dict[str, Any]
    width: int = 800
    height: int = 400
    colors: Optional[List[str]] = None


class MetricsDashboard:
    """
    Metrics visualization dashboard generator.

    Creates HTML dashboards with charts and tables for
    evaluation metrics visualization and screenshot capture.

    Example:
        dashboard = MetricsDashboard("RAG Evaluation Dashboard")
        dashboard.add_comparison_chart("Baseline vs Enhanced", baseline, enhanced)
        dashboard.add_metrics_table("Retrieval Metrics", retrieval_data)
        dashboard.export_to_html("dashboard.html")
    """

    # Default color scheme
    DEFAULT_COLORS = [
        "#4285f4",  # Google Blue
        "#34a853",  # Google Green
        "#fbbc05",  # Google Yellow
        "#ea4335",  # Google Red
        "#673ab7",  # Purple
        "#00bcd4",  # Cyan
    ]

    def __init__(self, title: str = "Metrics Dashboard"):
        """Initialize the dashboard."""
        self.title = title
        self.charts: List[ChartConfig] = []
        self.tables: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "title": title,
        }
        logger.info(f"Created MetricsDashboard: {title}")

    def add_comparison_chart(
        self,
        title: str,
        baseline: Dict[str, float],
        enhanced: Dict[str, float],
        chart_type: ChartType = ChartType.BAR,
    ) -> None:
        """
        Add a comparison chart between baseline and enhanced metrics.

        Args:
            title: Chart title
            baseline: Baseline metrics dictionary
            enhanced: Enhanced metrics dictionary
            chart_type: Type of chart to generate
        """
        self.charts.append(
            ChartConfig(
                title=title,
                chart_type=chart_type,
                data={
                    "labels": list(baseline.keys()),
                    "baseline": list(baseline.values()),
                    "enhanced": list(enhanced.values()),
                },
            )
        )

    def add_metrics_table(
        self,
        title: str,
        metrics: Dict[str, Any],
        show_grade: bool = True,
    ) -> None:
        """
        Add a metrics table to the dashboard.

        Args:
            title: Table title
            metrics: Metrics dictionary
            show_grade: Whether to show letter grades
        """
        rows = []
        for key, value in metrics.items():
            row = {"metric": key, "value": value}
            if show_grade and isinstance(value, (int, float)):
                row["grade"] = self._value_to_grade(float(value))
            rows.append(row)

        self.tables.append(
            {
                "title": title,
                "rows": rows,
            }
        )

    def add_distribution_chart(
        self,
        title: str,
        values: List[float],
        labels: Optional[List[str]] = None,
        chart_type: ChartType = ChartType.PIE,
    ) -> None:
        """Add a distribution chart."""
        if not labels:
            labels = [f"Item {i + 1}" for i in range(len(values))]

        self.charts.append(
            ChartConfig(
                title=title,
                chart_type=chart_type,
                data={
                    "labels": labels,
                    "values": values,
                },
            )
        )

    def add_trend_chart(
        self,
        title: str,
        data_series: Dict[str, List[float]],
        x_labels: List[str],
    ) -> None:
        """Add a trend/line chart."""
        self.charts.append(
            ChartConfig(
                title=title,
                chart_type=ChartType.LINE,
                data={
                    "x_labels": x_labels,
                    "series": data_series,
                },
            )
        )

    def _value_to_grade(self, value: float) -> str:
        """Convert a 0-1 value to letter grade."""
        if value >= 0.9:
            return "A"
        elif value >= 0.8:
            return "B"
        elif value >= 0.7:
            return "C"
        elif value >= 0.6:
            return "D"
        else:
            return "F"

    def _grade_to_color(self, grade: str) -> str:
        """Get color for grade."""
        colors = {
            "A": "#28a745",  # Green
            "B": "#17a2b8",  # Cyan
            "C": "#ffc107",  # Yellow
            "D": "#fd7e14",  # Orange
            "F": "#dc3545",  # Red
        }
        return colors.get(grade, "#6c757d")

    def export_to_html(self, output_path: str) -> str:
        """
        Export dashboard to interactive HTML file.

        Args:
            output_path: Path for HTML output

        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_html()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Exported dashboard to {output_path}")
        return output_path

    def _generate_html(self) -> str:
        """Generate complete HTML dashboard."""
        charts_html = "\n".join(self._generate_chart_html(c) for c in self.charts)
        tables_html = "\n".join(self._generate_table_html(t) for t in self.tables)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary-color: #4285f4;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --dark-color: #1a1a2e;
            --light-color: #f8f9fa;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .dashboard-header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #4285f4, #34a853);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .dashboard-header p {{
            color: #888;
            font-size: 1rem;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(66, 133, 244, 0.2);
        }}
        
        .card-title {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #fff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.5rem;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            background: rgba(66, 133, 244, 0.2);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .grade {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.875rem;
        }}
        
        .grade-A {{ background: #28a745; color: #fff; }}
        .grade-B {{ background: #17a2b8; color: #fff; }}
        .grade-C {{ background: #ffc107; color: #000; }}
        .grade-D {{ background: #fd7e14; color: #000; }}
        .grade-F {{ background: #dc3545; color: #fff; }}
        
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: #666;
            font-size: 0.875rem;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{self.title}</h1>
        <p>Generated: {self.metadata["created_at"]}</p>
    </div>
    
    <div class="dashboard-grid">
        {charts_html}
        {tables_html}
    </div>
    
    <div class="footer">
        <p>AI Developer Roadmap - Task 6: Accuracy Evaluation</p>
        <p>Roneira Document Intelligence System</p>
    </div>
    
    <script>
        {self._generate_chart_scripts()}
    </script>
</body>
</html>"""

    def _generate_chart_html(self, chart: ChartConfig) -> str:
        """Generate HTML for a single chart."""
        chart_id = f"chart_{id(chart)}"
        return f"""
        <div class="card">
            <h3 class="card-title">{chart.title}</h3>
            <div class="chart-container">
                <canvas id="{chart_id}"></canvas>
            </div>
        </div>"""

    def _generate_table_html(self, table: Dict[str, Any]) -> str:
        """Generate HTML for a table."""
        rows_html = ""
        for row in table.get("rows", []):
            grade_html = ""
            if "grade" in row:
                grade_html = (
                    f'<span class="grade grade-{row["grade"]}">{row["grade"]}</span>'
                )

            value = row.get("value", "")
            if isinstance(value, float):
                value = f"{value:.4f}"

            rows_html += f"""
            <tr>
                <td>{row.get("metric", "")}</td>
                <td>{value}</td>
                <td>{grade_html}</td>
            </tr>"""

        return f"""
        <div class="card">
            <h3 class="card-title">{table.get("title", "Metrics")}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Grade</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""

    def _generate_chart_scripts(self) -> str:
        """Generate JavaScript for charts."""
        scripts = []

        for chart in self.charts:
            chart_id = f"chart_{id(chart)}"

            if chart.chart_type == ChartType.BAR:
                datasets = []
                if "baseline" in chart.data and "enhanced" in chart.data:
                    datasets = [
                        {
                            "label": "Baseline",
                            "data": chart.data["baseline"],
                            "backgroundColor": "rgba(234, 67, 53, 0.7)",
                            "borderColor": "#ea4335",
                            "borderWidth": 2,
                        },
                        {
                            "label": "Enhanced",
                            "data": chart.data["enhanced"],
                            "backgroundColor": "rgba(52, 168, 83, 0.7)",
                            "borderColor": "#34a853",
                            "borderWidth": 2,
                        },
                    ]
                else:
                    datasets = [
                        {
                            "label": "Value",
                            "data": chart.data.get("values", []),
                            "backgroundColor": "rgba(66, 133, 244, 0.7)",
                        }
                    ]

                scripts.append(f"""
                new Chart(document.getElementById('{chart_id}'), {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(chart.data.get("labels", []))},
                        datasets: {json.dumps(datasets)}
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 1,
                                ticks: {{ color: '#888' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            x: {{
                                ticks: {{ color: '#888' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});""")

            elif chart.chart_type == ChartType.PIE:
                scripts.append(f"""
                new Chart(document.getElementById('{chart_id}'), {{
                    type: 'pie',
                    data: {{
                        labels: {json.dumps(chart.data.get("labels", []))},
                        datasets: [{{
                            data: {json.dumps(chart.data.get("values", []))},
                            backgroundColor: {json.dumps(self.DEFAULT_COLORS[: len(chart.data.get("values", []))])}
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ 
                                labels: {{ color: '#fff' }},
                                position: 'right'
                            }}
                        }}
                    }}
                }});""")

            elif chart.chart_type == ChartType.LINE:
                datasets = []
                series = chart.data.get("series", {})
                for i, (name, values) in enumerate(series.items()):
                    color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                    datasets.append(
                        {
                            "label": name,
                            "data": values,
                            "borderColor": color,
                            "backgroundColor": f"{color}33",
                            "fill": True,
                            "tension": 0.4,
                        }
                    )

                scripts.append(f"""
                new Chart(document.getElementById('{chart_id}'), {{
                    type: 'line',
                    data: {{
                        labels: {json.dumps(chart.data.get("x_labels", []))},
                        datasets: {json.dumps(datasets)}
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            y: {{
                                ticks: {{ color: '#888' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            x: {{
                                ticks: {{ color: '#888' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});""")

        return "\n".join(scripts)

    def generate_markdown_report(self) -> str:
        """Generate Markdown version of the dashboard."""
        lines = [
            f"# {self.title}",
            f"Generated: {self.metadata['created_at']}",
            "",
        ]

        # Add tables
        for table in self.tables:
            lines.append(f"## {table.get('title', 'Metrics')}")
            lines.append("")
            lines.append("| Metric | Value | Grade |")
            lines.append("|--------|-------|-------|")

            for row in table.get("rows", []):
                value = row.get("value", "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                grade = row.get("grade", "-")
                lines.append(f"| {row.get('metric', '')} | {value} | {grade} |")

            lines.append("")

        # Add chart data as text
        for chart in self.charts:
            lines.append(f"## {chart.title}")
            lines.append("")

            if chart.chart_type == ChartType.BAR and "baseline" in chart.data:
                lines.append("| Metric | Baseline | Enhanced | Δ% |")
                lines.append("|--------|----------|----------|-----|")

                labels = chart.data.get("labels", [])
                baseline = chart.data.get("baseline", [])
                enhanced = chart.data.get("enhanced", [])

                for i, label in enumerate(labels):
                    b = baseline[i] if i < len(baseline) else 0
                    e = enhanced[i] if i < len(enhanced) else 0
                    delta = ((e - b) / b * 100) if b > 0 else 0
                    delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
                    lines.append(f"| {label} | {b:.3f} | {e:.3f} | {delta_str} |")

            lines.append("")

        return "\n".join(lines)
