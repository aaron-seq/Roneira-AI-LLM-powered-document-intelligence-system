"""
Demo Script for Task 4: Model Integration & Orchestration

Generates HTML visualizations for:
1. Agent Pipeline - workflow visualization
2. Agent Performance Dashboard - execution metrics
3. Integration Architecture - component relationships
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_agent_pipeline_html(output_path: Path):
    """Generate agent pipeline workflow visualization."""

    # Simulated workflow execution data
    agents = [
        {
            "name": "Router Agent",
            "type": "router",
            "color": "#4285f4",
            "description": "Query classification and routing",
            "time_ms": 45,
        },
        {
            "name": "Retriever Agent",
            "type": "retriever",
            "color": "#34a853",
            "description": "Document search and retrieval",
            "time_ms": 230,
        },
        {
            "name": "Reranker Agent",
            "type": "reranker",
            "color": "#fbbc05",
            "description": "Relevance scoring and reordering",
            "time_ms": 85,
        },
        {
            "name": "Synthesizer Agent",
            "type": "synthesizer",
            "color": "#ea4335",
            "description": "Response generation with LLM",
            "time_ms": 450,
        },
        {
            "name": "Validator Agent",
            "type": "validator",
            "color": "#9c27b0",
            "description": "Output quality verification",
            "time_ms": 120,
        },
        {
            "name": "Reflector Agent",
            "type": "reflector",
            "color": "#00bcd4",
            "description": "Self-improvement feedback",
            "time_ms": 55,
        },
    ]

    total_time = sum(a["time_ms"] for a in agents)

    # Build pipeline cards
    pipeline_html = ""
    for i, agent in enumerate(agents):
        percentage = (agent["time_ms"] / total_time) * 100
        pipeline_html += f"""
        <div class="agent-card">
            <div class="agent-header" style="background: {agent["color"]}">
                <span class="agent-number">{i + 1}</span>
                <span class="agent-name">{agent["name"]}</span>
            </div>
            <div class="agent-body">
                <p class="agent-desc">{agent["description"]}</p>
                <div class="agent-stats">
                    <div class="stat">
                        <span class="stat-value">{agent["time_ms"]}ms</span>
                        <span class="stat-label">Execution Time</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value">{percentage:.1f}%</span>
                        <span class="stat-label">Pipeline Share</span>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%; background: {agent["color"]}"></div>
                </div>
            </div>
            {f'<div class="arrow">→</div>' if i < len(agents) - 1 else ""}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 4: Agent Pipeline Workflow</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding: 24px;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; color: #fff; }}
        .header p {{ color: #9ca3af; }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card .stat-value {{ font-size: 32px; font-weight: 700; color: #4285f4; }}
        .stat-card .stat-label {{ font-size: 14px; color: #9ca3af; margin-top: 4px; }}
        .pipeline-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .pipeline-container h2 {{ margin-bottom: 24px; color: #fff; text-align: center; }}
        .pipeline {{
            display: flex;
            justify-content: space-between;
            align-items: stretch;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .agent-card {{
            flex: 1;
            min-width: 180px;
            position: relative;
        }}
        .agent-header {{
            padding: 12px 16px;
            border-radius: 12px 12px 0 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .agent-number {{
            background: rgba(255,255,255,0.2);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
        }}
        .agent-name {{ font-weight: 600; color: #fff; }}
        .agent-body {{
            background: rgba(255,255,255,0.03);
            padding: 16px;
            border-radius: 0 0 12px 12px;
            border: 1px solid rgba(255,255,255,0.1);
            border-top: none;
        }}
        .agent-desc {{ font-size: 13px; color: #9ca3af; margin-bottom: 16px; }}
        .agent-stats {{ display: flex; gap: 16px; margin-bottom: 12px; }}
        .stat {{ text-align: center; }}
        .stat .stat-value {{ font-size: 18px; font-weight: 600; color: #fff; display: block; }}
        .stat .stat-label {{ font-size: 11px; color: #6b7280; }}
        .progress-bar {{
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        .progress-fill {{ height: 100%; transition: width 0.3s; }}
        .arrow {{
            position: absolute;
            right: -20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 24px;
            color: #4285f4;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Agent Pipeline Workflow</h1>
            <p>Task 4: Model Integration - Orchestrating specialized agents for RAG processing</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(agents)}</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_time}ms</div>
                <div class="stat-label">Total Pipeline Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">0.92</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Max Iterations</div>
            </div>
        </div>

        <div class="pipeline-container">
            <h2>Agent Execution Flow</h2>
            <div class="pipeline">
                {pipeline_html}
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_performance_dashboard_html(output_path: Path):
    """Generate agent performance metrics dashboard."""

    # Simulated performance data
    performance_data = [
        {
            "agent": "Router",
            "calls": 1250,
            "avg_time": 42,
            "success_rate": 99.8,
            "color": "#4285f4",
        },
        {
            "agent": "Retriever",
            "calls": 1200,
            "avg_time": 215,
            "success_rate": 98.5,
            "color": "#34a853",
        },
        {
            "agent": "Reranker",
            "calls": 1180,
            "avg_time": 78,
            "success_rate": 99.2,
            "color": "#fbbc05",
        },
        {
            "agent": "Synthesizer",
            "calls": 1150,
            "avg_time": 420,
            "success_rate": 97.8,
            "color": "#ea4335",
        },
        {
            "agent": "Validator",
            "calls": 1150,
            "avg_time": 105,
            "success_rate": 100.0,
            "color": "#9c27b0",
        },
        {
            "agent": "Reflector",
            "calls": 380,
            "avg_time": 52,
            "success_rate": 99.5,
            "color": "#00bcd4",
        },
    ]

    total_calls = sum(p["calls"] for p in performance_data)
    avg_success = sum(p["success_rate"] for p in performance_data) / len(
        performance_data
    )

    # Build performance table rows
    table_rows = ""
    for p in performance_data:
        success_class = (
            "success"
            if p["success_rate"] >= 99
            else "warning"
            if p["success_rate"] >= 95
            else "error"
        )
        table_rows += f"""
        <tr>
            <td><span class="agent-badge" style="background: {p["color"]}">{p["agent"]}</span></td>
            <td>{p["calls"]:,}</td>
            <td>{p["avg_time"]}ms</td>
            <td><span class="status {success_class}">{p["success_rate"]}%</span></td>
        </tr>"""

    # Chart data
    chart_labels = json.dumps([p["agent"] for p in performance_data])
    chart_times = json.dumps([p["avg_time"] for p in performance_data])
    chart_colors = json.dumps([p["color"] for p in performance_data])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 4: Agent Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding: 24px;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; color: #fff; }}
        .header p {{ color: #9ca3af; }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{ font-size: 32px; font-weight: 700; color: #34a853; }}
        .stat-label {{ font-size: 14px; color: #9ca3af; margin-top: 4px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h3 {{ margin-bottom: 20px; color: #fff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ color: #9ca3af; font-weight: 500; font-size: 13px; }}
        .agent-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: #fff;
        }}
        .status {{
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status.success {{ background: rgba(52,168,83,0.2); color: #34a853; }}
        .status.warning {{ background: rgba(251,188,5,0.2); color: #fbbc05; }}
        .status.error {{ background: rgba(234,67,53,0.2); color: #ea4335; }}
        #performanceChart {{ max-height: 300px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Agent Performance Dashboard</h1>
            <p>Task 4: Model Integration - Real-time metrics for multi-agent orchestration</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{total_calls:,}</div>
                <div class="stat-label">Total Agent Calls</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_success:.1f}%</div>
                <div class="stat-label">Avg Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">152ms</div>
                <div class="stat-label">Avg Latency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">99.1%</div>
                <div class="stat-label">Pipeline Success</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Agent Execution Times</h3>
                <canvas id="performanceChart"></canvas>
            </div>
            <div class="card">
                <h3>Agent Statistics</h3>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Calls</th>
                        <th>Avg Time</th>
                        <th>Success</th>
                    </tr>
                    {table_rows}
                </table>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {chart_labels},
                datasets: [{{
                    label: 'Average Execution Time (ms)',
                    data: {chart_times},
                    backgroundColor: {chart_colors},
                    borderRadius: 8,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#9ca3af' }},
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }},
                    }},
                }},
            }},
        }});
    </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_integration_architecture_html(output_path: Path):
    """Generate integration architecture visualization."""

    services = {
        "existing": [
            {"name": "Embedding Service", "desc": "Text to vector conversion"},
            {"name": "Vector Store", "desc": "Similarity search"},
            {"name": "Retrieval Service", "desc": "Document retrieval"},
            {"name": "Prompt Service", "desc": "Template management"},
            {"name": "Chat Service", "desc": "LLM interaction"},
            {"name": "Memory Service", "desc": "Conversation history"},
        ],
        "new": [
            {"name": "Knowledge Ontology", "desc": "Semantic graph"},
            {"name": "RAG Evaluator", "desc": "Quality metrics"},
            {"name": "Data Catalog", "desc": "Source discovery"},
        ],
        "agents": [
            {"name": "Router", "connects": ["Ontology", "Catalog"]},
            {"name": "Retriever", "connects": ["Vector Store", "Embedding"]},
            {"name": "Synthesizer", "connects": ["Prompt", "Chat"]},
            {"name": "Validator", "connects": ["Evaluator"]},
        ],
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 4: Integration Architecture</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding: 24px;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; color: #fff; }}
        .header p {{ color: #9ca3af; }}
        .architecture {{
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .column {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .column h3 {{
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }}
        .column.existing h3 {{ background: #4285f4; color: #fff; }}
        .column.agents h3 {{ background: #9c27b0; color: #fff; }}
        .column.new h3 {{ background: #34a853; color: #fff; }}
        .service-item {{
            padding: 14px;
            margin-bottom: 12px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            border-left: 3px solid;
        }}
        .column.existing .service-item {{ border-color: #4285f4; }}
        .column.agents .service-item {{ border-color: #9c27b0; }}
        .column.new .service-item {{ border-color: #34a853; }}
        .service-name {{ font-weight: 600; color: #fff; margin-bottom: 4px; }}
        .service-desc {{ font-size: 12px; color: #9ca3af; }}
        .flow-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .flow-section h3 {{ margin-bottom: 20px; color: #fff; text-align: center; }}
        .flow {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .flow-step {{
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            min-width: 140px;
        }}
        .flow-step .icon {{ font-size: 32px; margin-bottom: 8px; }}
        .flow-step .label {{ font-weight: 600; color: #fff; }}
        .flow-arrow {{ font-size: 24px; color: #4285f4; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Integration Architecture</h1>
            <p>Task 4: Model Integration - How new components enhance existing services</p>
        </div>

        <div class="architecture">
            <div class="column existing">
                <h3>Existing Services</h3>
                {"".join(f'<div class="service-item"><div class="service-name">{s["name"]}</div><div class="service-desc">{s["desc"]}</div></div>' for s in services["existing"])}
            </div>
            
            <div class="column agents">
                <h3>Agent Orchestrator</h3>
                {"".join(f'<div class="service-item"><div class="service-name">{a["name"]} Agent</div><div class="service-desc">Connects: {", ".join(a["connects"])}</div></div>' for a in services["agents"])}
            </div>
            
            <div class="column new">
                <h3>New Components</h3>
                {"".join(f'<div class="service-item"><div class="service-name">{s["name"]}</div><div class="service-desc">{s["desc"]}</div></div>' for s in services["new"])}
            </div>
        </div>

        <div class="flow-section">
            <h3>Query Processing Flow</h3>
            <div class="flow">
                <div class="flow-step">
                    <div class="icon">👤</div>
                    <div class="label">User Query</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">🔀</div>
                    <div class="label">Router</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">🔍</div>
                    <div class="label">Retriever</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">📊</div>
                    <div class="label">Reranker</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">🧠</div>
                    <div class="label">Synthesizer</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">✅</div>
                    <div class="label">Validator</div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <div class="icon">💬</div>
                    <div class="label">Response</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def main():
    """Generate all Task 4 visualizations."""
    output_dir = Path(__file__).parent.parent / "docs" / "roadmap" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task 4: Model Integration & Orchestration - Demo Script")
    print("=" * 60)

    print("\n1. Generating Agent Pipeline visualization...")
    generate_agent_pipeline_html(output_dir / "task4_agent_pipeline.html")

    print("\n2. Generating Performance Dashboard...")
    generate_performance_dashboard_html(output_dir / "task4_performance.html")

    print("\n3. Generating Integration Architecture...")
    generate_integration_architecture_html(output_dir / "task4_integration.html")

    print("\n" + "=" * 60)
    print("Task 4 demo complete! Open HTML files in browser to view:")
    print(f"  - {output_dir / 'task4_agent_pipeline.html'}")
    print(f"  - {output_dir / 'task4_performance.html'}")
    print(f"  - {output_dir / 'task4_integration.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
