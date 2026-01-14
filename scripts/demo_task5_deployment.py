"""
Demo Script for Task 5: Deployment via Existing Pipeline

Generates HTML visualizations for:
1. CI/CD Pipeline Dashboard - deployment stages and status
2. Deployment Metrics - success rates, timing, environment health
3. Pipeline Run History - recent deployments and their status
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_pipeline_dashboard_html(output_path: Path):
    """Generate CI/CD pipeline dashboard visualization."""

    stages = [
        {
            "name": "Code Quality",
            "icon": "🔍",
            "status": "success",
            "time": "45s",
            "checks": ["Black", "isort", "Flake8", "MyPy"],
        },
        {
            "name": "Security Scan",
            "icon": "🔒",
            "status": "success",
            "time": "32s",
            "checks": ["Bandit", "Safety", "Dependency Check"],
        },
        {
            "name": "Unit Tests",
            "icon": "🧪",
            "status": "success",
            "time": "2m 15s",
            "checks": ["87 tests passed", "92% coverage"],
        },
        {
            "name": "Integration Tests",
            "icon": "🔗",
            "status": "success",
            "time": "3m 42s",
            "checks": ["API tests", "DB tests", "Service tests"],
        },
        {
            "name": "Docker Build",
            "icon": "🐳",
            "status": "success",
            "time": "1m 28s",
            "checks": ["Build image", "Push to registry"],
        },
        {
            "name": "Deploy Staging",
            "icon": "🚀",
            "status": "success",
            "time": "1m 05s",
            "checks": ["Railway deploy", "Smoke tests"],
        },
        {
            "name": "Deploy Production",
            "icon": "✅",
            "status": "success",
            "time": "1m 12s",
            "checks": ["Health check", "Performance tests"],
        },
    ]

    total_time = "10m 19s"

    # Build stage cards
    stage_cards = ""
    for i, stage in enumerate(stages):
        status_class = stage["status"]
        checks_html = "".join(
            f'<div class="check-item">✓ {c}</div>' for c in stage["checks"]
        )
        stage_cards += f"""
        <div class="stage-card {status_class}">
            <div class="stage-header">
                <span class="stage-icon">{stage["icon"]}</span>
                <span class="stage-name">{stage["name"]}</span>
                <span class="stage-time">{stage["time"]}</span>
            </div>
            <div class="stage-checks">
                {checks_html}
            </div>
            <div class="stage-status">
                <span class="status-badge {status_class}">{status_class.upper()}</span>
            </div>
        </div>
        {f'<div class="connector">→</div>' if i < len(stages) - 1 else ""}"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 5: CI/CD Pipeline Dashboard</title>
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
        .stat-value {{ font-size: 32px; font-weight: 700; color: #34a853; }}
        .stat-label {{ font-size: 14px; color: #9ca3af; margin-top: 4px; }}
        .pipeline-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .pipeline-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }}
        .pipeline-header h2 {{ color: #fff; }}
        .pipeline-info {{ color: #9ca3af; font-size: 14px; }}
        .pipeline {{
            display: flex;
            align-items: stretch;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 16px;
        }}
        .stage-card {{
            min-width: 160px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }}
        .stage-card.success {{ border-left: 3px solid #34a853; }}
        .stage-card.running {{ border-left: 3px solid #fbbc05; }}
        .stage-card.failed {{ border-left: 3px solid #ea4335; }}
        .stage-header {{
            padding: 16px;
            background: rgba(255,255,255,0.03);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }}
        .stage-icon {{ font-size: 28px; }}
        .stage-name {{ font-weight: 600; color: #fff; font-size: 13px; text-align: center; }}
        .stage-time {{ font-size: 12px; color: #9ca3af; }}
        .stage-checks {{
            padding: 12px;
        }}
        .check-item {{
            font-size: 11px;
            color: #34a853;
            margin-bottom: 4px;
        }}
        .stage-status {{
            padding: 12px;
            text-align: center;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
        }}
        .status-badge.success {{ background: rgba(52,168,83,0.2); color: #34a853; }}
        .status-badge.running {{ background: rgba(251,188,5,0.2); color: #fbbc05; }}
        .status-badge.failed {{ background: rgba(234,67,53,0.2); color: #ea4335; }}
        .connector {{
            display: flex;
            align-items: center;
            font-size: 20px;
            color: #4285f4;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CI/CD Pipeline Dashboard</h1>
            <p>Task 5: Deployment via Existing Pipeline - Automated build and deployment workflow</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(stages)}</div>
                <div class="stat-label">Pipeline Stages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_time}</div>
                <div class="stat-label">Total Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">98.5%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">v1.2.3</div>
                <div class="stat-label">Current Version</div>
            </div>
        </div>

        <div class="pipeline-container">
            <div class="pipeline-header">
                <h2>Pipeline Run #142 - main branch</h2>
                <span class="pipeline-info">Triggered by push • 5 minutes ago</span>
            </div>
            <div class="pipeline">
                {stage_cards}
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_deployment_metrics_html(output_path: Path):
    """Generate deployment metrics and environment health dashboard."""

    environments = [
        {
            "name": "Development",
            "status": "healthy",
            "version": "1.2.4-dev",
            "uptime": "99.9%",
            "last_deploy": "2 hours ago",
            "color": "#4285f4",
        },
        {
            "name": "Staging",
            "status": "healthy",
            "version": "1.2.3",
            "uptime": "99.8%",
            "last_deploy": "1 day ago",
            "color": "#fbbc05",
        },
        {
            "name": "Production",
            "status": "healthy",
            "version": "1.2.2",
            "uptime": "99.95%",
            "last_deploy": "3 days ago",
            "color": "#34a853",
        },
    ]

    metrics = [
        {"name": "Deployments (30d)", "value": "47", "change": "+12%"},
        {"name": "Avg Deploy Time", "value": "8m 32s", "change": "-15%"},
        {"name": "Rollbacks", "value": "2", "change": "-50%"},
        {"name": "Failed Builds", "value": "3", "change": "-25%"},
    ]

    # Build environment cards
    env_cards = ""
    for env in environments:
        status_icon = "✅" if env["status"] == "healthy" else "⚠️"
        env_cards += f"""
        <div class="env-card">
            <div class="env-header" style="border-color: {env["color"]}">
                <span class="env-name">{env["name"]}</span>
                <span class="env-status">{status_icon} {env["status"].upper()}</span>
            </div>
            <div class="env-body">
                <div class="env-metric">
                    <span class="label">Version</span>
                    <span class="value">{env["version"]}</span>
                </div>
                <div class="env-metric">
                    <span class="label">Uptime</span>
                    <span class="value">{env["uptime"]}</span>
                </div>
                <div class="env-metric">
                    <span class="label">Last Deploy</span>
                    <span class="value">{env["last_deploy"]}</span>
                </div>
            </div>
        </div>"""

    # Build metrics cards
    metric_cards = ""
    for m in metrics:
        change_class = (
            "positive"
            if m["change"].startswith("+")
            or m["change"].startswith("-5")
            or m["change"].startswith("-2")
            or m["change"].startswith("-1")
            else "negative"
        )
        if "-" in m["change"] and m["name"] != "Deployments (30d)":
            change_class = "positive"
        metric_cards += f"""
        <div class="metric-card">
            <div class="metric-value">{m["value"]}</div>
            <div class="metric-name">{m["name"]}</div>
            <div class="metric-change {change_class}">{m["change"]}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 5: Deployment Metrics</title>
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
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-value {{ font-size: 32px; font-weight: 700; color: #4285f4; }}
        .metric-name {{ font-size: 13px; color: #9ca3af; margin-top: 4px; }}
        .metric-change {{
            margin-top: 8px;
            font-size: 12px;
            font-weight: 600;
        }}
        .metric-change.positive {{ color: #34a853; }}
        .metric-change.negative {{ color: #ea4335; }}
        .section-title {{
            font-size: 18px;
            color: #fff;
            margin-bottom: 16px;
        }}
        .environments {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        .env-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }}
        .env-header {{
            padding: 16px;
            background: rgba(255,255,255,0.03);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid;
        }}
        .env-name {{ font-weight: 600; color: #fff; }}
        .env-status {{ font-size: 12px; color: #34a853; }}
        .env-body {{ padding: 16px; }}
        .env-metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .env-metric:last-child {{ border-bottom: none; }}
        .env-metric .label {{ color: #9ca3af; font-size: 13px; }}
        .env-metric .value {{ color: #fff; font-weight: 500; font-size: 13px; }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #deployChart {{ max-height: 250px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Deployment Metrics Dashboard</h1>
            <p>Task 5: Environment Health & Deployment Statistics</p>
        </div>

        <div class="metrics-row">
            {metric_cards}
        </div>

        <h3 class="section-title">Environment Status</h3>
        <div class="environments">
            {env_cards}
        </div>

        <div class="chart-container">
            <h3 class="section-title">Deployments Over Time (Last 7 Days)</h3>
            <canvas id="deployChart"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('deployChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{{
                    label: 'Successful',
                    data: [5, 8, 6, 9, 7, 3, 4],
                    borderColor: '#34a853',
                    backgroundColor: 'rgba(52, 168, 83, 0.1)',
                    fill: true,
                    tension: 0.4,
                }}, {{
                    label: 'Failed',
                    data: [0, 1, 0, 0, 1, 0, 0],
                    borderColor: '#ea4335',
                    backgroundColor: 'rgba(234, 67, 53, 0.1)',
                    fill: true,
                    tension: 0.4,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#9ca3af' }} }},
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


def generate_pipeline_history_html(output_path: Path):
    """Generate pipeline run history visualization."""

    runs = [
        {
            "id": 142,
            "branch": "main",
            "commit": "feat: add evaluation",
            "status": "success",
            "duration": "10m 19s",
            "time": "5 min ago",
            "author": "dev-team",
        },
        {
            "id": 141,
            "branch": "main",
            "commit": "fix: memory leak",
            "status": "success",
            "duration": "9m 45s",
            "time": "2 hours ago",
            "author": "dev-team",
        },
        {
            "id": 140,
            "branch": "feature/ontology",
            "commit": "add knowledge graph",
            "status": "success",
            "duration": "11m 02s",
            "time": "5 hours ago",
            "author": "ai-team",
        },
        {
            "id": 139,
            "branch": "main",
            "commit": "update dependencies",
            "status": "failed",
            "duration": "4m 12s",
            "time": "1 day ago",
            "author": "dev-team",
        },
        {
            "id": 138,
            "branch": "main",
            "commit": "add orchestrator",
            "status": "success",
            "duration": "10m 55s",
            "time": "1 day ago",
            "author": "ai-team",
        },
        {
            "id": 137,
            "branch": "hotfix/auth",
            "commit": "fix token refresh",
            "status": "success",
            "duration": "8m 30s",
            "time": "2 days ago",
            "author": "security-team",
        },
    ]

    # Build run rows
    run_rows = ""
    for run in runs:
        status_class = run["status"]
        status_icon = "✅" if status_class == "success" else "❌"
        run_rows += f"""
        <tr>
            <td><span class="run-id">#{run["id"]}</span></td>
            <td><span class="branch">{run["branch"]}</span></td>
            <td class="commit">{run["commit"]}</td>
            <td><span class="status-badge {status_class}">{status_icon} {run["status"].upper()}</span></td>
            <td>{run["duration"]}</td>
            <td>{run["time"]}</td>
            <td><span class="author">{run["author"]}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 5: Pipeline History</title>
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
        .history-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .history-container h2 {{
            color: #fff;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            color: #9ca3af;
            font-weight: 500;
            font-size: 13px;
            text-transform: uppercase;
        }}
        .run-id {{
            font-family: monospace;
            color: #4285f4;
            font-weight: 600;
        }}
        .branch {{
            background: rgba(66,133,244,0.2);
            color: #4285f4;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-family: monospace;
        }}
        .commit {{
            color: #9ca3af;
            font-size: 13px;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .status-badge.success {{ background: rgba(52,168,83,0.2); color: #34a853; }}
        .status-badge.failed {{ background: rgba(234,67,53,0.2); color: #ea4335; }}
        .author {{
            color: #9ca3af;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Run History</h1>
            <p>Task 5: Recent CI/CD Pipeline Executions</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">142</div>
                <div class="stat-label">Total Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">96.5%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">9m 48s</div>
                <div class="stat-label">Avg Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">5</div>
                <div class="stat-label">Today's Runs</div>
            </div>
        </div>

        <div class="history-container">
            <h2>Recent Runs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Run</th>
                        <th>Branch</th>
                        <th>Commit</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Time</th>
                        <th>Author</th>
                    </tr>
                </thead>
                <tbody>
                    {run_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def main():
    """Generate all Task 5 visualizations."""
    output_dir = Path(__file__).parent.parent / "docs" / "roadmap" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task 5: Deployment via Existing Pipeline - Demo Script")
    print("=" * 60)

    print("\n1. Generating CI/CD Pipeline Dashboard...")
    generate_pipeline_dashboard_html(output_dir / "task5_pipeline.html")

    print("\n2. Generating Deployment Metrics...")
    generate_deployment_metrics_html(output_dir / "task5_metrics.html")

    print("\n3. Generating Pipeline History...")
    generate_pipeline_history_html(output_dir / "task5_history.html")

    print("\n" + "=" * 60)
    print("Task 5 demo complete! Open HTML files in browser to view:")
    print(f"  - {output_dir / 'task5_pipeline.html'}")
    print(f"  - {output_dir / 'task5_metrics.html'}")
    print(f"  - {output_dir / 'task5_history.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
