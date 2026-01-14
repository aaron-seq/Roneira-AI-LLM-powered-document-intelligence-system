"""
Demo Script for Task 3: Prompt & Ontology Creation

Generates HTML visualizations for:
1. Ontology Graph - visual representation of knowledge graph
2. Concept Hierarchy - tree view of ontology structure
3. Prompt Template Library - available templates and examples
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ontology.knowledge_ontology import (
    KnowledgeOntology,
    NodeType,
    RelationType,
)
from backend.services.prompt_service import PromptService, PromptTemplate


def create_enhanced_ontology():
    """Create an enhanced ontology with AI/ML concepts."""
    ontology = KnowledgeOntology("document_intelligence", auto_init_core=True)

    # Add AI/ML concepts
    ai = ontology.add_concept(
        "Artificial Intelligence", "The simulation of human intelligence by machines"
    )
    ml = ontology.add_concept(
        "Machine Learning", "Subset of AI that enables learning from data"
    )
    dl = ontology.add_concept(
        "Deep Learning", "ML using neural networks with multiple layers"
    )
    nlp = ontology.add_concept(
        "Natural Language Processing", "AI for understanding human language"
    )
    rag = ontology.add_concept(
        "RAG", "Retrieval-Augmented Generation combining search and generation"
    )

    # Add relationships
    ontology.add_relation(ml.node_id, ai.node_id, RelationType.IS_A)
    ontology.add_relation(dl.node_id, ml.node_id, RelationType.IS_A)
    ontology.add_relation(nlp.node_id, ml.node_id, RelationType.IS_A)
    ontology.add_relation(rag.node_id, nlp.node_id, RelationType.DEPENDS_ON)

    # Add RAG components
    vec_db = ontology.add_concept(
        "Vector Database", "Database for storing and searching embeddings"
    )
    embeddings = ontology.add_concept(
        "Embeddings", "Dense vector representations of text"
    )
    llm = ontology.add_concept("LLM", "Large Language Model for text generation")
    retrieval = ontology.add_concept(
        "Retrieval", "Finding relevant documents from a corpus"
    )

    ontology.add_relation(rag.node_id, vec_db.node_id, RelationType.DEPENDS_ON)
    ontology.add_relation(rag.node_id, llm.node_id, RelationType.DEPENDS_ON)
    ontology.add_relation(rag.node_id, retrieval.node_id, RelationType.CONTAINS)
    ontology.add_relation(vec_db.node_id, embeddings.node_id, RelationType.CONTAINS)

    # Add document processing concepts
    ocr = ontology.add_concept(
        "OCR", "Optical Character Recognition for text extraction"
    )
    chunking = ontology.add_concept(
        "Chunking", "Breaking documents into smaller pieces"
    )
    parsing = ontology.add_concept(
        "Parsing", "Extracting structured data from documents"
    )

    ontology.add_relation(
        ocr.node_id,
        ontology.get_node_by_name("document").node_id,
        RelationType.RELATES_TO,
    )
    ontology.add_relation(chunking.node_id, retrieval.node_id, RelationType.PART_OF)
    ontology.add_relation(
        parsing.node_id,
        ontology.get_node_by_name("extract").node_id,
        RelationType.RELATES_TO,
    )

    return ontology


def generate_ontology_graph_html(ontology: KnowledgeOntology, output_path: Path):
    """Generate interactive ontology graph HTML."""
    stats = ontology.get_statistics()

    # Build nodes and edges for vis.js
    nodes_js = []
    for node in ontology.nodes.values():
        color = {
            NodeType.CATEGORY: "#4285f4",
            NodeType.CONCEPT: "#34a853",
            NodeType.ENTITY: "#fbbc05",
            NodeType.ATTRIBUTE: "#ea4335",
            NodeType.ACTION: "#9c27b0",
        }.get(node.node_type, "#666666")

        nodes_js.append(
            {
                "id": node.node_id,
                "label": node.name,
                "title": f"{node.description}\nType: {node.node_type.value}",
                "color": color,
                "shape": "box" if node.node_type == NodeType.CATEGORY else "ellipse",
            }
        )

    edges_js = []
    for rel in ontology.relations.values():
        edge_style = {
            RelationType.IS_A: {"arrows": "to", "dashes": False, "color": "#4285f4"},
            RelationType.PART_OF: {"arrows": "to", "dashes": True, "color": "#34a853"},
            RelationType.DEPENDS_ON: {
                "arrows": "to",
                "dashes": True,
                "color": "#ea4335",
            },
            RelationType.CONTAINS: {
                "arrows": "to",
                "dashes": False,
                "color": "#fbbc05",
            },
            RelationType.RELATES_TO: {"arrows": "", "dashes": True, "color": "#9e9e9e"},
        }.get(rel.relation_type, {"arrows": "to", "dashes": False, "color": "#666"})

        edges_js.append(
            {
                "from": rel.source_id,
                "to": rel.target_id,
                "label": rel.relation_type.value,
                **edge_style,
            }
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 3: Ontology Graph Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
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
        .stat-value {{ font-size: 32px; font-weight: 700; color: #4285f4; }}
        .stat-label {{ font-size: 14px; color: #9ca3af; margin-top: 4px; }}
        .graph-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 24px;
        }}
        .graph-container h2 {{ margin-bottom: 16px; color: #fff; }}
        #ontology-graph {{ height: 500px; border-radius: 12px; background: #0d1117; }}
        .legend {{
            display: flex;
            gap: 24px;
            justify-content: center;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .details-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }}
        .detail-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .detail-card h3 {{ margin-bottom: 16px; color: #fff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ color: #9ca3af; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Knowledge Ontology Graph</h1>
            <p>Task 3: Prompt and Ontology Creation - Interactive visualization of domain concepts</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{stats.total_nodes}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.total_relations}</div>
                <div class="stat-label">Total Relations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.max_depth}</div>
                <div class="stat-label">Max Depth</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.avg_relations_per_node:.1f}</div>
                <div class="stat-label">Avg Relations/Node</div>
            </div>
        </div>

        <div class="graph-container">
            <h2>Interactive Ontology Graph</h2>
            <div id="ontology-graph"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#4285f4"></div>Category</div>
                <div class="legend-item"><div class="legend-color" style="background:#34a853"></div>Concept</div>
                <div class="legend-item"><div class="legend-color" style="background:#fbbc05"></div>Entity</div>
                <div class="legend-item"><div class="legend-color" style="background:#ea4335"></div>Attribute</div>
                <div class="legend-item"><div class="legend-color" style="background:#9c27b0"></div>Action</div>
            </div>
        </div>

        <div class="details-grid">
            <div class="detail-card">
                <h3>Nodes by Type</h3>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
                    {"".join(f"<tr><td>{t.capitalize()}</td><td>{c}</td></tr>" for t, c in stats.nodes_by_type.items())}
                </table>
            </div>
            <div class="detail-card">
                <h3>Relations by Type</h3>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
                    {"".join(f"<tr><td>{t}</td><td>{c}</td></tr>" for t, c in stats.relations_by_type.items())}
                </table>
            </div>
        </div>
    </div>

    <script>
        var nodes = new vis.DataSet({repr(nodes_js).replace("'", '"')});
        var edges = new vis.DataSet({repr(edges_js).replace("'", '"')});

        var container = document.getElementById('ontology-graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                font: {{ color: '#ffffff', size: 12 }},
                borderWidth: 2,
                shadow: true,
            }},
            edges: {{
                font: {{ color: '#9ca3af', size: 10, strokeWidth: 0 }},
                width: 2,
                smooth: {{ type: 'continuous' }},
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                }},
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
            }},
        }};

        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_hierarchy_html(ontology: KnowledgeOntology, output_path: Path):
    """Generate concept hierarchy tree view."""

    def build_hierarchy_html(node_id, depth=0):
        node = ontology.get_node(node_id)
        if not node:
            return ""

        children = ontology.get_children(node_id)
        indent = depth * 20

        type_badge = {
            NodeType.CATEGORY: '<span class="badge category">Category</span>',
            NodeType.CONCEPT: '<span class="badge concept">Concept</span>',
            NodeType.ENTITY: '<span class="badge entity">Entity</span>',
            NodeType.ACTION: '<span class="badge action">Action</span>',
        }.get(node.node_type, '<span class="badge">Other</span>')

        children_html = ""
        if children:
            children_html = (
                '<ul class="children">'
                + "".join(build_hierarchy_html(c.node_id, depth + 1) for c in children)
                + "</ul>"
            )

        return f"""
        <li class="tree-item" style="margin-left: {indent}px;">
            <div class="node-content">
                <span class="node-name">{node.name}</span>
                {type_badge}
                <span class="node-desc">{node.description[:80] if node.description else "No description"}</span>
            </div>
            {children_html}
        </li>"""

    # Find root nodes
    roots = [n for n in ontology.nodes.values() if n.parent_id is None]

    tree_html = (
        '<ul class="tree">'
        + "".join(build_hierarchy_html(r.node_id, 0) for r in roots)
        + "</ul>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 3: Concept Hierarchy</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
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
        .tree-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .tree, .children {{ list-style: none; }}
        .tree-item {{
            padding: 8px 0;
            border-left: 2px solid rgba(255,255,255,0.1);
            padding-left: 16px;
            margin-left: 8px;
        }}
        .node-content {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            transition: background 0.2s;
        }}
        .node-content:hover {{ background: rgba(255,255,255,0.08); }}
        .node-name {{ font-weight: 600; color: #fff; min-width: 150px; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        .badge.category {{ background: #4285f4; color: #fff; }}
        .badge.concept {{ background: #34a853; color: #fff; }}
        .badge.entity {{ background: #fbbc05; color: #000; }}
        .badge.action {{ background: #9c27b0; color: #fff; }}
        .node-desc {{ color: #9ca3af; font-size: 13px; }}
        .children {{ margin-top: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Concept Hierarchy</h1>
            <p>Task 3: Ontology Structure - Hierarchical view of domain concepts and relationships</p>
        </div>

        <div class="tree-container">
            {tree_html}
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_prompt_templates_html(output_path: Path):
    """Generate prompt template library visualization."""
    prompt_service = PromptService()

    # Add custom ontology-guided template
    ontology_template = PromptTemplate(
        name="ontology_guided",
        template="""Use the ontology context and retrieved documents to answer.

Ontology Context:
{ontology_context}

Document Context:
{context}

Question: {question}

Provide an answer that leverages semantic relationships and factual content.

Answer:""",
        description="Ontology-enhanced question answering",
    )
    prompt_service.register_template(ontology_template)

    # Add entity extraction template
    entity_template = PromptTemplate(
        name="entity_extraction",
        template="""Extract named entities from the following text.

Text:
{text}

Extract and categorize:
1. Persons
2. Organizations
3. Locations
4. Dates
5. Key concepts

Entities:""",
        description="Named entity extraction from text",
    )
    prompt_service.register_template(entity_template)

    templates = prompt_service.list_templates()

    template_cards = ""
    for t in templates:
        vars_html = "".join(
            f'<span class="var-badge">{{{v}}}</span>' for v in t["variables"]
        )
        template_content = prompt_service.get_template(t["name"]).template
        template_cards += f"""
        <div class="template-card">
            <div class="template-header">
                <h3>{t["name"]}</h3>
                <span class="template-desc">{t["description"]}</span>
            </div>
            <div class="template-vars">
                <span class="vars-label">Variables:</span>
                {vars_html}
            </div>
            <div class="template-content">
                <pre>{template_content}</pre>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 3: Prompt Template Library</title>
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
            grid-template-columns: repeat(3, 1fr);
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
        .templates-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
        }}
        .template-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .template-header {{
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .template-header h3 {{
            color: #fff;
            font-size: 18px;
            margin-bottom: 4px;
        }}
        .template-desc {{ color: #9ca3af; font-size: 13px; }}
        .template-vars {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        .vars-label {{ color: #9ca3af; font-size: 13px; }}
        .var-badge {{
            background: #4285f4;
            color: #fff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-family: monospace;
        }}
        .template-content {{
            background: #0d1117;
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
        }}
        .template-content pre {{
            font-family: 'Consolas', monospace;
            font-size: 12px;
            color: #c9d1d9;
            white-space: pre-wrap;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Prompt Template Library</h1>
            <p>Task 3: Prompt Engineering - Reusable templates for RAG and document processing</p>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(templates)}</div>
                <div class="stat-label">Total Templates</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(len(t["variables"]) for t in templates)}</div>
                <div class="stat-label">Total Variables</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">2</div>
                <div class="stat-label">Custom Templates</div>
            </div>
        </div>

        <div class="templates-grid">
            {template_cards}
        </div>
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Generated: {output_path}")


def main():
    """Generate all Task 3 visualizations."""
    output_dir = Path(__file__).parent.parent / "docs" / "roadmap" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task 3: Prompt & Ontology Creation - Demo Script")
    print("=" * 60)

    # Create enhanced ontology
    print("\n1. Creating enhanced ontology...")
    ontology = create_enhanced_ontology()
    stats = ontology.get_statistics()
    print(f"   - Total nodes: {stats.total_nodes}")
    print(f"   - Total relations: {stats.total_relations}")
    print(f"   - Max depth: {stats.max_depth}")

    # Generate visualizations
    print("\n2. Generating HTML visualizations...")

    generate_ontology_graph_html(ontology, output_dir / "task3_ontology_graph.html")
    generate_hierarchy_html(ontology, output_dir / "task3_hierarchy.html")
    generate_prompt_templates_html(output_dir / "task3_prompt_templates.html")

    # Export ontology
    print("\n3. Exporting ontology...")
    ontology.export_json(str(output_dir / "ontology_export.json"))
    print(f"   - Exported to: {output_dir / 'ontology_export.json'}")

    # Show context generation
    print("\n4. Sample context generation for RAG:")
    context = ontology.get_context_for_concept("RAG", include_related=True, depth=1)
    print(f"   {context[:200]}...")

    # Show related concepts
    print("\n5. Related concepts to 'Machine Learning':")
    related = ontology.find_related_concepts("Machine Learning", depth=2)
    for node in related[:5]:
        print(f"   - {node.name}: {node.description[:50]}...")

    print("\n" + "=" * 60)
    print("Task 3 demo complete! Open HTML files in browser to view:")
    print(f"  - {output_dir / 'task3_ontology_graph.html'}")
    print(f"  - {output_dir / 'task3_hierarchy.html'}")
    print(f"  - {output_dir / 'task3_prompt_templates.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
