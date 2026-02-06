"""Visualize the LangGraph workflow."""

import sys
from pathlib import Path
from typing import Optional

# Add project root directory to path to import askany modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.workflow.workflow_langgraph import AgentWorkflow  # noqa: E402


def visualize_workflow(
    workflow: Optional[AgentWorkflow] = None,
    output_format: str = "mermaid",
    output_file: Optional[str] = None,
    show_in_browser: bool = False,
) -> str:
    """Visualize the LangGraph workflow.

    Args:
        workflow: AgentWorkflow instance. If None, will create a minimal instance.
        output_format: Output format - "mermaid" (Mermaid code), "png" (PNG image),
                       or "html" (HTML with embedded Mermaid).
        output_file: Optional output file path. If None, will print to stdout or
                     use default filename.
        show_in_browser: If True and output_format is "html", open in browser.

    Returns:
        The visualization content (Mermaid code, PNG bytes, or HTML string).
    """
    # Create workflow instance if not provided
    if workflow is None:
        # Try to create a minimal workflow instance for visualization
        # Note: This may fail if dependencies are not available
        try:
            # Import minimal dependencies to create workflow
            from llama_index.core.llms import MockLLM

            from askany.rag.router import QueryRouter

            # Create a minimal mock RAGQueryEngine
            class MockRAGQueryEngine:
                def retrieve(self, query: str, metadata_filters=None):
                    return []

                def retrieve_with_scores(self, query: str, metadata_filters=None):
                    return [], 0.0

            # Create minimal router and LLM for visualization
            mock_engine = MockRAGQueryEngine()
            router = QueryRouter(
                docs_query_engine=mock_engine,  # type: ignore
                faq_query_engine=None,
            )
            llm = MockLLM()
            workflow = AgentWorkflow(router=router, llm=llm)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create workflow instance for visualization: {e}\n"
                "Please provide an existing AgentWorkflow instance, or ensure "
                "all dependencies are properly initialized."
            ) from e

    # Get the compiled graph
    graph = workflow.graph

    # Get the graph structure
    graph_structure = graph.get_graph()

    if output_format == "mermaid":
        # Generate Mermaid code
        mermaid_code = graph_structure.draw_mermaid()
        content = mermaid_code

        # Write to file or print
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            print(f"Mermaid diagram saved to: {output_file}")
        else:
            print(mermaid_code)

        return content

    elif output_format == "png":
        # Generate PNG image
        try:
            png_data = graph_structure.draw_mermaid_png()
            content = png_data

            # Write to file or display
            if output_file:
                with open(output_file, "wb") as f:
                    f.write(png_data)
                print(f"PNG diagram saved to: {output_file}")
            else:
                # Try to display in IPython if available
                try:
                    from IPython.display import Image, display

                    display(Image(png_data))
                except ImportError:
                    print("PNG data generated. Install IPython to display inline.")
                    print(f"PNG data size: {len(png_data)} bytes")

            return content
        except Exception as e:
            print(f"Error generating PNG: {e}")
            print("Falling back to Mermaid code...")
            return visualize_workflow(workflow, "mermaid", output_file, show_in_browser)

    elif output_format == "html":
        # Generate HTML with embedded Mermaid
        mermaid_code = graph_structure.draw_mermaid()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LangGraph Workflow Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .mermaid {{
            text-align: center;
        }}
        .info {{
            margin-top: 20px;
            padding: 10px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LangGraph Workflow Visualization</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
        <div class="info">
            <p><strong>Note:</strong> This diagram shows the structure of the LangGraph workflow.</p>
            <p>Nodes represent workflow steps, and edges show the flow between steps.</p>
        </div>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""

        content = html_content

        # Write to file
        if output_file is None:
            output_file = "langgraph_workflow.html"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML visualization saved to: {output_file}")

        # Open in browser if requested
        if show_in_browser:
            try:
                import webbrowser

                webbrowser.open(f"file://{Path(output_file).absolute()}")
            except Exception as e:
                print(f"Could not open browser: {e}")

        return content

    else:
        raise ValueError(
            f"Unknown output_format: {output_format}. "
            "Supported formats: 'mermaid', 'png', 'html'"
        )


def main():
    """Main entry point for visualization script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize LangGraph workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Mermaid code and print to stdout
  python -m askany.workflow.visual_workflow_langgraph

  # Generate HTML file
  python -m askany.workflow.visual_workflow_langgraph -f html -o workflow.html

  # Generate PNG image
  python -m askany.workflow.visual_workflow_langgraph -f png -o workflow.png

  # Generate HTML and open in browser
  python -m askany.workflow.visual_workflow_langgraph -f html -o workflow.html --browser
        """,
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["mermaid", "png", "html"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on format)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open HTML file in browser (only for HTML format)",
    )

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output is None:
        if args.format == "html":
            args.output = "langgraph_workflow.html"
        elif args.format == "png":
            args.output = "langgraph_workflow.png"
        elif args.format == "mermaid":
            args.output = "langgraph_workflow.mmd"

    try:
        visualize_workflow(
            workflow=None,
            output_format=args.format,
            output_file=args.output,
            show_in_browser=args.browser,
        )
    except Exception as e:
        print(f"Error visualizing workflow: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# 生成 HTML 文件（默认）
# python -m askany.workflow.visual_workflow_langgraph

# # 生成指定格式的文件
# python -m askany.workflow.visual_workflow_langgraph -f html -o workflow.html
# python -m askany.workflow.visual_workflow_langgraph -f png -o workflow.png
# python -m askany.workflow.visual_workflow_langgraph -f mermaid -o workflow.mmd

# # 生成 HTML 并在浏览器中打开
# python -m askany.workflow.visual_workflow_langgraph -f html --browser
