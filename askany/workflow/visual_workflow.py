"""Visualize the workflow."""

import sys
from pathlib import Path

# Add project root directory to path to import askany modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from llama_index.utils.workflow import draw_all_possible_flows

from askany.workflow.workflow_llamaindex import AgentWorkflowLlama

# Draw all
draw_all_possible_flows(AgentWorkflowLlama, filename="all_paths.html")
