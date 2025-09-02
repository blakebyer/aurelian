"""
Gradio interface for the HPOA agent.
"""
from typing import List, Optional
import os
import json

import gradio as gr

from aurelian.utils.async_utils import run_sync
from .hpoa_agent import hpoa_agent, call_agent_with_retry
from .hpoa_config import HPOADependencies


def chat(deps: Optional[HPOADependencies] = None, **kwargs):
    """
    Initialize a chat interface for the HPOA agent.
    
    Args:
        deps: Optional dependencies configuration
        **kwargs: Additional arguments to pass to the agent
        
    Returns:
        A Gradio chat interface
    """
    if deps is None:
        deps = HPOADependencies()

    def get_info(query: str, history: List[str]) -> str:
        print(f"QUERY: {query}")
        print(f"HISTORY: {history}")
        if history:
            query += "## History"
            for h in history:
                query += f"\n{h}"
        try:
            # Use the retrying runner defined in hpoa_agent.py
            result = call_agent_with_retry(query)
            data = result.output
            # Serialize pydantic models or Python objects to a string for ChatInterface
            if hasattr(data, "model_dump_json"):
                return data.model_dump_json(indent=2)
            if hasattr(data, "model_dump"):
                return json.dumps(data.model_dump(), indent=2)
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=2)
            return str(data)
        except Exception as e:
            return f"Error calling agent: {e}"

    return gr.ChatInterface(
        fn=get_info,
        type="messages",
        title="HPOA AI Assistant",
        examples=[
            ["Make HPOA file suggestions for Niemann-Pick, type C"],
            ["What phenotypes are associated with Coffin-Lowry syndrome?"],
            ["List the phenotype annotations for OMIM:301500"],
            ["Return the phenotypes for PMID:19473999"]
        ]
    )


if __name__ == "__main__":
    # Ensure required API keys are set before launching
    openai_key = os.environ.get("OPENAI_API_KEY")
    omim_key = os.environ.get("OMIM_API_KEY")
    if not openai_key:
        print("ERROR: Missing required environment variable: OPENAI_API_KEY")
        print("Set it before launching. Example (PowerShell):")
        print('$env:OPENAI_API_KEY = "sk-..."')
        raise SystemExit(1)
    if not omim_key:
        print("WARNING: OMIM_API_KEY not set. OMIM tools will be unavailable.")

    port = int(os.environ.get("AURELIAN_PORT", "7860"))
    host = os.environ.get("AURELIAN_HOST", "127.0.0.1")

    ui = chat()
    print(f"Launching Gradio on http://{host}:{port}")
    ui.launch(server_name=host, server_port=port, share=False, inbrowser=True)
