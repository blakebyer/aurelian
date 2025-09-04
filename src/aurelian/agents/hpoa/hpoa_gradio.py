"""
Gradio interface for the HPOA agent.
"""
from typing import List, Optional
import os
import json

import gradio as gr

from aurelian.utils.async_utils import run_sync
from .hpoa_agent import hpoa_agent, call_agent_with_retry, reset_tool_log, get_tool_log
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

    def get_info(query: str, history: List[str]):
        # Keep it stateless and fast; still log incoming history for debugging
        # Gradio will render Markdown/newlines in the returned string
        print(f"QUERY: {query}")
        try:
            print(f"HISTORY: {history}")
        except Exception:
            pass
        try:
            # Reset tool log for this turn
            reset_tool_log()
            # Use the retrying runner defined in hpoa_agent.py
            result = call_agent_with_retry(query)
            data = result.output
            # Prepare a reasoning/tool trace block first
            trace = get_tool_log()
            if trace:
                yield f"Reasoning trace (tools):\n\n```json\n{json.dumps(trace, indent=2)}\n```"
            # Prefer conversational text; append a copyable JSON block when annotations are present
            if hasattr(data, "model_dump"):
                dd = data.model_dump()
                text = dd.get("text") or ""
                ann = dd.get("annotations") or []
                if ann:
                    block = json.dumps({
                        "explanation": (text or ""),
                        "annotations": ann,
                    }, indent=2)
                    yield f"{text}\n\n```json\n{block}\n```"
                    return
                yield text if text else json.dumps(dd, indent=2)
                return
            if isinstance(data, (dict, list)):
                # Fallback: pretty print dicts/lists, which Gradio will render with newlines
                yield json.dumps(data, indent=2)
                return
            yield str(data)
            return
        except Exception as e:
            yield f"Error calling agent: {e}"

    return gr.ChatInterface(
        fn=get_info,
        type="messages",
        title="HPOA AI Assistant",
        chatbot=gr.Chatbot(show_copy_button=True, render_markdown=True),
        examples=[
            ["Make HPOA file suggestions for Niemann-Pick, type C"],
            ["What neurological phenotypes are associated with Coffin-Lowry syndrome?"],
            ["List the phenotype annotations for OMIM:301500"],
            ["Return the phenotypes for PMID:19473999"]
        ]
    )


if __name__ == "__main__":
    # Ensure required API keys are set before launching
    openai_key = os.environ.get("OPENAI_API_KEY")
    omim_key = os.environ.get("OMIM_API_KEY")
    ncbi_key = os.environ.get("NCBI_API_KEY")
    if not openai_key:
        print("ERROR: Missing required environment variable: OPENAI_API_KEY")
        print("Set it before launching. Example (PowerShell):")
        print('$env:OPENAI_API_KEY = "sk-..."')
        raise SystemExit(1)
    if not omim_key:
        print("WARNING: OMIM_API_KEY not set. OMIM tools will be unavailable.")
    if not ncbi_key:
        print("WARNING: NCBI_API_KEY not set. PubMed tools will be rate-limited.")

    port = int(os.environ.get("AURELIAN_PORT", "7860"))
    host = os.environ.get("AURELIAN_HOST", "127.0.0.1")

    ui = chat()
    print(f"Launching Gradio on http://{host}:{port}")
    ui.launch(server_name=host, server_port=port, share=False, inbrowser=True)
