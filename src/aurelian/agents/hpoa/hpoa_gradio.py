"""Gradio interface for the HPOA agent (simple)."""
from typing import List, Optional
import os
import json

import gradio as gr

from .hpoa_agent import call_agent_with_retry
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
        # Minimal handler; Gradio renders Markdown/newlines in returned string
        try:
            # Preflight checks for required API keys and common setup issues
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                return (
                    "Missing required environment variable: OPENAI_API_KEY.\n\n"
                    "Set it before launching. Examples:\n"
                    "- PowerShell: `$env:OPENAI_API_KEY = 'sk-...'`\n"
                    "- Bash: `export OPENAI_API_KEY=sk-...`\n\n"
                    "After setting the key, restart the app."
                )

            # Stateless agent call per request
            result = call_agent_with_retry(query)
            data = result.output
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
                    return f"{text}\n\n```json\n{block}\n```"
                return text if text else json.dumps(dd, indent=2)
            if isinstance(data, (dict, list)):
                # Fallback: pretty print dicts/lists, which Gradio will render with newlines
                return json.dumps(data, indent=2)
            return str(data)
        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() or "429" in msg:
                return (
                    "Error: rate limit exceeded. Please try again in a few seconds.\n\n"
                    f"Details: {msg}"
                )
            # Improve error visibility for missing credentials
            if "OPENAI" in msg.upper() or "API KEY" in msg.upper():
                return (
                    "Error: model call failed. This often indicates a missing or invalid OPENAI_API_KEY.\n\n"
                    f"Details: {msg}"
                )
            return f"Error calling agent: {msg}"

    return gr.ChatInterface(
        fn=get_info,
        title="HPOA Assistant",
        description="<div style='text-align: center;'>"
                "An AI assistant for querying and curating Human Phenotype Ontology Annotations (HPOA)."
                "</div>",
        chatbot=gr.Chatbot(type="messages", show_copy_button=True, render_markdown=True),
        examples=[
            ["List the phenotypes and source studies for OMIM:300615"],
            ["What neurological phenotypes are associated with Coffin-Lowry syndrome?"],
            ["Propose new annotations for Fabry disease based on PMID:21092187"],
            ["Which system does HP:0004322 (Short stature) belong to?"]
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
    ui.launch(server_name=host, server_port=port, inbrowser=True, share=True)
