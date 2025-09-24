"""Gradio interface for the HPOA agent (simple)."""
from typing import List, Optional, Any
import os
import json
import gradio as gr
from aurelian.agents.hpoa.hpoa_agent import call_agent_with_retry, call_agent
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

    use_retry = kwargs.pop("use_retry", True)
    use_history = kwargs.pop("use_history", None)
    output_path = kwargs.pop("output_path", None)
    if output_path:
        print("--output is ignored in UI mode; results are shown in the interface.")
    agent_kwargs = dict(kwargs)

    def format_agent_result(result):
        data = result.output

        if hasattr(data, "model_dump"):
            dd = data.model_dump()

            explanation = dd.get("explanation") or ""
            explanation_stripped = explanation.strip()
            if (
                explanation
                and "```" not in explanation
                and (explanation_stripped.startswith("{") or explanation_stripped.startswith("["))
            ):
                explanation_out = f"```json\n{explanation}\n```"
            else:
                explanation_out = explanation

            ann = dd.get("annotations")
            if ann:
                ann_block = f"```json\n{json.dumps({'annotations': ann}, indent=2, ensure_ascii=False)}\n```"
                return f"{explanation_out}\n\n{ann_block}" if explanation_out else ann_block

            return explanation_out

        if isinstance(data, (dict, list)):
            return f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"

        return str(data)
    
    async def get_info(query: str, history: List[Any]) -> str:
        # Debug prints
        print(f"QUERY = {query}")
        print("HISTORY =", json.dumps(history, indent=2, ensure_ascii=False) if isinstance(history, list) else str(history))

        # Check for required key before running
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return (
                "Missing required environment variable: OPENAI_API_KEY.\n\n"
                "Set it before launching. Examples:\n"
                "- PowerShell: `$env:OPENAI_API_KEY = 'sk-...'`\n"
                "- Bash: `export OPENAI_API_KEY=sk-...`\n\n"
                "After setting the key, restart the app."
            )

        try:
            runner = call_agent_with_retry if use_retry else call_agent
            run_kwargs = dict(agent_kwargs)
            if use_history is not None:
                run_kwargs.setdefault("use_history", use_history)

            result = await runner(query, **run_kwargs)
            return format_agent_result(result)

        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() or "429" in msg:
                return f"Error: rate limit exceeded. Details: {msg}"
            return f"Error calling agent: {msg}"

    return gr.ChatInterface(
        fn=get_info,
        type="messages",
        title="HPOA Assistant",
        description="<div style='text-align: center;'>"
                "An AI assistant for querying and curating Human Phenotype Ontology Annotations (HPOA)"
                "</div>",
        chatbot=gr.Chatbot(type="messages", show_copy_button=True, render_markdown=True),
        examples=[
            ["Tell me what you know about MONDO:0011518 (Wiedemann-Steiner syndrome)"],
            ["What body system is HP:0009939 (mandibular aplasia)?"],
            ["Which phenotypes for Charcot-Marie tooth disease affect females?"],
            ["List all the phenotypes for Digeorge syndrome"],
            ["Return the OMIM Clinical Synopsis for Cystic fibrosis"],
            ["Suggest removal of phenotype annotations with poor evidence for ORPHA:580"],
            ["Propose new annotations for Fabry disease based on PMID:21092187"],
            ["Compare phenotypes for Down syndrome in HPOA to those in PMID:34440331"],
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
        print("ERROR: OMIM_API_KEY not set. OMIM tools will be unavailable.")
        raise SystemExit(1)
    if not ncbi_key:
        print("WARNING: NCBI_API_KEY not set. PubMed tools will be rate-limited.")

    port = int(os.environ.get("AURELIAN_PORT", "7860"))
    host = os.environ.get("AURELIAN_HOST", "127.0.0.1")

    ui = chat()
    print(f"Launching Gradio on http://{host}:{port}")
    ui.launch(server_name=host, server_port=port, inbrowser=True)
