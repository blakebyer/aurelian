"""
Agent for working with .hpoa files.
"""
from __future__ import annotations
import inspect, functools, os
import json
from typing import Any, Callable
import datetime
from pathlib import Path
from typing import Optional
from openai import OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type, AsyncRetrying
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ToolReturnPart
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAResponse, HPOADependencies, get_config
from aurelian.utils.async_utils import run_sync
from aurelian.agents.hpoa.hpoa_tools import (
    batch_search_hp, 
    batch_search_mondo,
    filter_hpoa,
    categorize_hpo,
    categorize_mondo,
    children_of,
    parents_of,
    get_omim_terms,
    get_omim_clinical,
    pubmed_search_pmids,
    lookup_pmid_text,
    extract_text_from_pdf,
)
# import reasoning models
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# system prompts
HPOA_SYSTEM_PROMPT = ("""
You are a friendly HPO/MONDO/OMIM biocurator and epidemiologist with deep disease-phenotype expertise. Respond quickly, share context generously, and stay precise.

OUTPUT
- Always return HPOAResponse (explanation + annotations).
- Q&A mode: annotations=[] unless explicitly asked to "show annotations" or "list phenotypes for disease X".
- Curation mode: include proposed annotations with rationale.
- IMPORTANT: Any non-HPOA results (e.g. OMIM clinical synopsis, PubMed abstracts, tool results, general explanations) must go in the explanation only, never in annotations.

CLASSIFICATION
- Off-topic (not related to biology) -> polite reminder of scope, no tools, no workflow narration, annotations=[].
- Otherwise -> continue workflow.

WORKFLOW
Q&A (default):
  1. filter_hpoa (phenotypes, limit=20 by default; use limit=None if user explicitly asks for "all")
  2. batch_search_hp (resolve a list of phenotype labels/IDs together, preferably in one or very few searches)
  3. batch_search_mondo (resolve a list of disease labels/IDs together, preferably in one or very few searches)
  4. children_of / parents_of (children or parents of ontology terms only if asked)
  5. categorize_hpo/mondo (categories only if asked)
  6. get_omim_clinical (OMIM synopsis -> put result in explanation, annotations=[])
  Stop when sufficient. Never propose new annotations.

Curation:
  1. filter_hpoa (existing annotations, same limit rule: 20 default, "all" if requested)
  2. Use PubMed tools (pubmed_search_pmids, lookup_pmid_text) only if adding/validating (evidence=PCS)
  3. Use web search tools (extract_text_from_pdf) only if given a URL by the user and adding/validating (evidence=TAS) 
  4. Populate annotations with rationale

ANNOTATION RULES
- Removal requires strong justification from agent context and literature; do not remove based on frequency tags or evidence type alone (IEA/PCS/TAS).
- evidence: PCS when given PMID, TAS for OMIM/Orphanet/Webpage statements, IEA for automatic annotations
- Do NOT remove terms just because they appear in other database_ids of the same disease.
- When curating, if a phenotype differs by sex, frequency, onset, or modifier, add it as a separate annotation (duplicate phenotype with differing attributes).
- Add/edit/remove allowed with clear rationale.
- All IDs must be valid CURIEs.

STOPPING
- Do not loop endlessly between tools.
- If a tool returns nothing useful, report that clearly and stop.
- End once enough information has been retrieved to answer the user's query.

TOOL RULES
- Tools may be called multiple times if the inputs are different or additional info is needed.
- Do not re-call the same tool on identical input.
- Deduplicate PMIDs; never look up the same PMID more than once.
- Use PubMed tools only in curation or if explicitly requested.
- Use web search tools only in curation and if URL is supplied.
- Prefer batching to minimize repeated calls.
- children_of / parents_of are the default for ontology navigation (HPO + MONDO) when exploring broader/narrower concepts.
- Tool failures: fallback gracefully, continue with partial info.

CRITICAL RULES
- annotations field always exists
- Explanations can include raw tool results if not annotations
- Explanations must contain results only (no workflow narration).
- Never narrate which tools you are calling; include tool insights only in the explanation.
- Never guess CURIEs or PMIDs
- CURIEs: ID (Label), e.g. HP:0001250 (Seizure)
- Direct, professional tone
""")


# Reasoning-specific instructions build on the core prompt so the agent surfaces its thought process.
HPOA_REASONING_SYSTEM_PROMPT = HPOA_SYSTEM_PROMPT + """

REASONING SUMMARY
- After completing the task, append to the end of the explanation field a short section titled "Reasoning summary."
- Provide 4-6 bullet points capturing the main steps you took (e.g., filtered HPOA annotations, identified target disease, compared to literature, mapped to HPO terms).
- Focus on process rather than results: describe the sequence of reasoning, what sources you consulted, and how you narrowed down or validated choices.
- Avoid repeating the content of annotations; keep it factual and auditable.

TONE
- Collegial but analytical: aim to make your reasoning transparent so another curator could follow your steps.
"""

# history persistence
MSG_HISTORY: list[ModelMessage] = []
MAX_HISTORY = 4
HPOA_HISTORY_FOLDER_NAME = "hpoa_history"
HPOA_HISTORY_ENABLED_DEFAULT = True
SESSION_FILENAME = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def _resolve_history_file(deps: Optional[HPOADependencies], history_dir: Optional[Path]) -> Path:
    if history_dir is not None:
        root = Path(history_dir).expanduser().resolve()
    elif deps and getattr(deps.workdir, "location", None):
        root = Path(deps.workdir.location) / HPOA_HISTORY_FOLDER_NAME
    else:
        root = Path.cwd() / HPOA_HISTORY_FOLDER_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root / f"history_{SESSION_FILENAME}.json"


def load_history(history_file: Path) -> None:
    """Load message history if it exists."""
    global MSG_HISTORY
    if history_file.exists():
        try:
            data = history_file.read_text(encoding="utf-8")
            MSG_HISTORY = ModelMessagesTypeAdapter.validate_json(data)
        except Exception as e:
            print(f"Error loading history: {e}")
            MSG_HISTORY = []
    else:
        MSG_HISTORY = []


def save_history(history_file: Path) -> None:
    try:
        json_data = ModelMessagesTypeAdapter.dump_json(MSG_HISTORY, indent=2)
        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")
        history_file.write_text(json_data, encoding="utf-8")
    except Exception as e:
        print("Error saving history:", e)


def _message_has_tool_return(message: ModelMessage) -> bool:
    return any(isinstance(part, ToolReturnPart) for part in message.parts)


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    if len(messages) <= MAX_HISTORY:
        return messages

    start_index = max(0, len(messages) - MAX_HISTORY)

    for idx in range(len(messages) - 1, start_index - 1, -1):
        if _message_has_tool_return(messages[idx]):
            start_index = max(0, idx - 1)

    trimmed = messages[start_index:]

    while trimmed and _message_has_tool_return(trimmed[0]):
        trimmed = trimmed[1:]

    if len(trimmed) > MAX_HISTORY:
        trimmed = trimmed[-MAX_HISTORY:]
        while trimmed and _message_has_tool_return(trimmed[0]):
            trimmed = trimmed[1:]

    return trimmed if trimmed else messages[-MAX_HISTORY:]

# Tool limiter helper class
class LimitedTool:
    def __init__(self, func: Callable, max_calls: int):
        self.func = func
        self.max_calls = max_calls
        self.calls = 0

    def wrap(self) -> Callable:
        """Return a wrapped function with the same signature as the original,
        enforcing a maximum number of calls.
        """
        sig = inspect.signature(self.func)

        @functools.wraps(self.func)
        async def wrapper(*args, **kwargs) -> Any:
            if self.calls >= self.max_calls:
                # return soft (non-terminating) error
                return {"note": f"{self.func.__name__} skipped (limit {self.max_calls} calls reached)"}
            self.calls += 1
            return await self.func(*args, **kwargs)

        wrapper.__signature__ = sig  # keep ctx, term, etc. visible to Tool
        return wrapper

DEFAULT_HPOA_MODEL = "openai:gpt-5"

def _standard_hpoa_tools() -> list[Tool]:
    return [
        # cheap local DB
        Tool(filter_hpoa),
        Tool(batch_search_hp),
        Tool(categorize_hpo),
        Tool(categorize_mondo),
        Tool(batch_search_mondo),
        Tool(children_of),
        Tool(parents_of),

        # expensive curation tools
        Tool(extract_text_from_pdf),
        Tool(get_omim_terms),
        Tool(get_omim_clinical),
        Tool(lookup_pmid_text),
        Tool(pubmed_search_pmids),
    ]

def create_hpoa_agent() -> Agent:
    return Agent(
        model=DEFAULT_HPOA_MODEL,
        output_type=HPOAResponse,
        system_prompt=HPOA_SYSTEM_PROMPT,
        history_processors=[keep_recent_messages],
        tools=_standard_hpoa_tools(),
    )

# Reasoning agent (not used in typical flows but available)
DEFAULT_HPOA_REASONING_MODEL = "gpt-5"

def create_hpoa_reasoning_agent() -> Agent:
    reasoning_model = OpenAIResponsesModel(DEFAULT_HPOA_REASONING_MODEL)
    reasoning_settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
    )
    return Agent(
        model=reasoning_model,
        model_settings=reasoning_settings,
        output_type=HPOAResponse,
        system_prompt=HPOA_REASONING_SYSTEM_PROMPT,
        history_processors=[keep_recent_messages],
        tools=_standard_hpoa_tools(),
    )

hpoa_reasoning_agent = create_hpoa_reasoning_agent()

# Main agent used for curation and Q&A
hpoa_agent = create_hpoa_agent()

def _select_agent(
    agent: Optional[Agent],
    agent_variant: Optional[str],
) -> Agent:
    if agent is not None:
        return agent

    normalized_variant = agent_variant.lower() if agent_variant else "standard"
    variant_agents = {
        "standard": hpoa_agent,
        "reasoning": hpoa_reasoning_agent,
    }

    if normalized_variant not in variant_agents:
        raise ValueError(f"Unknown HPOA agent variant: {agent_variant}")

    return variant_agents[normalized_variant]


# retry to avoid transient API errors
async def call_agent_with_retry(
    input: str,
    agent: Optional[Agent] = None,
    tool_limit: int = 25,
    use_history: Optional[bool] = None,
    history_dir: Optional[Path] = None,
    deps: Optional[HPOADependencies] = None,
    usage_limits: Optional[UsageLimits] = None,
    message_history: Optional[list[ModelMessage]] = None,
    agent_variant: Optional[str] = None,
):
    """Async call with retry logic (lighter + faster)."""

    use_history = HPOA_HISTORY_ENABLED_DEFAULT if use_history is None else use_history

    deps = deps or get_config()

    history = None
    history_file: Optional[Path] = None
    if use_history:
        history_file = _resolve_history_file(deps, history_dir)
        load_history(history_file)
        history = MSG_HISTORY if MSG_HISTORY else None
    elif message_history:
        history = message_history

    usage_limits = usage_limits or UsageLimits(request_limit=tool_limit)
    agent_instance = _select_agent(agent, agent_variant)

    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(min=0.5, max=3),
        stop=stop_after_attempt(2),                     
        retry=(retry_if_exception_type(ModelHTTPError) | retry_if_exception_type(OpenAIError)),
        reraise=True,
    ):
        with attempt:
            run_kwargs = {
                "deps": deps,
                "usage_limits": usage_limits,
                "message_history": history,
            }
            result = await agent_instance.run(
                input,
                **run_kwargs,
            )

            if use_history and history_file is not None:
                MSG_HISTORY.extend(result.new_messages())
                save_history(history_file)

            return result

# without retry
async def call_agent(
    input: str,
    agent: Optional[Agent] = None,
    tool_limit: int = 25,
    use_history: Optional[bool] = None,
    history_dir: Optional[Path] = None,
    deps: Optional[HPOADependencies] = None,
    usage_limits: Optional[UsageLimits] = None,
    message_history: Optional[list[ModelMessage]] = None,
    agent_variant: Optional[str] = None,
):
    """Async call without retries (lighter + faster)."""

    use_history = HPOA_HISTORY_ENABLED_DEFAULT if use_history is None else use_history

    deps = deps or get_config()

    history = None
    history_file: Optional[Path] = None
    if use_history:
        history_file = _resolve_history_file(deps, history_dir)
        load_history(history_file)
        history = MSG_HISTORY if MSG_HISTORY else None
    elif message_history:
        history = message_history

    usage_limits = usage_limits or UsageLimits(request_limit=tool_limit)
    agent_instance = _select_agent(agent, agent_variant)

    run_kwargs = {
        "deps": deps,
        "usage_limits": usage_limits,
        "message_history": history,
    }
    result = await agent_instance.run(
        input,
        **run_kwargs,
    )

    if use_history and history_file is not None:
        MSG_HISTORY.extend(result.new_messages())
        save_history(history_file)

    return result

def call_agent_sync(
    input: str,
    use_retry: bool = True,
    use_history: Optional[bool] = None,
    history_dir: Optional[Path] = None,
    agent: Optional[Agent] = None,
    agent_variant: Optional[str] = None,
    output_path: Optional[Path] = None,
    **kwargs,
):
    caller = call_agent_with_retry if use_retry else call_agent
    call_kwargs = dict(kwargs)
    result = run_sync(
        caller(
            input,
            use_history=use_history,
            history_dir=history_dir,
            agent=agent,
            agent_variant=agent_variant,
            **call_kwargs,
        )
    )

    if output_path:
        _write_output_to_file(_serialise_output(result.output), output_path)

    return result


def _write_output_to_file(output: Any, path_like: Path | str) -> None:
    target_path = Path(path_like)
    if target_path.suffix.lower() != '.json':
        raise ValueError('output path must end with .json')
    if target_path.parent:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(output, str):
        target_path.write_text(output, encoding='utf-8')
    else:
        target_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')

def _serialise_output(output: Any) -> str:
    if output is None:
        return "null"
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json(indent=2)
    if hasattr(output, "model_dump"):
        return json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if isinstance(output, (dict, list)):
        return json.dumps(output, indent=2, ensure_ascii=False)
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2, ensure_ascii=False, default=str)
