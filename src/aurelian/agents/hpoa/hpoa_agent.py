"""
Agent for working with .hpoa files.
"""
from __future__ import annotations
import inspect, functools, os
from typing import Any, Callable
import datetime
from pathlib import Path
from typing import Optional
from openai import OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay, retry_if_exception_type, RetryError
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ToolReturnPart
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAResponse, HPOADependencies, get_config
from aurelian.utils.async_utils import run_sync as agent_run_sync
from aurelian.agents.hpoa.hpoa_tools import (
    batch_search_hp, 
    batch_search_mondo,
    get_omim_terms,
    get_omim_clinical,
    lookup_pmid as lookup_pmid_text,
    pubmed_search_pmids,
    filter_hpoa,
    categorize_hpo,
    categorize_mondo,
    children_of,
    parents_of,
)
# import reasoning models
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# system prompts
HPOA_SYSTEM_PROMPT = ("""
You are an expert HPO/MONDO/OMIM biocurator and epidemiologist. Respond quickly and accurately.

OUTPUT
- Always return HPOAResponse (explanation + annotations).
- Q&A mode: annotations=[] unless explicitly asked to "show annotations" or "list phenotypes for disease X".
- Curation mode: include proposed annotations with rationale.
- IMPORTANT: Any non-HPOA results (e.g. OMIM clinical synopsis, PubMed abstracts, tool results, general explanations) must go in the explanation only, never in annotations.

CLASSIFICATION
- Off-topic (not related to biology) → polite redirect, no tools, no workflow narration, annotations=[].
- Otherwise → continue workflow.

WORKFLOW
Q&A (default):
  1. filter_hpoa (phenotypes, limit=20 by default; use limit=None if user explicitly asks for "all")
  2. batch_search_hp (resolve a list of phenotype labels/IDs together, preferably in one or very few searches)
  3. batch_search_mondo (resolve a list of disease labels/IDs together, preferably in one or very few searches)
  4. categorize_hpo/mondo (categories only if asked)
  5. get_omim_clinical (OMIM synopsis → put result in explanation, annotations=[])
  Stop when sufficient. Never propose new annotations.

Curation:
  1. filter_hpoa (existing annotations, same limit rule: 20 default, "all" if requested)
  2. Use PubMed tools (pubmed_search_pmids, lookup_pmid_text) only if adding/validating (evidence=PCS)
  3. Populate annotations with rationale

ANNOTATION RULES
- Removal requires strong justification from agent context and literature; do not remove based on frequency tags or evidence type alone (IEA/PCS/TAS).
- evidence: PCS when given PMID, TAS for OMIM/Orphanet statements, IEA for automatic annotations
- Do NOT remove terms just because they appear in other database_ids of the same disease.
- When curating, if a phenotype differs by sex, frequency, onset, or modifier, add it as a separate annotation (duplicate phenotype with differing attributes).
- Add/edit allowed with clear rationale.
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
- Prefer batching to minimize repeated calls.
- Tool failures: fallback gracefully, continue with partial info.

CRITICAL RULES
- annotations field always exists
- Explanations can include raw tool results if not annotations
- Explanations must contain results only (no workflow narration).
- Never guess CURIEs or PMIDs
- CURIEs: ID (Label), e.g. HP:0001250 (Seizure)
- Direct, professional tone
""")

# This simpler prompt is used for the "simple" agent variant.
HPOA_SIMPLE_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO and MONDO and a skilled epidemiologist. Default to fast, friendly Q&A. Do not narrate your process; ask follow ups if needed.
Use your own scientific knowledge for general explanations, but all ontology IDs/labels/definitions MUST come from tools.

WORKFLOW
1) HPO terms:
   - batch_search_hp(list of IDs/labels) → ID, label, definition.
   - categorize_hpo(HP:ID) if organ-system category requested.
   - children_of(HP:ID) / parents_of(HP:ID) for direct children/parents (max 10).

2) MONDO terms:
   - batch_search_mondo(list of IDs/labels) → ID, label, definition.
   - categorize_mondo(MONDO:ID) if high-level grouping requested.
   - children_of(MONDO:ID) / parents_of(MONDO:ID) for direct children/parents (max 10).

3) If unclear: ask one short clarifying question.
4) If an ID/label is invalid or not found: say so directly.

TOOL USE
- Only call tools when needed to confirm ontology facts.
- Prefer batching: resolve multiple terms together, avoid repeated calls.

FORMATTING
- Show terms as: ID (Label), e.g. HP:0004322 (Short stature).
- Children/parents: list up to 10, otherwise say "no children" or "no parents".
- Be brief and direct.

RELIABILITY
- No hallucinations: all ontology facts must come from tools.
- If tools return nothing, state that clearly.
- Stay on scope; politely decline off-topic requests.
""")

# history persistence
MSG_HISTORY: list[ModelMessage] = []
MAX_HISTORY = 3

# create history directory and history file per session
HPOA_HISTORY = os.environ.get("HPOA_HISTORY", "1") == "1"
HISTORY_FOLDER = Path("hpoa_history")
HISTORY_FOLDER.mkdir(exist_ok=True, parents=True)
SESSION_FILENAME = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
SESSION_HISTORY_FILE = HISTORY_FOLDER / f"history_{SESSION_FILENAME}.json"

def load_history():
    """Load message history if it exists."""
    global MSG_HISTORY
    if SESSION_HISTORY_FILE.exists():
        try:
            data = SESSION_HISTORY_FILE.read_text(encoding="utf-8")
            MSG_HISTORY = ModelMessagesTypeAdapter.validate_json(data)
        except Exception as e:
            print(f"Error loading history: {e}")
            MSG_HISTORY = []
    else:
        MSG_HISTORY = []

def save_history() -> None:
    try:
        json_data = ModelMessagesTypeAdapter.dump_json(MSG_HISTORY, indent=2)
        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")

        SESSION_HISTORY_FILE.write_text(json_data, encoding="utf-8")
    except Exception as e:
        print("Error saving history:", e)

async def message_at_index_contains_tool_return_parts(messages: list[ModelMessage], index: int) -> bool:
    return any(isinstance(part, ToolReturnPart) for part in messages[index].parts)
    
async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    number_of_messages = len(messages)
    if number_of_messages <= MAX_HISTORY:
        return messages
    if (await message_at_index_contains_tool_return_parts(messages, number_of_messages - MAX_HISTORY)):
        return messages
    return messages[-MAX_HISTORY:]

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

# Reasoning agent (not used in typical flows but available)
oai_model = OpenAIResponsesModel("gpt-5")
oai_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort="medium",
    openai_reasoning_summary="concise",
)
hpoa_reasoning_agent = Agent(
    model=oai_model,
    model_settings=oai_settings,
    output_type=HPOAResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        # local DB
        Tool(filter_hpoa),
        Tool(batch_search_hp),
        Tool(categorize_hpo),
        Tool(categorize_mondo),
        Tool(batch_search_mondo),

        # APIs limited to 5 calls each
        Tool(LimitedTool(get_omim_terms, max_calls=5).wrap()),
        Tool(LimitedTool(get_omim_clinical, max_calls=5).wrap()),
        Tool(LimitedTool(lookup_pmid_text, max_calls=5).wrap()),
        Tool(LimitedTool(pubmed_search_pmids, max_calls=5).wrap()),
    ],
)

# Main agent used for curation and Q&A
hpoa_agent = Agent(
    model="gpt-5",
    output_type=HPOAResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        # local DB
        Tool(filter_hpoa),
        Tool(batch_search_hp),
        Tool(categorize_hpo),
        Tool(categorize_mondo),
        Tool(batch_search_mondo),

        # APIs limited to 5 calls each
        Tool(LimitedTool(get_omim_terms, max_calls=5).wrap()),
        Tool(LimitedTool(get_omim_clinical, max_calls=5).wrap()),
        Tool(LimitedTool(lookup_pmid_text, max_calls=5).wrap()),
        Tool(LimitedTool(pubmed_search_pmids, max_calls=5).wrap()),
    ],
)

# Simplified agent used for ontology lookups and quick answers
hpoa_simple_agent = Agent(
    model="gpt-5",
    output_type=Optional[str],
    system_prompt=HPOA_SIMPLE_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        Tool(batch_search_hp),
        Tool(categorize_hpo),
        Tool(batch_search_mondo),
        Tool(categorize_mondo),
        Tool(children_of),
        Tool(parents_of),
    ],
)

# retry to avoid transient API errors
async def call_agent_with_retry(
    input: str,
    agent: Agent = hpoa_agent,
    model: Optional[str] = None,
    tool_limit: int = 25,
    use_history: bool = HPOA_HISTORY,
    deps: Optional[HPOADependencies] = None,
    usage_limits: Optional[UsageLimits] = None,
    message_history: Optional[list[ModelMessage]] = None,
):
    """Async call with retry logic (lighter + faster)."""

    if model:
        agent.model = model

    history = None
    if use_history:
        load_history()
        history = MSG_HISTORY if MSG_HISTORY else None
    elif message_history:
        history = message_history

    deps = deps or get_config()
    usage_limits = usage_limits or UsageLimits(request_limit=tool_limit)

    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(min=0.5, max=3),
        stop=stop_after_attempt(2),                     
        retry=(retry_if_exception_type(ModelHTTPError) | retry_if_exception_type(OpenAIError)),
        reraise=True,
    ):
        with attempt:
            result = await agent.run(
                input,
                deps=deps,
                usage_limits=usage_limits,
                message_history=history,
            )

            if use_history:
                MSG_HISTORY.extend(result.new_messages())
                save_history()

            return result

# without retry
async def call_agent(
    input: str,
    agent: Agent = hpoa_agent,
    model: Optional[str] = None,
    tool_limit: int = 25,
    use_history: bool = HPOA_HISTORY,
    deps: Optional[HPOADependencies] = None,
    usage_limits: Optional[UsageLimits] = None,
    message_history: Optional[list[ModelMessage]] = None,
):
    """Async call without retries (lighter + faster)."""

    if model:
        agent.model = model

    history = None
    if use_history:
        load_history()
        history = MSG_HISTORY if MSG_HISTORY else None
    elif message_history:
        history = message_history

    deps = deps or get_config()
    usage_limits = usage_limits or UsageLimits(request_limit=tool_limit)

    result = await agent.run(
        input,
        deps=deps,
        usage_limits=usage_limits,
        message_history=history,
    )

    if use_history:
        MSG_HISTORY.extend(result.new_messages())
        save_history()

    return result

def call_agent_sync(*args, **kwargs):
    """Synchronous wrapper for call_agent_with_retry."""
    return agent_run_sync(call_agent_with_retry(*args, **kwargs))