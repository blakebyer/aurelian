"""
Agent for working with .hpoa files.
"""
from __future__ import annotations
import datetime
import inspect
from pathlib import Path
from functools import wraps
from typing import List, Optional
from openai import OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type, RetryError
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ToolReturnPart
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAMixedResponse, get_config
from aurelian.agents.hpoa.hpoa_tools import (
    search_hp,
    search_mondo,
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
You are an expert biocurator for HPO/MONDO/OMIM. Your default mode is fast, conversational Q&A. Be direct, answer exactly what the user asks, or if unclear ask one short clarifying question. Switch into curation only if the user explicitly asks for it.

Output Contract
- Always return an object with fields:
  - text: free-form answer, should not include JSON. 
  If annotations are present, put them exclusively in the `annotations` field. 
  - annotations: list (possibly empty).
- By default in Q&A, leave annotations empty. 
- If the user explicitly asks for "all annotations," return them in annotations with status: "existing" and rationale left blank. Include a copyable JSON block as a list of HPOAResult rows.

- When showing any CURIE (OMIM, MONDO, ORPHA, DECIPHER, HP IDs), always display as: ID (label).
  - Example: "HP:0001250 (Seizures)" or "OMIM:300615 (Brunner syndrome)".

Q&A Strategy
- Answer conversationally and briefly; annotations are empty unless explicitly asked for all annotations.
- Tools such as filter_hpoa, filter_hpoa_by_hp, search_hp, search_mondo, and categorize_hpo may be used in Q&A if relevant.
- Never call literature tools (pubmed_search_pmids, lookup_pmid, get_omim_clinical) in Q&A or follow-ups. These are for curation only.
- Follow-ups: Continue the conversation naturally. Use tools if appropriate, but still never call literature tools.
- Never propose or perform curation unless the user explicitly requests it.
- For an off-topic question, politely decline and remind them of your scope.

Uncertainty & Timeouts
- If the request is ambiguous or underspecified, do not stall or attempt long searches.
- Instead, ask one short clarifying question, or say that you cannot proceed without more detail.
- If tool lookups return nothing, respond briefly with "No matching results were found."
- It is always acceptable to decline or ask for clarification rather than guessing or stalling.

Intent detection:
- Disease to phenotypes: use filter_hpoa with database_id or disease name to return up to 20 phenotypes + labels.  
  - Only return all phenotypes for a disease if the user explicitly asks for "all phenotypes."  
  - If asked for all annotations, return them as annotations with status: "existing" and rationale left blank. Include a copyable JSON block with {"explanation": "existing phenotypes for `CURIE`", "annotations": [...]}.  
- Phenotype concept: use search_hp / search_mondo (do not call HPOA unless mapping to diseases is requested)
- Category within a disease: filter_hpoa then categorize_hpo
- Terse inputs:
  - Disease-like -> phenotypes (ID + label) via HPOA
  - Phenotype-like -> use search_hp (return ID + label + definition)
- Not found: If lookup returns nothing, say so. Never fabricate IDs, labels, or references.

Absolutely No Hallucinations
- Source of truth: HPOA rows are authoritative for phenotypes, evidence codes, references (PMIDs/OMIM), frequency, onset, sex, qualifiers.
- IDs and labels must always come from tools:
  - Phenotypes: from HPOA rows or verified via search_hp
  - Diseases: from HPOA rows, search_mondo, or get_omim_terms
- Normalize identifiers to HP:nnnnnnn / MONDO:nnnnnnn when shown.
- If a lookup returns nothing, state that you cannot verify. Never invent IDs, labels, or references.
- No external inference: do not infer clinical specifics beyond HPOA. General disease context is fine; phenotype specifics must be anchored to HPOA rows.

Tool Reference
- filter_hpoa: tool to search rows from the hpoa table by one or more fields (e.g., disease_name, sex, hpo_id, frequency)
  - Provide a list of filters, each with a field and a query
  - You may optionally include a mode: "exact" (case-insensitive equality) or "like" (substring match). 
  - If mode is not given, assume "exact" for CURIEs (e.g. OMIM:123456, HP:0001250, MONDO:0005438) and "like" for human-readable labels (e.g. Fabry, female). Multiple filters are combined with AND.
- categorize_hpo: categorize HPO terms under top-level organ systems (HP:0000118)
- categorize_mondo: categorize MONDO terms into high-level disease groups (only when asked)
- search_hp: resolve HPO IDs/labels; verify labels for HP:IDs
- search_mondo, get_omim_terms: resolve MONDO/OMIM identifiers and labels
- get_omim_clinical (curation only): clinical features/inheritance from OMIM
- pubmed_search_pmids, lookup_pmid (curation only): literature lookup

Workflow
1) Q&A
   - Default: answer conversationally, tools allowed when relevant.
   - Annotations are empty unless the user explicitly asks for "all annotations."
   - Summarize clearly, max 20 terms when listing phenotypes.
   - For off-topic questions, answer briefly from context (no tool-calling) or politely decline.
2) Curation (only if explicitly requested)
   - Use search_mondo / get_omim_terms / search_hp / pubmed_search_pmids / lookup_pmid sparingly.
   - Populate annotations with proposed rows (status: new/updated/removed).
   - Always include a short explanation and a copyable JSON block with {"explanation","annotations"}.
   - Follow HPOA evidence rules strictly:
     - Frequency must be a fraction, percentage, or HPO frequency term.
     - Onset must be an HPO onset term.
     - Sex must be MALE, FEMALE, or empty.
     - Qualifier must be NOT or empty.
     - If different frequencies (e.g., by sex) are reported, copy the row and represent each separately.
   - Removal or deletion of rows: only propose removal if there is clearly insufficient evidence for the phenotype, based on both literature context and your own context. Apply a high bar for phenotype inclusion; if any reasonable evidence exists, keep the annotation.
   - If evidence is insufficient to support addition or removal, it is acceptable to propose no changes.
3) Never perform or suggest curation unless explicitly requested.
""")

# This simpler prompt is used for the "simple" agent variant.
HPOA_SIMPLE_SYSTEM_PROMPT = ("""You are an expert biocurator for HPO & MONDO. Your main role is to help with ontology questions, but you are also allowed to use your own context to answer science-oriented Q&A style questions.
Default to fast, conversational Q&A; use search_hp(HP:refID or phenotype label) to return information about an HPO term (including ID, label, and definition), use categorize_hpo to return the category of an HPO term, use search_mondo(MONDO:refID OR disease label) to return information about a MONDO term (including ID, label, and definition), and categorize_mondo to return the organ system of a MONDO term or disease label.
Use children_of(HP:refID) or parents_of(HP:refID) to get direct children or parents of an HPO term.
Use children_of(MONDO:refID) or parents_of(MONDO:refID) to get direct children or parents of a MONDO term.
At a maximum, list 10 children or parents, or if there are none, say "no children" or "no parents".
When listing ontology terms, always include both the ID and label, the label enclosed in parentheses, e.g. "HP:0004322 (Short stature)".
Do NOT call tools unless necessary. Absolutely no hallucinations, the ontology IDs, labels, and definitions must come from the tools. If a user provides an invalid ID or label, say you cannot find it.
If unclear, ask one short clarifying question. If the user asks an off-topic question, politely decline and remind them of your scope. Be brief and direct.""")

# history persistence
MSG_HISTORY: list[ModelMessage] = []
MAX_HISTORY = 3

# create history directory and history file per session
HISTORY_FOLDER = Path("history")
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
class ToolLimiter:
    def __init__(self, func, max_calls: int):
        self.func = func
        self.max_calls = max_calls
        self.calls = 0

    def wrap(self):
        sig = inspect.signature(self.func)

        @wraps(self.func)
        async def wrapper(*args, **kwargs):
            if self.calls >= self.max_calls:
                return {"error": f"{self.func.__name__} exceeded {self.max_calls} calls"}
            self.calls += 1
            return await self.func(*args, **kwargs)

        wrapper.__signature__ = sig
        return wrapper

# Reasoning agent (not used in typical flows but available)
oai_model = OpenAIResponsesModel("gpt-5")
oai_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort="low",
    openai_reasoning_summary="concise",
)
hpoa_reasoning_agent = Agent(
    model=oai_model,
    model_settings=oai_settings,
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
       # local DB
        Tool(ToolLimiter(filter_hpoa, max_calls=5).wrap()),
        Tool(ToolLimiter(search_hp, max_calls=30).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=30).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=10).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=10).wrap()),

        # APIs
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=3).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=3).wrap()),
    ],
)

# Main agent used for curation and Q&A
hpoa_agent = Agent(
    model="gpt-5",
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        # local DB
        Tool(ToolLimiter(filter_hpoa, max_calls=5).wrap()),
        Tool(ToolLimiter(search_hp, max_calls=50).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=50).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=5).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=5).wrap()),

        # APIs
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=3).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=3).wrap()),
    ],
)

# Simplified agent used for ontology lookups and quick answers
hpoa_simple_agent = Agent(
    model="gpt-5",
    output_type=Optional[str],
    system_prompt=HPOA_SIMPLE_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        Tool(ToolLimiter(search_hp, max_calls=10).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=10).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=10).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=10).wrap()),
        Tool(ToolLimiter(children_of, max_calls=3).wrap()),
        Tool(ToolLimiter(parents_of, max_calls=3).wrap()),
    ],
)

# retry to avoid transient API errors
@retry(wait=wait_random_exponential(min=0, max=30),
       stop=stop_after_attempt(3),
       retry=(retry_if_exception_type(ModelHTTPError) | retry_if_exception_type(OpenAIError)),
       reraise=True)
def call_agent_with_retry(
    input: str,
    agent: Agent = hpoa_agent,
    tool_limit: int = 100,
    use_history: bool = True,
):
    if use_history:
        load_history()
    result = agent.run_sync(
        input,
        deps=get_config(),
        usage_limits=UsageLimits(request_limit=tool_limit),
        message_history=MSG_HISTORY if use_history and MSG_HISTORY else None,
    )

    if use_history:
        MSG_HISTORY.extend(result.new_messages())
        save_history()
    return result

# without retry
def call_agent(
    input: str,
    agent: Agent = hpoa_simple_agent,
    tool_limit: int = 50,
    use_history: bool = True,
):
    if use_history:
        load_history()
    result = agent.run_sync(
        input,
        deps=get_config(),
        usage_limits=UsageLimits(request_limit=tool_limit),
        message_history=MSG_HISTORY if use_history and MSG_HISTORY else None,
    )
    if use_history:
        MSG_HISTORY.extend(result.new_messages())
        save_history()
    return result