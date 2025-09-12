"""
Agent for working with .hpoa files.
"""
from __future__ import annotations
import datetime
import inspect
from pathlib import Path
from functools import wraps
from typing import List, Optional
import anyio
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_core import to_json
from aurelian.agents.hpoa.hpoa_config import HPOAMixedResponse, get_config, close_client
from aurelian.agents.hpoa.hpoa_tools import (
    search_hp,
    search_mondo,
    get_omim_terms,
    get_omim_clinical,
    lookup_pmid as lookup_pmid_text,
    pubmed_search_pmids,
    filter_hpoa,
    filter_hpoa_by_pmid,
    filter_hpoa_by_hp,
    categorize_hpo,
    categorize_mondo,
    children_of,
    parents_of,
)
# import reasoning models
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# system prompts
HPOA_SYSTEM_PROMPT = ("""You are an expert biocurator for HPO/MONDO/OMIM. Default to fast, conversational Q&A; switch to curation only when explicitly asked. If unclear, ask one short clarifying question.

Output Contract
- Always return an object with fields:
  - text: free-form answer
  - annotations: list (possibly empty). Leave empty for Q&A.

Q&A Strategy (fast path - choose the right source)
- Do NOT call tools in Q&A unless the user provided or explicitly asked about an ontology/database term: HP:nnnnnnn, MONDO:nnnnnnn, OMIM:nnnnnnn, ORPHA:nnnnnnn, DECIPHER IDs, or PMID:nnnnnnn.
- Intent detection:
  - Disease->phenotypes (e.g., list phenotypes for OMIM/MONDO/label/PMID; does disease X have phenotype Y?; organ-system phenotypes in a disease):
    - Use HPOA only via filter_hpoa / filter_hpoa_by_pmid / filter_hpoa_by_hp
    - Summarize up to 10 phenotypes, returning both the ID and label from search_hp, e.g. "HP:0000083 (Renal insufficiency)" (return all if <10)
  - Phenotype concept (e.g., what is HP:0001250? map a phenotype label to HP:ID; compare phenotypes):
    - Use ontology tools only: search_hp / search_mondo; do NOT call HPOA
  - Category within a disease:
    - Make one baseline HPOA call (filter_hpoa), then call categorize_hpo(HP:ID) to keep matches; summarize up to 10 (ID and label)
  - Terse inputs:
    - Disease-like (OMIM/MONDO/ORPHA/DECIPHER or a disease label): list phenotypes via HPOA
    - Phenotype-like (HP:nnnnnnn or a phenotype label): use search_hp only; do NOT call HPOA unless asked for phenotype->diseases
  - Not found: if baseline returns zero rows say: "Sorry, the given ID/label is not found in the HPOA file. Please try alternate spelling or verify the disease ID." Do not fabricate results or call literature tools in Q&A.

In all Q&A cases
- Be brief and direct. Leave annotations empty.
- Do NOT call literature (PubMed) or OMIM tools.
- Do NOT call any tools for general/off-topic questions - answer briefly or decline.

Absolutely No Hallucinations
- Source of truth: HPOA rows are authoritative for phenotypes, evidence codes, references (PMIDs/OMIM), frequency, onset, sex, qualifiers. If a field is missing, say "not specified in HPOA".
- IDs/labels must come from tools:
  - Phenotypes: use hpo_id values from HPOA rows. If showing a label for an HP:ID, verify via search_hp(HP:nnnnnnn).
  - Diseases: use database_id/disease_name from HPOA rows. To resolve/verify IDs/labels, use search_mondo (MONDO) or get_omim_terms (OMIM).
  - Normalize identifiers to HP:nnnnnnn / MONDO:nnnnnnn when shown.
  - If a lookup returns nothing, state you cannot verify. Never invent IDs, labels, or references.
- No external inference: do not infer clinical specifics beyond HPOA. General disease context is fine; phenotype specifics must be anchored to HPOA rows.

Curation (slow path - only when explicitly asked)
- Use search_mondo / get_omim_terms / search_hp / pubmed_search_pmids / lookup_pmid sparingly to justify changes. Don't lookup the same PMID multiple times.
- Include reasoning in text and populate annotations with proposed rows (status: new/updated/removed).
- If a user asks for removal or modification of an annotation, only use evidence from the literature to support these curations.
- Include a small copyable JSON block with {"explanation","annotations"}. It is fine to propose no changes if evidence is insufficient.
- Annotation field guidelines: frequency must be a fraction, percentage, or HPO frequency term; if the text lists different frequencies by sex, duplicate the rows and list the frequency separately for each sex; if a frequency is a range, input the average of the bounds; onset must be an HPO onset term; sex must be MALE, FEMALE, or empty; qualifier is either NOT or empty.

Tools (only when explicitly relevant)
- filter_hpoa: load HPOA rows (database_id normalized equality; disease_name case-insensitive LIKE)
- filter_hpoa_by_pmid: rows citing a PMID (PMID:nnnnnnn or digits)
- filter_hpoa_by_hp: rows for a phenotype (HP:ID or label; labels resolved via search_hp)
- categorize_hpo: categorize HPO terms under top-level organ systems (HP:0000118)
- categorize_mondo: categorize MONDO terms into high-level disease groupings (use only when asked about MONDO categories)
- search_hp: resolve HPO IDs/labels; verify labels for HP:IDs; find onset/frequency terms when explicitly stated
- search_mondo, get_omim_terms: resolve MONDO/OMIM disease identifiers and labels
- get_omim_clinical (curation only): clinical features/inheritance from OMIM
- pubmed_search_pmids, lookup_pmid (curation only): literature lookup

Workflow
1) Q&A:
   - Disease->phenotypes: one HPOA call (filter_hpoa / filter_hpoa_by_pmid / filter_hpoa_by_hp); optionally categorize_hpo
   - Phenotype concept: use only ontology tools (search_hp/search_mondo)
   - No tools for general/off-topic questions
   - Summarize up to 10 (returning both ID and label); leave annotations empty; DO NOT call literature/OMIM tools
   - If a user asks a question, try to answer it. DO NOT say you are "going to" do something and terminate.
2) Curation (on request): use search_mondo/get_omim_terms/search_hp/pubmed_search_pmids/lookup_pmid sparingly; return 10 or fewer annotations + short explanation.
3) Be conservative and transparent; acceptable to propose no changes
4) Include onset/frequency/sex only when supported by HPOA or explicit evidence in curation""")

# This simpler prompt is used for the "simple" agent variant.
HPOA_SIMPLE_SYSTEM_PROMPT = ("""You are an expert biocurator for HPO & MONDO, who can also answer questions about biology, medicine, and human diseases and phenotypes. Default to fast, conversational Q&A; use search_hp(HP:refID or phenotype label) to return information about an HPO term (including ID, label, and definition), use categorize_hpo to return the category of an HPO term, use search_mondo(MONDO:refID OR disease label) to return information about a MONDO term (including ID, label, and definition), and categorize_mondo to return the organ system of a MONDO term or disease label.
Use children_of(HP:refID) or parents_of(HP:refID) to get direct children or parents of an HPO term.
Use children_of(MONDO:refID) or parents_of(MONDO:refID) to get direct children or parents of a MONDO term.
At a maximum, list 10 children or parents, or if there are none, say "no children" or "no parents".
When listing ontology terms, always include both the ID and label, the label enclosed in parentheses, e.g. "HP:0004322 (Short stature)".
Do NOT call tools unless necessary. Absolutely no hallucinations, the ontology IDs, labels, and definitions must come from the tools. If a user provides an invalid ID or label, say you cannot find it.
If unclear, ask one short clarifying question. If the user asks an off-topic question, politely decline and remind them of your scope. Be brief and direct.""")

# history persistence
MSG_HISTORY: list[ModelMessage] = []

# create history directory and history file per session
HISTORY_FOLDER = Path("history")
HISTORY_FOLDER.mkdir(exist_ok=True, parents=True)
SESSION_FILENAME = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
SESSION_HISTORY_FILE = HISTORY_FOLDER / f"history_{SESSION_FILENAME}.json"

def load_history() -> None:
    """Populate MSG_HISTORY from the session history file if it exists."""
    global MSG_HISTORY
    if SESSION_HISTORY_FILE.exists():
        try:
            data = SESSION_HISTORY_FILE.read_bytes()
            # Use Pydantic AI's adapter to convert JSON bytes into
            # ModelMessage objects.
            MSG_HISTORY = ModelMessagesTypeAdapter.validate_json(data)
        except Exception:
            MSG_HISTORY = []
    # If the file doesn't exist, leave MSG_HISTORY as-is (empty)

def save_history() -> None:
    """Write the current MSG_HISTORY to the session history file."""
    try:
        # Convert the list of ModelMessage objects to JSON bytes and
        # write them to the file.
        SESSION_HISTORY_FILE.write_bytes(to_json(MSG_HISTORY))
    except Exception:
        # If writing fails, ignore silently
        pass

MAX_HISTORY = 3 #  messages of history

def create_context(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Trim the message history for context but preserve the system prompt.

    This function takes the full list of messages and returns a new list
    where the first message (which should contain the system prompt) is
    preserved, and only the last MAX_HISTORY messages are kept. If the
    input list has fewer than MAX_HISTORY+1 entries, it returns the
    original list unchanged.
    """
    # Always return empty or original lists unchanged
    if not messages:
        return messages
    # Slice: head is first message, tail is last MAX_HISTORY messages
    head = messages[:1]
    tail = messages[-MAX_HISTORY:]
    return head + tail

def append_new_messages(new_msgs: List[ModelMessage]) -> None:
    """Append new messages to the global history and persist them.

    Any messages appended here will be saved to the session file so
    they are available on subsequent calls in the same session.
    """
    global MSG_HISTORY
    # Extend the in-memory history with new messages
    MSG_HISTORY.extend(new_msgs)
    # Immediately save to disk so the history is durable
    save_history()

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
oai_model = OpenAIResponsesModel("gpt-5-mini")
oai_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort="low",
    openai_reasoning_summary="concise",
)
hpoa_reasoning_agent = Agent(
    model=oai_model,
    model_settings=oai_settings,
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[create_context],
    tools=[
        Tool(ToolLimiter(filter_hpoa, max_calls=3).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=3).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=3).wrap()),
        Tool(ToolLimiter(search_hp, max_calls=25).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=25).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=3).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=3).wrap()),
    ],
)

# Main agent used for curation and Q&A
hpoa_agent = Agent(
    model="gpt-5-mini",
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    history_processors=[create_context],
    tools=[
        Tool(ToolLimiter(filter_hpoa, max_calls=3).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=3).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=3).wrap()),
        Tool(ToolLimiter(search_hp, max_calls=25).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=25).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=3).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=3).wrap()),
    ],
)

# Simplified agent used for ontology lookups and quick answers
hpoa_simple_agent = Agent(
    model="gpt-5-mini",
    output_type=Optional[str],
    system_prompt=HPOA_SIMPLE_SYSTEM_PROMPT,
    history_processors=[create_context],
    tools=[
        Tool(ToolLimiter(search_hp, max_calls=10).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=10).wrap()),
        Tool(ToolLimiter(categorize_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(children_of, max_calls=3).wrap()),
        Tool(ToolLimiter(parents_of, max_calls=3).wrap()),
    ],
)

# retry to avoid transient API errors
@retry(wait=wait_random_exponential(min=0, max=30),
       stop=stop_after_attempt(4),
       retry=retry_if_exception_type(ModelHTTPError))
def call_agent_with_retry(input: str, agent: Agent = hpoa_agent, tool_limit: int = 75):
    """Run an agent synchronously with retry and history persistence.

    This helper loads the history from disk, executes the agent,
    appends the new messages to MSG_HISTORY and saves them back to disk.
    """
    # Always load history at the start in case MSG_HISTORY has been reset
    load_history()
    try:
        result = agent.run_sync(
            input,
            deps=get_config(),
            usage_limits=UsageLimits(request_limit=tool_limit),
            message_history=MSG_HISTORY or None,
        )
        # Append new messages (and save to disk)
        # On the very first call (no prior history), include the system prompt
        if not MSG_HISTORY:
            MSG_HISTORY.extend(result.all_messages())
            save_history()
        else:
            append_new_messages(result.new_messages())
        return result
    finally:
        try:
            anyio.run(close_client)
        except Exception:
            pass

def call_agent(input: str, agent: Agent = hpoa_simple_agent, tool_limit: int = 50):
    """Run a simplified agent synchronously with history persistence."""
    # Load persisted history
    load_history()
    try:
        result = agent.run_sync(
            input,
            deps=get_config(),
            usage_limits=UsageLimits(request_limit=tool_limit),
            message_history=MSG_HISTORY or None,
        )
        # Same logic: include system prompt on first call, else just new messages
        if not MSG_HISTORY:
            MSG_HISTORY.extend(result.all_messages())
            save_history()
        else:
            append_new_messages(result.new_messages())
        return result
    finally:
        try:
            anyio.run(close_client)
        except Exception:
            pass