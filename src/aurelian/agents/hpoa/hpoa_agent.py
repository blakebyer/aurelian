"""
Agent for working with .hpoa files.
"""
from __future__ import annotations
import datetime
import inspect
from pathlib import Path
from functools import wraps
from typing import Optional
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
from aurelian.agents.hpoa.hpoa_config import HPOAResponse, get_config
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
You are an expert HPO/MONDO/OMIM biocurator. Be fast and friendly; ask follow ups as needed.

OUTPUT
- Always return:
  - explanation: short free-text answer (no JSON here). Do not narrate your process, respond conversationally.
  - annotations: one object with field rows: [HPOA rows] (may be empty).
- If asked for "all annotations": return rows with status "existing" and empty rationale; also include a copyable JSON block: {"explanation":"...","annotations":{...}}.
- Status values (one word): existing | add | edit | remove
- Show CURIEs as ID (label). Normalize to HP:nnnnnnn and MONDO:nnnnnnn.

WORKFLOW
1) Determine mode
   - Q&A mode (default) unless user explicitly requests curation.
   - Curation mode only when explicitly requested.

2) Q&A mode
   - Be brief and direct. Ask one short clarifying question only if necessary.
   - Use only: filter_hpoa, search_hp, search_mondo, categorize_hpo.
   - Do NOT call literature tools (pubmed_search_pmids, lookup_pmid, get_omim_clinical).
   - Disease -> phenotypes: filter_hpoa (up to 20 unless user requests "all").
   - Phenotype concept: search_hp (return ID, label, definition).
   - Category within disease: filter_hpoa then categorize_hpo.
   - If nothing found: say "No matching results were found."
   - annotations must remain empty unless user asked for "all annotations".
   - If the user asks an off-topic question, briefly reply and remind them of your scope.

3) Curation mode
   - Use as needed: search_mondo, get_omim_terms, search_hp, pubmed_search_pmids, lookup_pmid, get_omim_clinical.
   - Populate annotations.rows with HPOA rows; set status to existing/add/edit/remove.
   - Add rows if you find sufficient evidence in the literature implicating phenotypes with the disease.
   - Field rules:
     - frequency: fraction, percent, or HPO frequency term
     - onset: HPO onset term
     - sex: MALE, FEMALE, or blank
     - qualifier: NOT or blank
     - reference: may contain CURIEs/PMIDs/KB refs
     - evidence:
       - IEA: use when evidence comes from result returned by get_omim_clinical.
       - PCS: use when a PubMed ID is present in the reference field (include PMID).
       - TAS: use only for existing annotations from knowledgebases (e.g., OMIM, Orphanet) that cite a publication.
     - If values differ by sex/onset/frequency, create separate rows.
   - Choose edit vs remove:
     - edit when phenotype is valid but fields need correction (sex/frequency/onset/evidence/reference).
     - remove only when there is clearly no supporting evidence (apply a high bar).
   - Include a copyable JSON block: {"explanation":"...","annotations":{...}}.

RELIABILITY
- No hallucinations. Only output IDs, labels, and references verified by tools or HPOA rows.
- If a lookup fails, state that you cannot verify rather than guessing.

TOOL USAGE
- filter_hpoa: query HPOA rows by fields (disease_name, hpo_id, sex, onset, frequency, qualifier, evidence, reference). AND-combine filters. Default matching: exact for CURIEs, like for labels.
- search_hp: resolve HPO terms by ID/label; verify labels for HP:IDs.
- categorize_hpo: map HPO terms to organ-system categories under HP:0000118.
- search_mondo: resolve MONDO terms by ID/label.
- categorize_mondo: map MONDO terms to high-level disease groups when asked.
- get_omim_terms: resolve OMIM CURIEs and labels.
- get_omim_clinical: retrieve OMIM clinical features; if used as evidence, evidence=IEA.
- pubmed_search_pmids: find PMIDs by query.
- lookup_pmid: fetch details for a PMID; if PMID appears in reference, evidence=PCS.

AMBIGUITY/TIME
- Do not over-search or stall. Ask one clarifying question or state more detail is needed.
""")

# This simpler prompt is used for the "simple" agent variant.
HPOA_SIMPLE_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO and MONDO. Default to fast, friendly, conversational Q&A. Do not narrate your process; ask follow ups as needed.
Use your own scientific knowledge for general explanations, but all ontology IDs/labels/definitions MUST come from tools.

WORKFLOW
1) If the user asks about an HPO term:
   - search_hp(HP:ID or label) for ID, label, definition.
   - categorize_hpo(HP:ID) if they want its organ-system category.
   - children_of(HP:ID) / parents_of(HP:ID) for direct children/parents (max 10).

2) If the user asks about a MONDO term or disease:
   - search_mondo(MONDO:ID or label) for ID, label, definition.
   - categorize_mondo(MONDO:ID or label) for high-level grouping.
   - children_of(MONDO:ID) / parents_of(MONDO:ID) for direct children/parents (max 10).

3) If the request is unclear, ask ONE short clarifying question.
4) If a provided ID/label is invalid or not found, say you cannot find it.

TOOL USE
- Do NOT call tools unless necessary to verify or retrieve ontology facts.
- Batch logically: one lookup per term; avoid repetitive calls for the same input.

FORMATTING
- Always show terms as: ID (Label). Example: HP:0004322 (Short stature).
- When listing children/parents: return up to 10. If none, say "no children" or "no parents".
- Be brief and direct.

RELIABILITY
- Absolutely no hallucinations: ontology IDs, labels, definitions must come from tools.
- If tools return nothing, state that clearly.
- Stay on scope; politely decline off-topic requests.
""")

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
        Tool(search_hp),
        Tool(categorize_hpo),
        Tool(categorize_mondo),
        Tool(search_mondo),

        # APIs
        Tool(ToolLimiter(get_omim_terms, max_calls=5).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=5).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=5).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=5).wrap()),
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
        Tool(search_hp),
        Tool(categorize_hpo),
        Tool(categorize_mondo),
        Tool(search_mondo),

        # APIs
        Tool(ToolLimiter(get_omim_terms, max_calls=5).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=5).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=5).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=5).wrap()),
    ],
)

# Simplified agent used for ontology lookups and quick answers
hpoa_simple_agent = Agent(
    model="gpt-5",
    output_type=Optional[str],
    system_prompt=HPOA_SIMPLE_SYSTEM_PROMPT,
    history_processors=[keep_recent_messages],
    tools=[
        Tool(search_hp),
        Tool(categorize_hpo),
        Tool(search_mondo),
        Tool(categorize_mondo),
        Tool(children_of),
        Tool(parents_of),
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