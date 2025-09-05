"""
Agent for working with .hpoa files.
"""
from pathlib import Path
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, TextPart, UserPromptPart 
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
    )
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from typing import List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import inspect
from functools import wraps

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
    - Summarize up to 10 phenotypes (return all if =10)
  - Phenotype concept (e.g., what is HP:0001250? map a phenotype label to HP:ID; compare phenotypes):
    - Use ontology tools only: search_hp / search_mondo; do NOT call HPOA
  - Category within a disease:
    - Make one baseline HPOA call (filter_hpoa), then call categorize_hpo(HP:ID) to keep matches; summarize up to 10
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
   - Disease?phenotypes: one HPOA call (filter_hpoa / filter_hpoa_by_pmid / filter_hpoa_by_hp); optionally categorize_hpo
   - Phenotype concept: use only ontology tools (search_hp/search_mondo)
   - No tools for general/off-topic questions
   - Summarize up to 10; leave annotations empty; do not call literature/OMIM tools
   - If a user asks a question, try to answer it. Do not say you are "going to" do something and terminate.
2) Curation (on request): use search_mondo/get_omim_terms/search_hp/pubmed_search_pmids/lookup_pmid sparingly; return 10 or fewer annotations + short explanation.
3) Be conservative and transparent; acceptable to propose no changes
4) Include onset/frequency/sex only when supported by HPOA or explicit evidence in curation"""
)

MSG_HISTORY: list[ModelMessage] = []  # keep last few messages for context
HISTORY_PATH = Path("history.json")

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
                # Instead of crashing, return an error dict the model can see
                return {"error": f"{self.func.__name__} exceeded {self.max_calls} calls"}
            self.calls += 1
            return await self.func(*args, **kwargs)

        wrapper.__signature__ = sig  # keep schema for Pydantic-AI
        return wrapper

# Configure OpenAI reasoning model with summary to expose in responses
oai_model = OpenAIResponsesModel("gpt-5-mini")
oai_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort="low",
    openai_reasoning_summary="concise",
)

hpoa_agent = Agent(
    model=oai_model,
    model_settings=oai_settings,
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    tools=[
        # baseline
        Tool(ToolLimiter(filter_hpoa, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=2).wrap()),
        Tool(ToolLimiter(search_hp, max_calls=20).wrap()),
        Tool(ToolLimiter(categorize_hpo, max_calls=50).wrap()),
      
        # disease lookup
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=2).wrap()),
        
        # curation tools
        Tool(ToolLimiter(get_omim_clinical, max_calls=2).wrap()),
        Tool(ToolLimiter(lookup_pmid_text, max_calls=5).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=2).wrap()),
    ],
)

simple_hpoa_agent = Agent(
    model="gpt-5-mini",
    output_type=HPOAMixedResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    tools = [
    # filtering
    Tool(ToolLimiter(filter_hpoa, max_calls=2).wrap()),
    Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=2).wrap()),
    Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=2).wrap()),

    # phenotype lookup
    Tool(ToolLimiter(search_hp, max_calls=25).wrap()),
    Tool(ToolLimiter(categorize_hpo, max_calls=25).wrap()),
    Tool(ToolLimiter(categorize_mondo, max_calls=2).wrap()),

    # disease lookup
    Tool(ToolLimiter(get_omim_terms, max_calls=2).wrap()),
    Tool(ToolLimiter(search_mondo, max_calls=2).wrap()),

    # curation tools
    Tool(ToolLimiter(get_omim_clinical, max_calls=2).wrap()),
    Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
    Tool(ToolLimiter(pubmed_search_pmids, max_calls=2).wrap()),
  ],
)

# retry logic for transient API errors (shorter backoff)
@retry(wait=wait_random_exponential(min=0, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(ModelHTTPError))
def call_agent_with_retry(input: str):
    global MSG_HISTORY
  
    def create_context(messages: list[ModelMessage]) -> list[ModelMessage]:
      """Remove all but the last 2 messages to keep context."""
      return [msg for msg in messages if isinstance(msg, TextPart) or isinstance(msg, UserPromptPart)][-2:]
    
    try:
        result = simple_hpoa_agent.run_sync(
            input,
            deps=get_config(),
            message_history=MSG_HISTORY or None,
            history_processors=[create_context],
            usage_limits=UsageLimits(request_limit=75),
        )

        # append the new messages
        MSG_HISTORY.extend(result.new_messages())

        # save whole history as pretty JSON
        HISTORY_PATH.write_bytes(
            ModelMessagesTypeAdapter.dump_json(MSG_HISTORY, indent=2)
        )

        return result
    finally:
        # close shared HTTP client after each completion to reduce idle sockets
        # and ensure fresh client per user request/session
        import anyio
        try:
            anyio.run(close_client)
        except Exception:
            pass
