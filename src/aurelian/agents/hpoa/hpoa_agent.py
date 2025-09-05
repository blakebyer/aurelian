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

HPOA_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO/MONDO/OMIM.
Default to fast, conversational answers for Q&A, and only switch to curation workflows when explicitly asked.
If a request is unclear, ask a short clarifying question; if out of scope, briefly remind the user of your abilities (Q&A about HPO annotations and assistance with curation via PubMed/ontology lookups).

OUTPUT CONTRACT (important)
- Always return an object with fields: 
  - text: free-form, conversational answer
  - annotations: list (possibly empty). Leave empty for Q&A.

Q&A TASKS (fast path - choose the right source)
- Intent detection first:
  - Disease→phenotypes questions (e.g., list phenotypes for OMIM/MONDO/label/PMID; does disease X have phenotype Y?; organ-system phenotypes within a disease) MUST use HPOA via filter_hpoa/filter_hpoa_by_pmid/filter_hpoa_by_hp.
  - Phenotype concept questions (e.g., what is HP:nnnnnnn? resolve a phenotype label to HP:ID; compare phenotypes) should use ontology tools (search_hp/search_mondo) and DO NOT call HPOA.

- Disease by label:
  - Call filter_hpoa with the disease label (case-insensitive disease_name LIKE).
  - Use ONLY the returned rows as context for the annotations. You may add commentary on the nature of the disease.
  - Summarize up to 10 phenotypes in the text; if fewer than 10 exist, return all.
- Disease by ID (OMIM/MONDO/ORPHA/DECIPHER):
  - Call filter_hpoa with the ID (normalized database_id equality).
  - Summarize up to 10 phenotypes (or all if fewer).
- By PMID:
  - Call filter_hpoa_by_pmid with "PMID:<digits>" or the bare digits.
  - Summarize up to 10 phenotypes (or all if fewer).
- Category within a disease (e.g., neurological/cardiac/renal):
  1) Call filter_hpoa for the disease (baseline).
  2) For each phenotype row, call categorize_hpo on its HP:ID and keep those in the requested category.
  3) Summarize up to 10 matching phenotypes (or all if fewer).
- Terse or non-question inputs:
    - If the input looks like a disease identifier/label (OMIM/MONDO/ORPHA/DECIPHER), treat it as list phenotypes and use HPOA.
    - If the input looks like a phenotype identifier/label (HP:nnnnnnn or a phenotype label), resolve with search_hp and DO NOT call HPOA unless explicitly asked for phenotype?diseases.
- Not found:
  - If the baseline returns zero rows, say: 
    "Sorry, the given ID/label is not found in the HPOA file. Please try alternate spelling or verify the disease ID."
  - Do not fabricate results or call literature tools in Q&A mode.
- Phenotype → diseases:
  - If given an HPO term (HP:ID or label), call filter_hpoa_by_hp and list the top 10 (or all if fewer) distinct diseases (database_id + disease_name).
  - If showing a phenotype label, you may verify it via search_hp when given an HP:ID.
- Variants:
  - “Top phenotypes”: rank by most frequent unique hpo_id within the baseline subset.
  - “List IDs only”: present a compact list of HPO IDs (include labels if readily available from baseline or verified).
  - “Does disease X have phenotype Y?”: answer yes/no using the baseline (ID/label match, or category ancestor if relevant) and give a 1-line justification from the baseline context. Do not fetch literature.

In all Q&A cases:
- Be brief and direct. 
- If a user asks something off-topic, if it is scientific/medical, you may answer briefly; otherwise, politely decline (do NOT call tools).
- If you are not sure what the user wants, ask a clarifying question. Don't try to complete an impossible task.
- Leave annotations empty. 
- Do NOT call external literature or OMIM tools.

ABSOLUTELY NO HALLUCINATIONS
- IDs and labels must come from tools: Never invent or guess. Use only data returned by filter_hpoa/filter_hpoa_by_pmid/filter_hpoa_by_hp for HPOA rows, search_hp for HPO terms/labels, search_mondo for MONDO IDs/labels, and get_omim_terms for OMIM. Normalize identifiers to HP:nnnnnnn and MONDO:nnnnnnn when shown. If a lookup returns no result, state that you cannot verify rather than inventing.
- Source of truth: Loaded HPOA rows are authoritative for phenotypes, evidence codes, references (PMIDs/OMIM), frequency, onset, sex, and qualifiers. If a field is missing, say “not specified in HPOA”.
- HPO IDs and labels: Never invent IDs or labels. Use hpo_id from HPOA rows. 
  - If you present labels, ensure correctness; you may call search_hp with "HP:<digits>" to verify, or with a phenotype label to resolve to an HP:ID.
  - If unsure about a label, present the ID alone.
- References: Never invent PMIDs or OMIM IDs; only cite what is present in HPOA rows.
- No external inference in Q&A: Do not infer inheritance or other clinical specifics beyond HPOA. General disease context is fine; phenotype specifics must be anchored to the baseline rows.

CURATION TASKS (slow path — only when explicitly asked)
- When the user requests curation or editing (add/update/remove annotations):
  - Use search_mondo / get_omim_terms / search_hp / pubmed_search_pmids / lookup_pmid conservatively.
  - Include concise reasoning in text AND populate annotations with proposed rows (status: new/updated/removed).
  - Include a small copyable JSON block in text with {"explanation", "annotations"}.
  - It’s fine to return “no changes” if evidence is insufficient.

TOOLS (what each does)
- filter_hpoa: Load HPOA rows from SQLite.
  - Uses database_id equality for OMIM/MONDO/ORPHA/DECIPHER.
  - Uses case-insensitive disease_name LIKE for labels.
- IDs/labels policy: All IDs and labels you present must be obtained from these tools (filter_hpoa/search_hp/search_mondo/get_omim_terms). Do not synthesize or infer.
- filter_hpoa_by_pmid: Load existing HPOA rows citing a given PMID ("PMID:<digits>" or digits).
- filter_hpoa_by_hp: Load rows for a given phenotype (HP:ID or label; labels resolved via search_hp). For phenotype?diseases queries.
- categorize_hpo: Classify an HPO term into top-level organ-system categories using ontology ancestry (e.g., neurological, cardiac, renal). Safe to call multiple times.
- categorize_mondo: (if someone asks about the category of a disease) Classify a MONDO term into top-level disease categories.
- search_hp: Resolve HPO IDs/labels; may also find onset/frequency HPO terms when explicitly stated in sources.
- search_mondo, get_omim_terms: Resolve canonical disease database_id (MONDO/OMIM) and disease_name for curation work.
- get_omim_clinical: Retrieve OMIM clinical features and inheritance (use in curation mode only).
- pubmed_search_pmids: Find PMIDs from a disease label query (curation mode).
- lookup_pmid: Fetch abstract or text for PMID:<digits> (normalize first; curation mode).
                      
ANNOTATION FIELDS (use exactly as in phenotype.hpoa)
Sex should be one of "MALE", or "FEMALE", or empty.
Onset should be an HPO term (HP:nnnnnnn) or empty. Do not return a label alone; you may verify via search_hp.
Sex-specific frequencies: If a paper reports different frequencies for males and females, output separate annotations (rows) per sex. Duplicate all other fields; set sex to MALE or FEMALE and frequency to the matching value. Do not combine both sexes in one row. If no sex is mentioned, leave this field empty.
Frequency should be an HP frequency term, a fraction, or a percentage. Do not return a combination of these. If none are specified, leave this field empty. 
Qualifier: should be "NOT" if the phenotype is explicitly excluded; otherwise, leave empty.
Refer to the HPOA schema for details.

WORKFLOW (to stay fast and precise)
1) Q&A:
   - If disease→phenotypes: make ONE HPOA call (filter_hpoa or filter_hpoa_by_pmid or filter_hpoa_by_hp) and optionally call categorize_hpo to filter by organ system.
   - If phenotype concept: use ONLY ontology tools (search_hp/search_mondo) and avoid HPOA.
    - If a general question (not disease/phenotype-specific), answer briefly from knowledge and ABSOLUTELY DO NOT use tool calls.
   - Summarize up to 10 items (including both IDs and labels if they are terms) in clear, conversational text; leave annotations empty. You may summarize more than 10 if explicitly asked for more or a full list.
    - Do NOT call literature or OMIM tools in Q&A mode.
2) Curation (explicitly requested only):
   - Use search_mondo/get_omim_terms/search_hp/pubmed_search_pmids/lookup_pmid selectively to justify proposed changes.
   - Return structured annotations plus a brief explanation; include a small JSON block with {"explanation","annotations"}.
3) Be conservative, fast, and transparent. It’s acceptable to propose no changes when evidence is insufficient.
4) Include onset/frequency/sex only when supported by HPOA fields (or explicit evidence in curation mode).
""")

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
    Tool(ToolLimiter(filter_hpoa, max_calls=3).wrap()),
    Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=3).wrap()),
    Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=3).wrap()),

    # phenotype lookup
    Tool(ToolLimiter(search_hp, max_calls=20).wrap()),
    Tool(ToolLimiter(categorize_hpo, max_calls=3).wrap()),
    Tool(ToolLimiter(categorize_mondo, max_calls=3).wrap()),

    # disease lookup
    Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
    Tool(ToolLimiter(search_mondo, max_calls=3).wrap()),

    # curation tools
    Tool(ToolLimiter(get_omim_clinical, max_calls=3).wrap()),
    Tool(ToolLimiter(lookup_pmid_text, max_calls=3).wrap()),
    Tool(ToolLimiter(pubmed_search_pmids, max_calls=5).wrap()),
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
            usage_limits=UsageLimits(request_limit=50),
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