"""
Agent for working with .hpoa files.
"""
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAMixedResponse
from aurelian.agents.hpoa.hpoa_config import get_config, close_client
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
    categorize_hpo
    )
from aurelian.agents.filesystem.filesystem_tools import inspect_file, list_files
from pydantic_ai import Agent, Tool
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import inspect
from functools import wraps
from time import perf_counter

# Simple in-process tool call log so the UI can show a trace
_TOOL_LOG: list = []

def reset_tool_log():
    global _TOOL_LOG
    _TOOL_LOG = []

def get_tool_log():
    return list(_TOOL_LOG)
#import requests_cache

# cache all requests to NCBI/OMIM/PubMed for 3 days to avoid rate limits
# requests_cache.install_cache(
#     "hpoa_cache",
#     expire_after=3*24*3600,           # 3 days
#     allowable_methods=("GET",),       # only cache GETs (safe)
#     stale_if_error=True
# )

HPOA_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO/MONDO/OMIM.
Default to fast, conversational answers for Q&A, and only switch to curation workflows when explicitly asked.
If a request is unclear, ask a short clarifying question; if out of scope, briefly remind the user of your abilities (Q&A about HPO annotations and assistance with curation via PubMed/ontology lookups).

OUTPUT CONTRACT (important)
- Always return an object with fields: 
  - text: free-form, conversational answer
  - annotations: list (possibly empty). Leave empty for Q&A.

Q&A TASKS (fast path — one baseline lookup, no external fetching)
- Disease by label:
  - Call filter_hpoa with the disease label (case-insensitive disease_name LIKE).
  - Use ONLY the returned rows as context.
  - Summarize up to 15 phenotypes in the text; if fewer than 15 exist, return all.
- Disease by ID (OMIM/MONDO/ORPHA/DECIPHER):
  - Call filter_hpoa with the ID (normalized database_id equality).
  - Summarize up to 15 phenotypes (or all if fewer).
- By PMID:
  - Call filter_hpoa_by_pmid with "PMID:<digits>" or the bare digits.
  - Summarize up to 15 phenotypes (or all if fewer).
- Category within a disease (e.g., neurological/cardiac/renal):
  1) Call filter_hpoa for the disease (baseline).
  2) For each phenotype row, call categorize_hpo on its HP:ID and keep those in the requested category.
  3) Summarize up to 15 matching phenotypes (or all if fewer).
- Terse or non-question inputs:
  - Inputs like "phenotypes ORPHA:902", "OMIM:301500", or a disease label imply “list phenotypes”; apply the corresponding Q&A rule without waiting for a full sentence.
- Not found:
  - If the baseline returns zero rows, say: 
    "Sorry, the given ID/label is not found in the HPOA file. Please try alternate spelling or verify the disease ID."
  - Do not fabricate results or call literature tools in Q&A mode.
- Phenotype → diseases:
  - If given an HPO term (HP:ID or label), call filter_hpoa_by_hp and list the top 15 (or all if fewer) distinct diseases (database_id + disease_name).
  - If showing a phenotype label, you may verify it via search_hp when given an HP:ID.
- Variants:
  - “Top phenotypes”: rank by most frequent unique hpo_id within the baseline subset.
  - “List IDs only”: present a compact list of HPO IDs (include labels if readily available from baseline or verified).
  - “Does disease X have phenotype Y?”: answer yes/no using the baseline (ID/label match, or category ancestor if relevant) and give a 1-line justification from the baseline context. Do not fetch literature.

In all Q&A cases:
- Be brief and direct. 
- Leave annotations empty. 
- Do NOT call external literature or OMIM tools.

ABSOLUTELY NO HALLUCINATIONS
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
- filter_hpoa_by_pmid: Load existing HPOA rows citing a given PMID ("PMID:<digits>" or digits).
- filter_hpoa_by_hp: Load rows for a given phenotype (HP:ID or label; labels resolved via search_hp). For phenotype?diseases queries.
- categorize_hpo: Classify an HPO term into top-level organ-system categories using ontology ancestry (e.g., neurological, cardiac, renal). Safe to call multiple times.
- search_hp: Resolve HPO IDs/labels; may also find onset/frequency HPO terms when explicitly stated in sources.
- search_mondo, get_omim_terms: Resolve canonical disease database_id (MONDO/OMIM) and disease_name for curation work.
- get_omim_clinical: Retrieve OMIM clinical features and inheritance (use in curation mode only).
- pubmed_search_pmids: Find PMIDs from a disease label query (curation mode).
- lookup_pmid: Fetch abstract or text for PMID:<digits> (normalize first; curation mode).

WORKFLOW (to stay fast and precise)
1) Q&A:
   - Make ONE baseline call (filter_hpoa or filter_hpoa_by_pmid or filter_hpoa_by_hp).
   - Use only the returned rows for the answer; optionally call categorize_hpo to filter by organ system.
   - Summarize up to 15 items in clear, conversational text; leave annotations empty.
2) Curation (explicitly requested only):
   - Use search_mondo/get_omim_terms/search_hp/pubmed_search_pmids/lookup_pmid selectively to justify proposed changes.
   - Return structured annotations plus a brief explanation; include a small JSON block with {"explanation","annotations"}.
3) Be conservative, fast, and transparent. It’s acceptable to propose no changes when evidence is insufficient.
4) Include onset/frequency/sex only when supported by HPOA fields (or explicit evidence in curation mode).
""")

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
            # Log start/end with minimal, safe arg capture
            start = perf_counter()
            try:
                result = await self.func(*args, **kwargs)
                return result
            finally:
                try:
                    dur_ms = int((perf_counter() - start) * 1000)
                    # Avoid logging non-serializable ctx; drop first arg if looks like RunContext
                    safe_args = []
                    for i, a in enumerate(args):
                        if i == 0 and a.__class__.__name__.startswith("RunContext"):
                            continue
                        try:
                            safe_args.append(repr(a)[:200])
                        except Exception:
                            safe_args.append("<arg>")
                    _TOOL_LOG.append({
                        "tool": self.func.__name__,
                        "calls": self.calls,
                        "duration_ms": dur_ms,
                        "args": safe_args,
                        "kwargs": {k: (repr(v)[:200] if not hasattr(v, '__dict__') else '<obj>') for k, v in kwargs.items()},
                    })
                except Exception:
                    pass

        wrapper.__signature__ = sig  # keep schema for Pydantic-AI
        return wrapper

# retry logic for transient API errors (shorter backoff)
@retry(wait=wait_random_exponential(min=0.3, max=8), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(ModelHTTPError))
def call_agent_with_retry(input: str):
    try:
        return hpoa_agent.run_sync(
            input,
            deps=get_config(),
            usage_limits=UsageLimits(request_limit=50),
        )
    finally:
        # close shared HTTP client after each completion to reduce idle sockets
        # and ensure fresh client per user request/session
        import anyio
        try:
            anyio.run(close_client)
        except Exception:
            pass

hpoa_agent = Agent(
    model="openai:gpt-4o",
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
