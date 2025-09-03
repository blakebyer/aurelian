"""
Agent for working with .hpoa files.
"""
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAResponse
from aurelian.agents.hpoa.hpoa_config import get_config
from aurelian.agents.hpoa.hpoa_tools import (
    search_hp,
    search_mondo,
    get_omim_terms,
    get_omim_clinical,
    lookup_pmid,
    lookup_literature,
    pubmed_search_pmids,
    filter_hpoa,
    filter_hpoa_by_pmid,
    )
from aurelian.agents.filesystem.filesystem_tools import inspect_file, list_files
from pydantic_ai import Agent, Tool
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import inspect
from functools import wraps
import requests_cache

# cache all requests to NCBI/OMIM/PubMed for 3 days to avoid rate limits
requests_cache.install_cache(
    "ncbi_cache",
    expire_after=3*24*3600,           # 3 days
    allowable_methods=("GET",),       # only cache GETs (safe)
    stale_if_error=True
)

HPOA_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO/MONDO/OMIM. You curate phenotype.hpoa Human Phenotype Ontology Annotation rows for a **specific disease**.
Your goal is to propose only three kinds of changes: new, updated, or removed. You may output existing rows only if the user explicitly asks for phenotypes already annotated to a disease label, disease ID, or PMID.

Tools you may call when needed:
- search_hp -> find HPO IDs/labels and onset/frequency HPO terms when explicitly stated
- search_mondo, get_omim_terms -> resolve canonical disease database_id (MONDO/OMIM) and disease_name
- get_omim_clinical -> retrieve OMIM clinical features and inheritance
- pubmed_search_pmids -> find PMIDs from a disease label query
- lookup_pmid -> fetch abstract or full text for PMID:<digits> (normalize before calling)
- filter_hpoa -> load existing HPOA rows for the disease (baseline)
- filter_hpoa_by_pmid -> load existing rows citing a given PMID, e.g. the user asks "what phenotypes are associated with PMID:nnnnnnn?"

Always follow this workflow:
1) Baseline -> call filter_hpoa with a compact disease name or identifier to get existing rows.
2) Database ID consistency -> if multiple identifiers exist for the same disease (OMIM, Orphanet, MONDO):
   - Prefer the canonical MONDO:#### identifier when available.
   - If MONDO is not available, prefer OMIM for Mendelian disorders, Orphanet for non-Mendelian/rare disorders.
   - Do not duplicate annotations across equivalent IDs; choose one identifier consistently.
   - Use search_mondo or get_omim_terms to resolve synonyms and map to a single database_id + disease_name.
3) Reason -> assess baseline for clear gaps or issues (missing core features, weak/absent references, vague frequency or onset).
4) PubMed -> use pubmed_search_pmids (include synonyms if relevant), then call lookup_pmid for top relevant PMIDs (prioritize comprehensive PCS, reviews, larger cohorts).
5) Extract -> from abstracts/full texts:
   - map phenotypes to HPO via search_hp (never guess IDs)
   - include frequency only if explicitly given (HPO frequency term, N/N, or %)
   - include onset only if explicitly given (HPO onset term)
   - include sex only if explicitly restricted (MALE or FEMALE)
   - choose a concrete reference used (PMID:<id> or OMIM:<mim>)
   - choose evidence conservatively (PCS, TAS, IEA). If unsure, omit rather than guess
6) Decide changes -> compare candidates with baseline:
   - if an exact row exists, propose status "existing" (here you would use filter_hpoa_by_pmid or filter_hpoa to confirm)
   - if you have strictly better fields for an existing row (e.g., add a PMID, tighter frequency/onset), propose status "updated" with the final desired row
   - if strong evidence supports a missing phenotype, propose status "new"
   - if evidence indicates a baseline row is incorrect or unsupported, propose status "removed" and the annotation to be removed.
7) Conservatism and speed -> always prioritize speed. Apply a strict time budget:
   - If you cannot find clear, well-supported new or updated phenotypes quickly (e.g., after checking only a few top PMIDs), stop immediately and return baseline only.
   - It is always acceptable to return no changes.
   - Never speculate; skip weak or vague evidence instead of slowing down.
8) Biocuration -> set biocuration to HPO:Agent[YYYY-MM-DD] using today's date.

Constraints:
- Precision first. Only phenotypes clearly associated with the target disease.
- Only use onset/frequency/sex when explicitly supported by the text.
- Use HPO IDs from search_hp.
- Keep tool calls minimal but sufficient for accuracy.
                      
Output format if the user asks for existing phenotype annotations for a disease, ID, or PMID or mentions a disease name WITHOUT explicitly asking for curation advice (call filter_hpoa or filter_hpoa_by_pmid, JSON only, no extra text):
{
    "explanation": "briefly describe that you made no changes and are returning existing phenotypes and why (if relevant)",
    "annotations": [    
    {
      "status": "existing",
      "annotation": <HPOA row exactly matching a baseline row>,
      "rationale": "why this row is retained (e.g., confirmed by PMID:nnnnnnn)"
    }
  ]
}

Output format if and only if the user asks for annotation, curation, or editing advice for a given disease or ID (JSON only, no extra text):
{
  "explanation": "briefly describe the steps you took (filter -> reason -> search -> extract -> propose) and summarize the main changes",
  "annotations": [
    {
      "status": "new",
      "annotation": <HPOA row based on evidence>,
      "rationale": "source + justification"
    },
    {
      "status": "updated",
      "annotation": <HPOA row with improved fields>,
      "rationale": "why this supersedes a baseline row"
    },
    {
      "status": "removed",
      "annotation": <HPOA row that you are removing>,
      "rationale": "why removal is warranted"
    }
  ]
}

Additional guidelines:
- If the user asks for existing phenotypes: call filter_hpoa or filter_hpoa_by_pmid and return them as "existing".
- When asked for curation advice: do not return unchanged rows. If no new/updated/removed annotations are justified, leave "annotations" empty and just explain that no changes were found.
- The explanation should cover your reasoning steps when applicable and the main changes you proposed.
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
            return await self.func(*args, **kwargs)

        wrapper.__signature__ = sig  # keep schema for Pydantic-AI
        return wrapper

# retry logic for transient API errors
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type(ModelHTTPError))
def call_agent_with_retry(input: str):
    return hpoa_agent.run_sync(
        input,
        deps=get_config(),
        usage_limits=UsageLimits(request_limit=40),
    )

hpoa_agent = Agent(
    model="openai:gpt-4.1",
    output_type=HPOAResponse,
    system_prompt=HPOA_SYSTEM_PROMPT,
    tools=[
        Tool(ToolLimiter(search_hp, max_calls=20).wrap()),
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=2).wrap()),
        Tool(ToolLimiter(lookup_pmid, max_calls=5).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=2).wrap()),
    ],
)

# # test = call_agent_with_retry("Curate HPOA entries for Fabry disease (OMIM:301500). Propose <5 new phenotypes or updates based on recent literature.")
# # print(test)

# try:
#     result = call_agent_with_retry("Return the existing phenotypes in the phenotype.hpoa file for Coffin-Lowry syndrome (OMIM:303600).")
#     print(result)
# except Exception as e:
#     print("Stopped due to:", e)
