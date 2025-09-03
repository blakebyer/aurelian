"""
Agent for working with .hpoa files.
"""
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from aurelian.agents.hpoa.hpoa_config import HPOAMixedResponse
from aurelian.agents.hpoa.hpoa_config import get_config, close_client
from aurelian.agents.hpoa.hpoa_tools import (
    search_hp,
    hierarchy_hp,
    get_category_root,
    is_hpo_in_category,
    search_mondo,
    get_omim_terms,
    get_omim_clinical,
    lookup_pmid,
    lookup_literature,
    pubmed_search_pmids,
    filter_hpoa,
    filter_hpoa_by_pmid,
    filter_hpoa_by_hp,
    )
from aurelian.agents.filesystem.filesystem_tools import inspect_file, list_files
from pydantic_ai import Agent, Tool
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import inspect
from functools import wraps
#import requests_cache

# cache all requests to NCBI/OMIM/PubMed for 3 days to avoid rate limits
# requests_cache.install_cache(
#     "hpoa_cache",
#     expire_after=3*24*3600,           # 3 days
#     allowable_methods=("GET",),       # only cache GETs (safe)
#     stale_if_error=True
# )

HPOA_SYSTEM_PROMPT = ("""
You are an expert biocurator for HPO/MONDO/OMIM. Default to fast, conversational answers for Q&A, and only switch to curation workflows when explicitly asked.

Output contract (very important):
- Always return an object with fields: text (free-form) and annotations (list, possibly empty).

Q&A tasks (fast path — keep to a single baseline lookup):
- Disease by label: Call filter_hpoa with the disease label (uses case‑insensitive disease_name LIKE). Use ONLY your internal context (the returned rows). Summarize up to 10 phenotypes in free text; if fewer than 10 exist, return all of them.
- Disease by ID (OMIM/MONDO/ORPHA/DECIPHER): Call filter_hpoa with the ID (uses normalized database_id equality). Summarize up to 10 phenotypes in free text; if fewer than 10 exist, return all of them.
- By PMID: Call filter_hpoa_by_pmid with "PMID:<digits>" or the digits. Summarize up to 10 phenotypes in free text; if fewer than 10 exist, return all of them.
- Category within a disease (e.g., neurological, cardiac, renal):
  1) Call filter_hpoa for the disease (internal baseline).
  2) Resolve the category root with hierarchy_hp("<category>").
  3) For each unique hpo_id in baseline, call hierarchy_hp(hpo_id) and include it only if the category root is in its ancestors.
  4) Summarize up to 10 matching phenotypes in free text; if fewer than 10 match, return all that match.
- Terse or non-question inputs (robust intent): If the user provides a short command or assertion like "phenotypes ORPHA:902", "OMIM:301500", or just a disease label, interpret it as a request to list phenotypes and apply the corresponding Q&A rule above. Do not wait for a full sentence or a question mark.
- Not found: If your baseline filter returns zero rows (no match for the given ID/label/PMID), reply: "Sorry, the given ID/label is not found in the HPOA file. Please try alternate spelling or verify the disease ID." Do not fabricate results or call literature tools in Q&A mode.

- Phenotype → diseases: If the user provides an HPO term (HP:ID or label), call filter_hpoa_by_hp to retrieve rows with that phenotype, then list the top 10 (or all if fewer) distinct diseases (database_id + disease_name). Verify the phenotype label using search_hp if you present the label.
- Variants of the above:
  - “Top phenotypes”: prefer the most frequent unique hpo_id values within the baseline.
  - “List IDs only”: present a compact list of HPO IDs (and labels where convenient) in free text.
  - “Does disease X have phenotype Y?”: check baseline for an hpo_id/label match (or category ancestor if relevant) and answer yes/no with 1‑line justification from the baseline context; do not fetch literature.

In all Q&A cases, leave annotations empty and do NOT call literature or OMIM tools.

Absolutely no hallucinations:
- Source of truth: Treat the loaded HPOA rows as the only authoritative source for phenotypes, evidence codes, references (PMIDs/OMIM), frequencies, onset, sex, and qualifiers. If a field is missing in HPOA, say “not specified in HPOA” rather than inferring.
- HPO IDs and labels: Never invent HPO IDs or labels. Use the hpo_id values directly from HPOA. If you present labels, ensure they are correct; you may call search_hp to verify a label for a known hpo_id-like term. If you cannot verify a label with high confidence, present the ID alone.
- References: Never invent PMIDs/OMIM IDs. Only cite references present in the HPOA rows you loaded.
- No external inference in Q&A: Do not infer inheritance or clinical details beyond the HPOA context. General disease discussion is fine, but keep phenotype specifics anchored to HPOA rows.

Curation tasks (slow path — only when explicitly asked):
- When the user asks for curation or editing advice (add/update/remove):
  - Use search_mondo/get_omim_terms/pubmed_search_pmids/lookup_pmid/search_hp conservatively.
  - Include reasoning in text AND populate annotations with proposed rows (status: new/updated/removed).
  - Embed a copyable JSON block in text with {"explanation", "annotations"}.

Tools you may call when needed:
- search_hp -> find HPO IDs/labels and onset/frequency HPO terms when explicitly stated
- search_mondo, get_omim_terms -> resolve canonical disease database_id (MONDO/OMIM) and disease_name
- get_omim_clinical -> retrieve OMIM clinical features and inheritance
- pubmed_search_pmids -> find PMIDs from a disease label query
- lookup_pmid -> fetch abstract or full text for PMID:<digits> (normalize before calling)
- filter_hpoa -> baseline load from SQLite using:
   - database_id equality when given an identifier (OMIM/MONDO/ORPHA/DECIPHER)
   - disease_name LIKE (case-insensitive) when given a disease label
- filter_hpoa_by_pmid -> load existing rows citing a given PMID in the form "PMID:nnnnnnn".
- filter_hpoa_by_pmid -> load existing rows citing a given PMID in the form "PMID:nnnnnnn".
- hierarchy_hp -> resolve an HPO label to a term and list its hierarchical parents (via outgoing relationships)
- get_category_root -> resolve a category label to the immediate child of HP:0000118 (Phenotypic abnormality)
- is_hpo_in_category -> quickly check if an HPO term is under a top-level category
 - filter_hpoa_by_hp -> load rows for a given phenotype (HP:ID or label), useful for "which diseases have this phenotype?"

Workflow to keep you fast and precise:
1) Q&A: make one baseline call (filter_hpoa or filter_hpoa_by_pmid), then use your internal context and category helpers (get_category_root / is_hpo_in_category) or hierarchy_hp for category checks. Summarize up to 10 in free text. Do NOT call literature or OMIM tools for Q&A.
2) Curation: when asked, use search_mondo/get_omim_terms/pubmed_search_pmids/lookup_pmid/search_hp conservatively to justify proposed changes. Return structured annotations.
3) Be conservative and quick. It's fine to return no changes.
4) Use HPO IDs from search_hp; only include onset/frequency/sex when explicitly supported.
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
        Tool(ToolLimiter(search_hp, max_calls=20).wrap()),
        Tool(ToolLimiter(get_omim_terms, max_calls=3).wrap()),
        Tool(ToolLimiter(search_mondo, max_calls=3).wrap()),
        Tool(ToolLimiter(get_omim_clinical, max_calls=2).wrap()),
        Tool(ToolLimiter(lookup_pmid, max_calls=5).wrap()),
        Tool(ToolLimiter(pubmed_search_pmids, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa, max_calls=2).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_pmid, max_calls=2).wrap()),
        Tool(ToolLimiter(hierarchy_hp, max_calls=4).wrap()),
        Tool(ToolLimiter(get_category_root, max_calls=2).wrap()),
        Tool(ToolLimiter(is_hpo_in_category, max_calls=24).wrap()),
        Tool(ToolLimiter(filter_hpoa_by_hp, max_calls=2).wrap()),
    ],
)

## TODO: add a Q&A feature to the agent for general questions about the annotations, the process, OMIM/HPO/MONDO, etc.
## Output QA directly as text, no JSON schema
# # test = call_agent_with_retry("Curate HPOA entries for Fabry disease (OMIM:301500). Propose <5 new phenotypes or updates based on recent literature.")
# # print(test)

# try:
#     result = call_agent_with_retry("Return the existing phenotypes in the phenotype.hpoa file for Coffin-Lowry syndrome (OMIM:303600).")
#     print(result)
# except Exception as e:
#     print("Stopped due to:", e)
