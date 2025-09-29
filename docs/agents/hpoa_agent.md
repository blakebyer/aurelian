# HPOA Agent

The HPOA agent helps biocurators explore and maintain Human Phenotype Ontology annotations. Use it when you need to:
- look up phenotypes and diseases across HPO, MONDO, and OMIM
- inspect or triage existing HPOA rows for a disorder
- draft additions or removals with supporting evidence from PubMed or PDF URLs

It supports fast single-question lookups as well as longer, tool-assisted curation sessions.

---

## Quick Start
- CLI (default standard mode):
  ```bash
  aurelian hpoa "List all HPOA rows for Marfan syndrome"
  ```
- Gradio UI:
  ```bash
  aurelian hpoa --ui
  ```
- Python API:
  ```python
  from aurelian.agents.hpoa.hpoa_agent import call_agent
  from aurelian.agents.hpoa.hpoa_config import get_config
  from aurelian.utils.async_utils import run_sync

  deps = get_config()
  result = run_sync(call_agent("What phenotypes are associated with Gaucher disease?", deps=deps))
  print(result.output)
  ```

Need to point to custom files or caches? Instantiate `HPOADependencies` directly:

```python
from pathlib import Path
from aurelian.agents.hpoa.hpoa_agent import call_agent
from aurelian.agents.hpoa.hpoa_config import HPOADependencies
from aurelian.dependencies.workdir import WorkDir
from aurelian.utils.async_utils import run_sync

custom_cache = Path.home() / ".aurelian" / "hpoa-demo"
custom_deps = HPOADependencies(
    workdir=WorkDir(),
    cache_dir=str(custom_cache),
    curator_id="jcarberry",
)
result = run_sync(call_agent("Update annotations for ORPHA:580", deps=custom_deps))
print(result.output)
```

If you set the `CURATOR_ID` environment variable (or pass `--curator-id` on the CLI), the agent records both `HPO:Agent[YYYY-MM-DD]` and `HPO:<curator_id>[YYYY-MM-DD]` provenance tags in the `biocuration` field. If no PMID, OMIM, or database_id reference is found, the `reference` field becomes `HPO:<curator_id>`. 

Both the downloaded `phenotype.hpoa` and the derived `hpoa.db` live inside the cache directory, mirroring how oaklib manages its ontology caches.

---

## Choose a Mode (`--agent`)
- `standard` (default): full curation workflow with the complete ontology + curation toolbelt (`filter_hpoa`, `batch_search_hp`, `children_of`, `parents_of`, PubMed helpers, etc.) and the structured `HPOAResponse` schema. Pick this for day-to-day annotation review.
- `reasoning`: same tools as `standard`, but the request goes through the high-effort reasoning model and the explanation ends with a **Reasoning summary** section that lists the key decision steps and tool calls.

Models are fixed per variant (standard uses `openai:gpt-5`, reasoning uses `gpt-5`); there is no CLI override (subject to change).

If you omit `--agent`, the CLI runs in `standard` mode.

### Tool Reference
- `filter_hpoa`: query the HPOA TSV/SQLite cache by disease, phenotype, PMID, and other columns.
- `batch_search_hp`: resolve multiple HPO IDs or labels to canonical terms.
- `children_of` / `parents_of`: explore narrower or broader ontology terms within HPO and MONDO.
- `categorize_hpo`: map a phenotype to high-level organ-system categories.
- `batch_search_mondo`: resolve MONDO disease IDs or labels.
- `categorize_mondo`: map diseases to higher-level MONDO groupings.
- `get_omim_terms`: fetch OMIM records linked to a disease or phenotype.
- `get_omim_clinical`: retrieve OMIM clinical synopsis text.
- `map_doi_to_pmid`: convert DOI strings to `PMID:nnnnnnnn` which is ran before `lookup_pmid_text`.
- `lookup_pmid_text`: pull PubMed abstracts/full text for evidence review.
- `pubmed_search_pmids`: search PubMed and return matching PMIDs for follow-up.
- `extract_text_from_pdf`: from a supplied PDF URL, scrape text for evidence review.

Running `python -m aurelian.agents.hpoa.hpoa_mcp` exposes these same tools over MCP with identical names.

---

## What the Response Looks Like
`standard` and `reasoning` both emit an `HPOAResponse` object:
- `explanation`: narrative summary with citations and tool output. When you run in reasoning mode, a **Reasoning summary** section is appended to the end of the explanation so the decision path is easy to audit.
- `annotations`: list of proposed or confirmed phenotypes. Each entry includes a `status` (`add`, `existing`, `edit`, or `remove`), a human-readable `rationale`, and the structured annotation payload (disease ID, HPO term, evidence, modifiers, etc.).
- Each `annotation` in `annotations` complies with the `HPOA` schema: [ℹ️ HPOA Format](https://hpo-annotation-qc.readthedocs.io/en/latest/annotationFormat.html)

Example (truncated):
```json
{
  "explanation": "Source used: Germain DP. Fabry disease. Orphanet J Rare Dis. 2010;5:30. PMID:21092187. This review outlines hallmark Fabry disease manifestations including neuropathic pain, cochleo-vestibular involvement, cardiomyopathy/arrhythmias, proteinuria with progression to kidney failure, and cerebrovascular complications (TIA, stroke). Based on explicit mentions in the article (neurological pain; cochleo-vestibular signs; cardiomyopathy and arrhythmia; proteinuria and kidney failure; transient ischemic attacks and strokes), the following PCS annotations are proposed where not already captured in HPOA for OMIM:301500. I did not propose changes to existing terms (e.g., angiokeratoma, proteinuria, arrhythmia, TIA) already present for Fabry disease.",
  "annotations": [
    {
      "status": "add",
      "rationale": "The article cites characteristic neurological pain in Fabry disease; neuropathic pain reflects the small-fiber neuropathy underlying classic pain crises.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:6000040",
        "reference": "PMID:21092187",
        "evidence": "PCS",
        "onset": null,
        "frequency": null,
        "sex": "",
        "modifier": null,
        "aspect": "P",
        "biocuration": "HPO:Agent[2025-09-28]"
      }
    },
    {
      "status": "add",
      "rationale": "Cochleo-vestibular involvement is highlighted; sensorineural hearing loss is a frequent manifestation.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0000407",
        "reference": "PMID:21092187",
        "evidence": "PCS",
        "onset": null,
        "frequency": null,
        "sex": "",
        "modifier": null,
        "aspect": "P",
        "biocuration": "HPO:Agent[2025-09-28]"
      }
    }
  ]
}
```
---
## Model Validation
Every proposed annotation goes through a chain of validators before being accepted into the structured output. These validators ensure that the agent never suggests nonsensical or redundant rows.

- Ontology checks
  - `hpo_id` must resolve to a valid HPO term.
  - `onset` (if provided) must be a descendant of HP:0003674 (Age of onset).
  - `modifier` (if provided) must be a descendant of HP:0012823 (Clinical modifier).
  - `database_id` must already exist in `phenotype.hpoa` or resolve to a valid MONDO term.

- Biocuration field
  - Any user-supplied value is ignored.
  - The agent automatically records both `HPO:<curator_id>[YYYY-MM-DD]` (if configured) and `HPO:Agent[YYYY-MM-DD]`.
  - Existing curator tags from the cache are preserved, so provenance grows over time.

- Reference field
  - Preserved if it begins with "PMID:" or "OMIM:" or matches the disease `database_id`.
  - Otherwise rewritten to `HPO:<curator_id>` (or `HPO:Agent` if no curator ID is set). __Proceed with caution__ for these annotations.

- Duplicate detection
  - Before returning, each `HPOAResult` checks the cache to ensure the annotation is not already present.
  - Duplicates are defined by `database_id`, `disease_name`, `qualifier`, `hpo_id`, `onset`, `frequency`, `sex`, and `modifier`.
  - If a match is found, the status is flipped to `existing` instead of `add`, so the curator sees the overlap clearly.

These rules guarantee that only valid ontology terms, fresh biocuration provenance, and genuinely new annotations are proposed in agent outputs.

---

## Typical CLI Tasks
- **Direct query**
  ```bash
  aurelian hpoa "Summarize phenotypes for Fabry disease"
  ```
- **Pin a custom cache**
  ```bash
  HPOA_CACHE_DIR=~/.aurelian/hpoa-demo 
  aurelian hpoa "Review Gaucher disease annotations"
  ```
- **Store history in a specific directory**
  ```bash
  aurelian hpoa --history-dir logs/hpoa --history "Summarize phenotypes for Marfan syndrome"
  ```
- **Enable retry on failure**
  ```bash
  aurelian hpoa --retry "Suggest removal of low-confidence phenotypes for ORPHA:580"
  ```
- **Toggle history caching (history is off by default)**
  ```bash
  aurelian hpoa --history "Summarize phenotypes for polycystic kidney disease"
  aurelian hpoa --no-history "What is the genetic basis for hemophilia?"
  ```
- **Pick a variant**
  ```bash
  aurelian hpoa --agent reasoning "Explain the difference between OMIM and ORPHA CHIME syndrome phenotype annotations"
  ```
- **Record curator provenance**
  ```bash
  aurelian hpoa --curator-id jcarberry "Curate DOI:10.1002/ccr3.3704 for Wiedemann-Steiner annotations"
  ```
- **Save the structured output**
  ```bash
  aurelian hpoa "List all HPOA rows for Marfan syndrome" --output results/marfan.json
  ```
  Any suffix you supply is coerced to `.json`; for example, `--output logs/marfan.txt` writes `logs/marfan.json` inside the active working directory. 

### Sample Playbook
1. Audit existing annotations (standard):
   ```bash
   aurelian hpoa --agent standard "Review Gaucher disease annotations for outdated evidence"
   ```
2. Quick ontology lookup:
   ```bash
   aurelian hpoa "Parents of HP:0009939"
   ```
3. Deep-dive curation (reasoning):
   ```bash
   aurelian hpoa --agent reasoning --retry "Cross-check ORPHA:580 phenotypes against PMID:32201668"
   ```
4. Export results:
   ```bash
   aurelian hpoa --agent standard "Compile phenotypes for Wiedemann-Steiner syndrome" --output results/wdsts.json
   ```

---

## Example Prompts
- "What phenotypes are in HPOA for MPS-IIIA?"
- "Filter HPOA by PMID:7795640 and summarize the annotations"
- "Provide the OMIM clinical synopsis for Cystic fibrosis"
- "Which body system is HP:0001297 (Stroke) assigned to?"
- "Add phenotypes from DOI:10.1002/ccr3.3704 and use the mapped PMID as reference"
- "Using https://medlineplus.gov/download/genetics/condition/costello-syndrome.pdf as context, propose adjustments to HPOA for Costello syndrome"

---

## Configuration & Environment

### Important Environment Variables
- `OPENAI_API_KEY` (required)
- `OMIM_API_KEY` (required for OMIM tools)
- `NCBI_API_KEY` (recommended to avoid PubMed rate limits)
- `AURELIAN_WORKDIR`: optional override for where agents write per-run artifacts (defaults to the directory where you launch the CLI).
- `CURATOR_ID`: optional default curator identifier recorded in annotations and reference fallbacks.  
  Recommended format: lowercase `firstinitiallastname` (e.g., `jcarberry`) or an ORCID ID  
  (e.g., `ORCID:0000-0002-1825-0097`).
- `HPOA_CACHE_DIR`: optional override for the shared cache (defaults to `~/.aurelian/hpoa`).
Use `--curator-id` on the CLI to override `CURATOR_ID` for a single run.


### How the Database Path Is Chosen
1. If `phenotype.hpoa` or `hpoa.db` exist in the current working directory or inside the configured workdir, they are copied into the shared cache (`HPOA_CACHE_DIR` or the default `~/.aurelian/hpoa`) before the agent runs.
2. A populated `hpoa.db` in the cache directory is reused immediately.
3. If the database is missing but `phenotype.hpoa` is available in the cache, it is parsed and the SQLite cache is rebuilt in place.
4. When neither file exists, the agent downloads the latest `phenotype.hpoa` release into the cache directory and generates `hpoa.db` alongside it.

### Saving History
- By default, model history (including system prompt, tool calls, and response) is not saved.
- To enable history logging, you must pass `--history` in CLI
- To change the location of history output from the default `hpoa_history` folder, you must specify both:  
`--history --history-dir <path>`
- Each session will then create a JSON file in the given directory with a name in the format `history_MM-DD-YYYY_HR-MIN-SEC.json`, such as:
  `history_09-27-2025_22-37-32.json`
---

## Gradio & MCP
- Launch the web UI: `aurelian hpoa --ui` (defaults to http://127.0.0.1:7860)
- Serve tools over MCP: `python -m aurelian.agents.hpoa.hpoa_mcp`

---

## Need Help?
Run `aurelian hpoa --help` to see every flag and option.