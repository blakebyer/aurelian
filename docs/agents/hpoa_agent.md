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
)
result = run_sync(call_agent("Update annotations for ORPHA:580", deps=custom_deps))
print(result.output)
```

Both the downloaded `phenotype.hpoa` and the derived `hpoa.db` live inside the cache directory, mirroring how oaklib manages its ontology caches.

---

## Choose a Mode (`--agent`)
- `standard` (default): full curation workflow with the complete ontology + curation toolbelt (`filter_hpoa`, `batch_search_hp`, `children_of`, `parents_of`, PubMed helpers, etc.) and the structured `HPOAResponse` schema. Pick this for day-to-day annotation review.
- `reasoning`: same tools as `standard`, but the request goes through the high-effort reasoning model and the explanation ends with a **Reasoning summary** section that lists the key decision steps and tool calls.

Models are fixed per variant (standard uses `openai:gpt-5`, reasoning uses `gpt-5`); there is no CLI override.

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
- `lookup_pmid_text`: pull PubMed abstracts/full text for evidence review.
- `pubmed_search_pmids`: search PubMed and return matching PMIDs for follow-up.
- `extract_text_from_pdf`: from a supplied PDF URL, scrape text for evidence review.

Running `python -m aurelian.agents.hpoa.hpoa_mcp` exposes these same tools over MCP with identical names.

---

## What the Response Looks Like
`standard` and `reasoning` both emit an `HPOAResponse` object:
- `explanation`: narrative summary with citations and tool output. When you run in reasoning mode, a **Reasoning summary** section is appended to the end of the explanation so the decision path is easy to audit.
- `annotations`: list of proposed or confirmed phenotypes. Each entry includes a `status` (`add`, `existing`, `edit`, or `remove`), a human-readable `rationale`, and the structured annotation payload (disease ID, HPO term, evidence, modifiers, etc.).

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
        "aspect": "P"
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
        "aspect": "P"
      }
    }
  ]
}
```


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
- **Save the structured output**
  ```bash
  aurelian hpoa "List all HPOA rows for Marfan syndrome" --output results/marfan.json
  ```
  Whatever filename you pass is saved with a `.json` extension containing JSON-formatted text.

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
- "Suggest removal of low-evidence phenotypes for ORPHA:580"
- "Provide the OMIM clinical synopsis for Cystic fibrosis"
- "Which body system is HP:0001297 (Stroke) assigned to?"
- "Using https://medlineplus.gov/download/genetics/condition/costello-syndrome.pdf, propose adjustments to HPOA for Costello syndrome."

---

## Configuration & Environment

### Important Environment Variables
- `OPENAI_API_KEY` (required)
- `OMIM_API_KEY` (required for OMIM tools)
- `NCBI_API_KEY` (recommended to avoid PubMed rate limits)
- `AURELIAN_WORKDIR`: optional override for where agents write per-run artifacts (defaults to the directory where you launch the CLI).
- `HPOA_CACHE_DIR`: optional override for the shared cache (defaults to `~/.aurelian/hpoa`).

### How the Database Path Is Chosen
1. If `phenotype.hpoa` or `hpoa.db` exist in the current working directory or inside the configured workdir, they are copied into the shared cache (`HPOA_CACHE_DIR` or the default `~/.aurelian/hpoa`) before the agent runs.
2. A populated `hpoa.db` in the cache directory is reused immediately.
3. If the database is missing but `phenotype.hpoa` is available in the cache, it is parsed and the SQLite cache is rebuilt in place.
4. When neither file exists, the agent downloads the latest `phenotype.hpoa` release into the cache directory and generates `hpoa.db` alongside it.

### Saving History
- By default, model history (including system prompt, tool calls, and response) is not saved.
- To enable history logging, you must pass both in CLI:
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