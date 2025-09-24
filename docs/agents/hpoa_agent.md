# HPOA Agent

The HPOA agent helps biocurators explore and maintain Human Phenotype Ontology annotations. Use it when you need to:
- look up phenotypes and diseases across HPO, MONDO, and OMIM
- inspect or triage existing HPOA rows for a disorder
- draft additions or removals with supporting evidence from PubMed

It supports fast single-question lookups as well as longer, tool-assisted curation sessions.

---

## Quick Start
- CLI (default standard mode):
  ```bash
  aurelian hpoa "List phenotypes for Marfan syndrome"
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
from aurelian.agents.hpoa.hpoa_agent import call_agent
from aurelian.agents.hpoa.hpoa_config import HPOADependencies
from aurelian.utils.async_utils import run_sync

custom_deps = HPOADependencies(
    hpoa_db_path="data/hpoa.db",
    hpoa_tsv="data/phenotype.hpoa",
)
result = run_sync(call_agent("Update annotations for ORPHA:580", deps=custom_deps))
print(result.output)
```

---

## Choose a Mode (`--agent`)
- `standard` (default): full curation workflow. Uses the entire ontology + curation toolbelt (`filter_hpoa`, `batch_search_hp`, PubMed helpers, etc.) and returns the structured `HPOAResponse` schema. Pick this for day-to-day annotation review.
- `reasoning`: same tools as `standard`, but the request goes through the high-effort reasoning model and the explanation ends with a **Reasoning summary** section that lists the key decision steps and tool calls. The CLI strips any `openai:` prefix automatically, so `--model openai:gpt-5` and `--model gpt-5` behave the same.
- `simple`: lightweight ontology concierge. Only ontology navigation tools are available (batch searches, parent/child lookups), no curation helpers, and responses are usually plain text.

If you omit `--agent`, the CLI runs in `standard` mode.

### Tool Reference
- **Shared by `standard` and `reasoning`**:
  - `filter_hpoa`: query the HPOA TSV/SQLite cache by disease, phenotype, PMID, and other columns.
  - `batch_search_hp`: resolve multiple HPO IDs or labels to canonical terms.
  - `categorize_hpo`: map a phenotype to high-level organ-system categories.
  - `batch_search_mondo`: resolve MONDO disease IDs or labels.
  - `categorize_mondo`: map diseases to higher-level MONDO groupings.
  - `get_omim_terms`: fetch OMIM records linked to a disease or phenotype.
  - `get_omim_clinical`: retrieve OMIM clinical synopsis text.
  - `lookup_pmid_text`: pull PubMed abstracts/full text for evidence review.
  - `pubmed_search_pmids`: search PubMed and return matching PMIDs for follow-up.
- **Only in `simple`**:
  - `batch_search_hp`, `batch_search_mondo`, `categorize_hpo`, `categorize_mondo`, `children_of`, `parents_of` (no curation helpers).

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

`simple` returns free text (no `annotations`). If you need structured output, stick with `standard` or `reasoning`.

---

## Typical CLI Tasks
- **Direct query**
  ```bash
  aurelian hpoa "Summarize phenotypes for Fabry disease"
  ```
- **Switch models**
  ```bash
  aurelian hpoa --model openai:gpt-4o "List HPOA rows for Marfan syndrome"
  ```
- **Control retries**
  ```bash
  aurelian hpoa --no-retry "Suggest removal of low-confidence phenotypes for ORPHA:580"
  ```
- **Toggle history caching**
  ```bash
  aurelian hpoa --history "Summarize phenotypes for polycystic kidney disease"
  aurelian hpoa --no-history "What is the genetic basis for hemophilia?"
  ```
- **Pick a variant**
  ```bash
  aurelian hpoa --agent simple "What organ system does HP:0001297 belong to?"
  aurelian hpoa --agent reasoning --model gpt-5 "Explain the difference between OMIM and ORPHA CHIME syndrome phenotype annotations"
  ```
- **Save the structured output**
  ```bash
  aurelian hpoa "List phenotypes for Marfan syndrome" --output results/marfan.json
  ```

### Sample Playbook
1. Audit existing annotations (standard):
   ```bash
   aurelian hpoa --agent standard --model gpt-5 "Review Gaucher disease annotations for outdated evidence"
   ```
2. Quick ontology lookup (simple):
   ```bash
   aurelian hpoa --agent simple "Parents of HP:0009939"
   ```
3. Deep-dive curation (reasoning):
   ```bash
   aurelian hpoa --agent reasoning --model gpt-4o --no-retry "Cross-check ORPHA:580 phenotypes against PMID:32201668"
   ```
4. Export results:
   ```bash
   aurelian hpoa --agent standard "Compile phenotypes for Wiedemann-Steiner syndrome" --output results/wdsts.json
   ```

---

## Example Prompts
- "What phenotypes are in HPOA for MPS-III?"
- "Filter HPOA by PMID:7795640 and summarize the annotations"
- "Suggest removal of low-evidence phenotypes for ORPHA:580"
- "Provide the OMIM clinical synopsis for Cystic fibrosis"
- "Which body system is HP:0001297 (Stroke) assigned to?"

---

## Configuration & Environment

### Important Environment Variables
- `OPENAI_API_KEY` (required)
- `OMIM_API_KEY` (required for OMIM tools)
- `NCBI_API_KEY` (recommended to avoid PubMed rate limits)
- `HPOA_TSV`: optional explicit path to `phenotype.hpoa`
- `HPOA_DB`: optional location for the SQLite cache (`hpoa.db`)
- `HPOA_HISTORY`: `1` (default) saves session transcripts to `hpoa_history/`; `0` disables persistence. If set, the agent uses the previous three messages as context. The CLI `--history/--no-history` flags override this per run.

### How the Database Path Is Chosen
1. If you instantiate `HPOADependencies(hpoa_db_path=...)`, that path wins.
2. Else, `HPOA_DB` is used when set.
3. Else, a local `phenotype.hpoa` file seeds the database.
4. Else, the latest release is downloaded.
5. Default location: `hpoa.db` in the current workdir (or `AURELIAN_WORKDIR`).

---

## Gradio & MCP
- Launch the web UI: `aurelian hpoa --ui` (defaults to http://127.0.0.1:7860)
- Serve tools over MCP: `python -m aurelian.agents.hpoa.hpoa_mcp`

---

## Need Help?
Run `aurelian hpoa --help` to see every flag and option.
