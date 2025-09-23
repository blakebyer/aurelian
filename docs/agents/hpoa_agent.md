# HPOA Agent

The HPOA agent assists in querying and curating Human Phenotype Ontology Annotations (HPOA). It enables researchers and curators to explore disease–phenotype associations, integrate evidence from OMIM and PubMed, and validate ontology terms in HPO and MONDO.  

The agent can operate in **Q&A mode** for fast lookups, or **Curation mode** for editing and validating phenotype annotations.

---

## Features

- Ontology lookups: Search HPO terms, MONDO disease terms, OMIM IDs, and retrieve definitions.
- Annotation queries: Filter phenotype.hpoa by disease, phenotype (HP), or reference (PMID).
- OMIM integration: Fetch OMIM disease terms and clinical synopsis sections.
- Literature integration: Search PubMed and retrieve abstracts by PMID.
- Curation support: Propose add/edit/remove actions on phenotype annotations with structured JSON.
- Categorization: Map phenotypes to organ system categories, or diseases to high-level MONDO groups.
- Persistence: Maintains local SQLite DB (`hpoa.db`) for efficient phenotype.hpoa queries.
- History: Stores recent query history and tool calls in the `hpoa_history/` folder.

---

## Examples

### Disease Phenotypes
What are the phenotypes for Fabry disease?

### OMIM Clinical Synopsis
Return the OMIM Clinical Synopsis for Cystic fibrosis

### Filtering by PMID
Compare phenotypes for MPS-III in HPOA to those discussed in PMID:32201668.

### Curation Mode
Suggest removal of phenotype annotations with poor evidence for ORPHA:580

### Ontology Lookup
What body system is HP:0009939 (mandibular aplasia)?

---
## Complete Example I: Disease Facts

### Question
Tell me what you know about MONDO:0011518 (Wiedemann-Steiner syndrome)

### Response
```json
{
  "explanation": "Wiedemann-Steiner syndrome (WDSTS) — MONDO:0011518 — is an autosomal dominant neurodevelopmental disorder caused by heterozygous loss-of-function variants in KMT2A (MLL1), a histone H3K4 methyltransferase. Most cases are de novo. Core features include distinctive facial gestalt, hypertrichosis (classically hypertrichosis cubiti, i.e., “hairy elbows”), growth impairment, hypotonia, and global developmental delay/intellectual disability; behavioral concerns (including autistic features) are common.\n\nKey clinical features (from OMIM 605130 clinical synopsis and related sources):\n- Growth: Short stature (variable); poor growth/failure to thrive in infancy.\n- Craniofacial: Thick eyebrows and synophrys; long eyelashes (some); epicanthal folds; downslanting or narrow palpebral fissures; hypertelorism; wide nasal bridge with depressed tip; long philtrum; thin upper lip; high-arched palate; low-set/dysmorphic ears; abnormal dentition.\n- Hair/skin: Hypertrichosis (notably hairy elbows), sometimes generalized or patchy; sacral dimple in some. Hairy elbows often appear in infancy and can regress during adolescence.\n- Neurologic/developmental: Global developmental delay, speech delay, intellectual disability (variable); hypotonia; wide-based/unsteady gait; seizures reported in a minority.\n- Behavior: Aggressive behavior and autistic features variably reported.\n- Extremities/skeletal: Brachydactyly/short fingers, fifth-finger clinodactyly, short middle phalanges; short toes; “fleshy” hands/feet; delayed bone age in some.\n- Other: Constipation; feeding difficulties in infancy. Strabismus and other ophthalmologic issues are reported.\n\nGenetics\n- Gene: KMT2A (OMIM:159555), also known as MLL/MLL1/TRX1. Pathogenic variants are typically truncating or other loss-of-function; haploinsufficiency is the presumed mechanism.\n- Inheritance: Autosomal dominant; most probands have de novo variants. If a parent is affected, transmission risk to offspring is 50%; consider parental mosaicism.\n\nEpidemiology and prognosis\n- Rare; cases have increased with broader genomic testing. Developmental outcomes are variable; many children attain ambulation and functional speech with support. Life span is not clearly reduced based on current reports.\n\nCancer risk\n- Although KMT2A is frequently rearranged somatically in leukemia, current germline WDSTS data do not demonstrate a clearly elevated malignancy risk; no special oncologic surveillance is established beyond routine care.\n\nEvaluation and management (supportive, individualized)\n- Developmental services (speech/OT/PT), educational supports, and behavioral therapies.\n- Monitor growth/nutrition; manage feeding difficulties and constipation; consider endocrine evaluation for poor growth.\n- Neurology for hypotonia, motor delay, and seizures if present.\n- Ophthalmology (e.g., strabismus, refractive errors) and audiology as indicated.\n- Dental and craniofacial assessments; orthopedics/rehab for hand/foot anomalies and gait issues.\n\nDifferential diagnosis\n- Kabuki syndrome (KMT2D), Cornelia de Lange spectrum, Coffin–Siris syndrome, and other epigenetic regulator disorders.\n- Distinct from Suleiman-El-Hattab syndrome (SULEHS; OMIM 618950), an autosomal recessive TASP1-related condition that may share synophrys and hirsutism but differs in genetics and broader phenotype.\n\nCross-references\n- MONDO:0011518 (Wiedemann-Steiner syndrome)\n- OMIM:605130 (Wiedemann-Steiner syndrome, clinical synopsis); OMIM:159555 (KMT2A gene)\n\nSources incorporated: MONDO record for MONDO:0011518 and OMIM clinical synopsis for WDSTS (605130).",
  "annotations": []
}
```

## Complete Example II: Proposing New Phenotype Annotations

### Question
Propose new phenotype annotations for Fabry disease based on PMID:21092187

### Response
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
      "rationale": "Classic Fabry pain is often acral (acroparesthesias); the review discusses hallmark pain symptoms consistent with distal burning/tingling in hands/feet.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0031006",
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
    },
    {
      "status": "add",
      "rationale": "Cochleo-vestibular signs include tinnitus, commonly reported in Fabry disease.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0000360",
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
      "rationale": "Vestibular involvement is noted in the review; vertigo is a recognized component of cochleo-vestibular disease in Fabry.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0002321",
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
      "rationale": "Cardiac involvement includes cardiomyopathy; adding the parent term to capture this explicitly.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0001638",
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
      "rationale": "The review notes progression to kidney failure; ESRD is a key outcome in untreated males and females.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0003774",
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
      "rationale": "Cerebrovascular complications include strokes in Fabry disease, in addition to already-annotated TIA.",
      "annotation": {
        "database_id": "OMIM:301500",
        "disease_name": "Fabry disease",
        "qualifier": "",
        "hpo_id": "HP:0001297",
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

## Technical Details

The HPOA agent integrates HPO, MONDO, and OMIM resources with PubMed literature access.

### Tools
- `search_hp`: Search HPO by ID or label, return ID/label/definition.
- `search_mondo`: Search MONDO diseases by IDs or labels.
- `batch_search_hp`: Search HPO by multiple IDs or labels.
- `batch_search_mondo`: Search MONDO by multiple IDs or labels.
- `get_omim_terms`: Find OMIM IDs for diseases.
- `get_omim_clinical`: Retrieve OMIM clinical synopsis sections.
- `filter_hpoa`: Filter phenotype.hpoa rows by any column.
- `filter_hpoa_by_disease`: Retrieve all annotations for a specific disease.
- `filter_hpoa_by_hp`: Retrieve annotations by HPO term.
- `filter_hpoa_by_pmid`: Retrieve annotations tied to a PubMed reference.
- `categorize_hpo`: Map an HPO term to top-level system categories.
- `categorize_mondo`: Map MONDO terms to disease categories.
- `children_of` / `parents_of`: Traverse ontology hierarchies.
- `pubmed_search_pmids`: Search PubMed for PMIDs by query.
- `lookup_pmid`: Retrieve PubMed abstract or full-text by PMID.

---

## Configuration

The HPOA agent is configured via environment variables:

__Required__:
- `OPENAI_API_KEY`: OpenAI API key for running GPT models.
- `OMIM_API_KEY`: OMIM API key for OMIM term and clinical synopsis queries.

__Optional__:
- `NCBI_API_KEY`: NCBI API key (recommended to avoid PubMed rate limits).
- `AURELIAN_WORKDIR`: Path to working directory (default: current directory).
- `AURELIAN_PORT`, `AURELIAN_HOST`: Host/port for Gradio web UI (default: 127.0.0.1:7860).
- `HPOA_TSV`: Explicit path to a local `phenotype.hpoa` TSV file. If set, this file will be loaded instead of downloading or relying on the current working directory.
- `HPOA_DB`: Explicit path where the SQLite database should be saved/loaded. Overrides the default `hpoa.db` in the working directory.
- `HPOA_HISTORY`: If set to `1` (default), the agent saves and reloads conversation history between calls to use as context. If set to `0`, the agent starts fresh each call. 

Setting the Input TSV Path
--------------------------
The agent can be pointed directly to a local `phenotype.hpoa` file by setting `HPOA_TSV`.

Example:
```bash
export HPOA_TSV="$HOME/data/phenotype.hpoa"
```
This bypasses the need to place the file in the current working directory.

Setting the Database Path
-------------------------
The agent requires `phenotype.hpoa` to populate its annotation database (`hpoa.db`). 

Resolution order:

1. If you set `hpoa_db_path` explicitly in Python, that path is used.
   Example:
   ```python
   from aurelian.agents.hpoa.hpoa_config import HPOADependencies
   deps = HPOADependencies(hpoa_db_path="/custom/path/hpoa.db")
   ```
2. Otherwise, if `HPOA_DB` is set, that path is used.
   Example:
   ```bash
   export HPOA_DB="$HOME/custom/hpoa.db"
   ```

3. Otherwise, if `phenotype.hpoa` exists in the current working directory, it is loaded and used to build `hpoa.db`.

4. If not found, the agent will download the latest `phenotype.hpoa` release from GitHub and save it locally.

5. By default, the SQLite database is written as `hpoa.db` in the working directory (or in `AURELIAN_WORKDIR` if set).

### Query History
Given `HPOA_HISTORY` is set to `1` (default), every time you query the agent, the conversation (including tool calls and responses) is stored in the `hpoa_history/` folder. Each session is saved as a JSON file named with the current date and time.  

This allows you to review, replay, or parse previous interactions.

---

## Using the Agent

### Run a query using Aurelian's Python API
```python
from aurelian.agents.hpoa.hpoa_agent import call_agent
from aurelian.agents.hpoa.hpoa_config import get_config
from aurelian.utils.async_utils import run_sync

# configure dependencies
deps = get_config()
result = run_sync(call_agent("What phenotypes are associated with Gaucher disease?", deps=deps))
print(result.output)
```

## Command Line

### Direct query
```bash
aurelian hpoa "List phenotypes for Marfan syndrome"
```

### Web Interface (Gradio)
```bash
aurelian hpoa --ui
```

This launches a browser interface at http://127.0.0.1:7860 by default.

### MCP Server
Run the MCP server for tool integration:
```bash
python -m aurelian.agents.hpoa.hpoa_mcp
```
This exposes the agent’s tools over MCP.

---
## Getting Help
Run:
```bash
aurelian hpoa --help
```

This prints available options and usage.