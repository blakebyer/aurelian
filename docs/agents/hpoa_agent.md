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
Which phenotypes for MPS III are cited in PMID:32201668?

### Curation Mode
Suggest removal of phenotype annotations with poor evidence for ORPHA:580

### Ontology Lookup
What body system is HP:0009939 (mandibular aplasia)?

---

## Technical Details

The HPOA agent integrates HPO, MONDO, and OMIM resources with PubMed literature access.

### Tools
- `search_hp`: Search HPO by ID or label, return ID/label/definition.
- `search_mondo`: Search MONDO disease IDs or labels.
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
- `lookup_pmid`: Retrieve PubMed abstracts by PMID.

---

## Configuration

The HPOA agent is configured via environment variables:

Required:
- `OPENAI_API_KEY`: OpenAI API key for running GPT models.
- `OMIM_API_KEY`: OMIM API key for OMIM term and clinical synopsis queries.

Optional:
- `NCBI_API_KEY`: NCBI API key (recommended to avoid PubMed rate limits).
- `AURELIAN_WORKDIR`: Path to working directory (default: current directory).
- `AURELIAN_PORT`, `AURELIAN_HOST`: Host/port for Gradio web UI (default: 127.0.0.1:7860).

### Setting the Database Path
The agent requires `phenotype.hpoa` to populate its annotation database (`hpoa.db`). It follows this order:

1. If you set `hpoa_db_path` explicitly in Python, that path is used.  
   Example:
   ```python
   from aurelian.agents.hpoa.hpoa_config import HPOADependencies
   deps = HPOADependencies(hpoa_db_path="/custom/path/phenotype.hpoa.db")
   ```
2. Otherwise, if `phenotype.hpoa` exists in the current working directory, it is loaded and used to build `hpoa.db`.

3. If not found, the agent will download the latest `phenotype.hpoa` release from GitHub and save it locally.

4. By default, the SQLite database is written as `hpoa.db` in the working directory (or `AURELIAN_WORKDIR` if set).

### Query History
Every time you query the agent, the conversation (including tool calls and responses) is stored in the `hpoa_history/` folder.  
Each session is saved as a JSON file named with the current date and time.  
This allows you to review, replay, or parse previous interactions.

---

## Using the Agent

### Run a query using the Python API
```python
from aurelian.agents.hpoa.hpoa_agent import call_agent
from aurelian.agents.hpoa.hpoa_config import get_config

# configure dependencies
deps = get_config()
result = call_agent("What phenotypes are associated with Gaucher disease?", deps=deps)
print(result.output)
```

## Command Line

### Direct query
```
aurelian hpoa "List phenotypes for Marfan syndrome"
```

### Web Interface (Gradio)
```
aurelian hpoa --ui
```

This launches a browser interface at http://127.0.0.1:7860 by default.

### MCP Server
Run the MCP server for tool integration:
```
python -m aurelian.agents.hpoa.hpoa_mcp
```
This exposes the agent’s tools over MCP.

---

## Getting Help
Run:
```
aurelian hpoa --help
```

This prints available options and usage.