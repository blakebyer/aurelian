"""
Tools for interacting with MONDO, HPO, and HPOA files.
"""
from typing import List, Literal, Dict
from typing_extensions import TypedDict
import asyncio
import httpx
import re, sqlite3, inspect
from pydantic_ai import RunContext, ModelRetry
from aurelian.utils.pdf_fetcher import extract_text_from_pdf_url
from aurelian.utils.pubmed_utils import doi_to_pmid
from aurelian.agents.hpoa.hpoa_config import HPOADependencies, HPOA, get_config, get_client
from aurelian.agents.literature.literature_tools import (
    lookup_pmid,
    )
from oaklib.datamodels.search import SearchConfiguration

async def search_hp(ctx: RunContext[HPOADependencies], term: str) -> List[dict]:
    """Look up HPO phenotypic abnormality terms by CURIE or partial label.

    Args:
        ctx: Execution context providing ontology adapters.
        term: Search string; accepts HP CURIEs or free-text labels.

    Returns:
        List[dict]: Resolved HP term dictionaries with `id`, `label`, and `definition` keys.
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()

    q = (term or "").strip()
    if not q:
        return []

    # Direct ID lookup
    if q.lower().startswith("hp:"):
        curie = q.upper()
        try:
            return [{
                "id": curie,
                "label": hp.label(curie),
                "definition": hp.definition(curie),
            }]
        except Exception:
            return [{"id": curie, "label": None, "definition": None}]

    # Label search
    try:
        bs = hp.basic_search(q, SearchConfiguration(is_partial=True))
        if inspect.iscoroutine(bs):
            bs = await bs
        found = list(bs)
    except Exception:
        found = []

    results: List[dict] = []
    for curie in found:
        if not isinstance(curie, str) or not curie.startswith("HP:"):
            continue
        try:
            results.append({
                "id": curie,
                "label": hp.label(curie),
                "definition": hp.definition(curie),
            })
        except Exception:
            results.append({"id": curie, "label": None, "definition": None})
    return results

async def search_mondo(ctx: RunContext[HPOADependencies], term: str) -> List[dict]:
    """Look up MONDO disease terms by CURIE or partial label.

    Args:
        ctx: Execution context providing ontology adapters.
        term: Search string; accepts MONDO CURIEs or free-text labels.

    Returns:
        List[dict]: Resolved MONDO term dictionaries with `id`, `label`, and `definition` keys.
    """
    config = ctx.deps or get_config()
    mondo = config.get_mondo_adapter()

    q = (term or "").strip()
    if not q:
        return []

    # Direct ID lookup
    if q.lower().startswith("mondo:"):
        curie = q.upper()
        try:
            return [{
                "id": curie,
                "label": mondo.label(curie),
                "definition": mondo.definition(curie),
            }]
        except Exception:
            return [{"id": curie, "label": None, "definition": None}]

    # Label search
    try:
        bs = mondo.basic_search(q, SearchConfiguration(is_partial=True))
        if inspect.iscoroutine(bs):
            bs = await bs
        found = list(bs)
    except Exception:
        found = []

    results: List[dict] = []
    for curie in found:
        if not isinstance(curie, str) or not curie.startswith("MONDO:"):
            continue
        try:
            results.append({
                "id": curie,
                "label": mondo.label(curie),
                "definition": mondo.definition(curie),
            })
        except Exception:
            results.append({"id": curie, "label": None, "definition": None})
    return results

async def batch_search_hp(ctx: RunContext[HPOADependencies], terms: List[str]) -> List[Dict]:
    """Look up multiple HPO phenotypic abnormality terms by CURIE or partial label.

    Args:
        ctx: Execution context providing ontology adapters.
        terms: List of search strings; each may be an HP CURIE or free-text label.

    Returns:
        List[dict]: Resolved HP term dictionaries with `id`, `label`, and `definition` keys.
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    results: List[Dict] = []
    ids: List[str] = []

    for t in terms:
        q = (t or "").strip()
        if not q:
            continue

        if q.lower().startswith("hp:"):
            ids.append(q.upper())
            continue

        try:
            bs = hp.multiterm_search([q], SearchConfiguration(is_partial=True))
            found = await bs if inspect.iscoroutine(bs) else list(bs)
            ids.extend([c for c in found if isinstance(c, str) and c.startswith("HP:")])
        except Exception:
            pass

    for curie in set(ids):
        try:
            results.append({
                "id": curie,
                "label": hp.label(curie),
                "definition": hp.definition(curie),
            })
        except Exception:
            results.append({"id": curie, "label": None, "definition": None})

    return results


async def batch_search_mondo(ctx: RunContext[HPOADependencies], terms: List[str]) -> List[Dict]:
    """Look up multiple MONDO disease terms by CURIE or partial label.

    Args:
        ctx: Execution context providing ontology adapters.
        terms: List of search strings; each may be a MONDO CURIE or free-text label.

    Returns:
        List[dict]: Resolved MONDO term dictionaries with `id`, `label`, and `definition` keys.
    """
    config = ctx.deps or get_config()
    mondo = config.get_mondo_adapter()
    results: List[Dict] = []
    ids: List[str] = []

    for t in terms:
        q = (t or "").strip()
        if not q:
            continue

        if q.lower().startswith("mondo:"):
            ids.append(q.upper())
            continue

        try:
            bs = mondo.multiterm_search([q], SearchConfiguration(is_partial=True))
            found = await bs if inspect.iscoroutine(bs) else list(bs)
            ids.extend([c for c in found if isinstance(c, str) and c.startswith("MONDO:")])
        except Exception:
            pass

    for curie in set(ids):
        try:
            results.append({
                "id": curie,
                "label": mondo.label(curie),
                "definition": mondo.definition(curie),
            })
        except Exception:
            results.append({"id": curie, "label": None, "definition": None})

    return results

async def get_omim_terms(ctx: RunContext[HPOADependencies], label: str):
    """Retrieve OMIM entry metadata for diseases that match the label.

    Args:
        ctx: Execution context providing configuration and API keys.
        label: Free-text search term sent to the OMIM entry search endpoint.

    Returns:
        dict: Parsed JSON payload describing candidate OMIM entries.

    Raises:
        ModelRetry: If the HTTP request fails or the payload cannot be decoded.
    """
    config = ctx.deps or get_config()
    OMIM_API_KEY = config.omim_api_key

    url = "https://api.omim.org/api/entry/search"
    params = {
        "search": label,
        "format": "json",
        "apiKey": OMIM_API_KEY or "",
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "aurelian-hpoa/1.0",
    }
    client = get_client()
    r = await client.get(url, params=params, headers=headers)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ModelRetry(f"OMIM search failed: {e.response.status_code} {e.response.text[:200]}")
    try:
        return r.json()
    except ValueError:
            raise ModelRetry("OMIM search returned non-JSON response")

async def get_omim_clinical(ctx: RunContext[HPOADependencies], label: str):
    """Fetch OMIM clinical synopsis data for diseases that match the label.

    Args:
        ctx: Execution context providing configuration and API keys.
        label: Free-text disease term sent to the OMIM entry search endpoint.

    Returns:
        dict: Parsed JSON payload containing clinical synopsis sections.

    Raises:
        ModelRetry: If the HTTP request fails or the response cannot be decoded.
    """
    config = ctx.deps or get_config()
    OMIM_API_KEY = config.omim_api_key

    url = "https://api.omim.org/api/entry/search"
    params = {
        "search": label,
        "format": "json",
        "include": "clinicalSynopsis",
        "apiKey": OMIM_API_KEY or "",
        "limit": "5", # limit results
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "aurelian-hpoa/1.0",
    }
    client = get_client()
    r = await client.get(url, params=params, headers=headers)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ModelRetry(f"OMIM clinical search failed: {e.response.status_code} {e.response.text[:200]}")
    try:
        return r.json()
    except ValueError:
            raise ModelRetry("OMIM clinical search returned non-JSON response")
    
CURIE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]+:\d+$")

class FilterSpec(TypedDict, total=False):
    field: Literal[
        "database_id",
        "disease_name",
        "qualifier",
        "hpo_id",
        "reference",
        "evidence",
        "onset",
        "frequency",
        "sex",
        "modifier",
        "aspect",
        "biocuration",
    ]
    query: str
    mode: Literal["exact", "like"]

async def filter_hpoa(
    ctx: RunContext[HPOADependencies],
    filters: List[FilterSpec],
) -> List[HPOA]:
    """Filter the local phenotype.hpoa database using structured predicates.

    Args:
        ctx: Execution context providing the local SQLite database.
        filters: Sequence of filter specifications describing field, query, and match mode.

    Returns:
        List[HPOA]: Rows that satisfy every provided filter.
    """
    config = ctx.deps or get_config()
    await config.ensure_hpoa_db()

    con = sqlite3.connect(config.hpoa_db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()

        clauses = []
        params = []

        # Build only from allowed fields
        for spec in filters:
            field = spec["field"]
            query = spec["query"].strip()
            mode = spec.get("mode")

            # Auto mode if not specified
            if mode is None:
                if CURIE_PATTERN.match(query):
                    mode = "exact"
                else:
                    mode = "like"

            if mode == "exact":
                clauses.append(f"UPPER({field}) = ?")
                params.append(query.upper())
            elif mode == "like":
                clauses.append(f"{field} LIKE ? COLLATE NOCASE")
                params.append(f"%{query}%")
            else:
                raise ValueError(f"Invalid mode {mode} for field {field}")

        if not clauses:
            return []  # no filters → nothing

        sql = f"SELECT * FROM hpoa WHERE {' AND '.join(clauses)}"
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    # Strictly cast rows to HPOA model, invalid rows skipped
    results: List[HPOA] = []
    for row in rows:
        try:
            results.append(HPOA(**row))
        except Exception as e:
            print(f"Skipping invalid row: {e}")
    return results

async def filter_hpoa_by_disease(ctx: RunContext[HPOADependencies], label: str) -> List[HPOA]:
    """
    Return all phenotype.hpoa rows for a disease.

    Matching strategy:
    - If input contains an OMIM/ORPHA/MONDO/DECIPHER CURIE, match against database_id.
    - Otherwise, perform case-insensitive substring match against disease_name.

    Args:
        ctx: RunContext with HPOADependencies loaded
        label: e.g., "Fabry" or "OMIM:301500"

    Returns:
        List of HPOA rows (as objects).
    """
    config = ctx.deps or get_config()
    await config.ensure_hpoa_db()

    q_raw = label.strip()
    # Detect CURIE-style IDs
    id_pattern = r"(OMIM|ORPHA|MONDO|DECIPHER):[A-Z0-9_.-]+"
    id_search = re.search(id_pattern, q_raw.upper().replace(" ", ""))
    q_id = id_search.group(0) if id_search else None

    con = sqlite3.connect(config.hpoa_db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        if q_id:
            # Fast normalized equality on database_id (OMIM/MONDO/ORPHA/DECIPHER)
            cur.execute("SELECT * FROM hpoa WHERE UPPER(REPLACE(database_id,' ','')) = ?", (q_id,))
            rows = [dict(r) for r in cur.fetchall()]
        else:
            # Case-insensitive label search using LIKE; callers pass compact labels
            cur.execute("SELECT * FROM hpoa WHERE disease_name LIKE ? COLLATE NOCASE", (f"%{q_raw}%",))
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    results: List[HPOA] = []
    for row in rows:
        try:
            results.append(HPOA(**row))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return results

async def filter_hpoa_by_pmid(ctx: RunContext[HPOADependencies], pmid: str) -> List[HPOA]:
    """Return phenotype.hpoa rows that cite the provided PMID.

    Args:
        ctx: Execution context with access to the local HPOA database.
        pmid: PubMed identifier supplied as "PMID:n" or bare digits.

    Returns:
        List[HPOA]: Rows whose reference field contains the PMID.
    """
    config = ctx.deps or get_config()
    await config.ensure_hpoa_db()
    pid = pmid.strip().replace("PMID:", "").strip()

    con = sqlite3.connect(config.hpoa_db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM hpoa WHERE UPPER(reference) LIKE ?", (f"%PMID:{pid}%",))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    results: List[HPOA] = []
    for row in rows:
        try:
            results.append(HPOA(**row))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return results

async def lookup_pmid_text(pmid: str) -> str:
    """Return the abstract or full text for a PubMed article.

    Args:
        pmid: PubMed identifier supplied as "PMID:n" or bare digits.

    Returns:
        str: Text blob provided by the backing literature tool.
    """
    return await lookup_pmid(pmid)

async def map_doi_to_pmid(doi: str) -> str:
    """Return a PMID from DOI to be used in the reference field of HPOA.

    Args: 
        doi: Digital Object Identifier supplied as "10.1126/science.aar3646".

    Returns:
        str: PMID as provided by backing pubmed utility tool.
    """
    print(f"CONVERT DOI: {doi}")
    try:
        return doi_to_pmid(doi)
    except:
        raise ModelRetry(f"Error converting DOI {doi} to PMID.")

async def extract_text_from_pdf(pdf_url: str) -> str:
    """
    Extract text from a PDF at the given URL.

    Args:
        ctx: Execution context providing shared HTTP client reuse.
        pdf_url: URL to the PDF file

    Returns:
        str: The extracted text content
    """
    print(f"EXTRACT PDF: {pdf_url}")
    try:
        return extract_text_from_pdf_url(pdf_url)
    except Exception as exc:
        raise ModelRetry(f"Error retrieving PDF from URL: {pdf_url}: {exc}")

async def filter_hpoa_by_hp(ctx: RunContext[HPOADependencies], hp: str) -> List[HPOA]:
    """Return phenotype.hpoa rows tied to the requested HPO identifier.

    Args:
        ctx: Execution context with access to ontology adapters and database.
        hp: HPO CURIE or label; labels are resolved to an ID before filtering.

    Returns:
        List[HPOA]: Rows whose `hpo_id` matches the resolved identifier.
    """
    config = ctx.deps or get_config()
    await config.ensure_hpoa_db()
    raw = hp.strip()
    # Resolve label to HP:ID if needed
    if not raw.upper().startswith("HP:"):
        try:
            matches = await search_hp(ctx, raw)
            if not matches:
                return []
            hp_norm = (matches[0].get("id") or "").upper()
        except Exception:
            return []
    else:
        hp_norm = raw.upper()

    con = sqlite3.connect(config.hpoa_db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        # Fast normalized equality on HPO IDs
        cur.execute("SELECT * FROM hpoa WHERE UPPER(hpo_id) = ?", (hp_norm.upper(),))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    results: List[HPOA] = []
    for row in rows:
        try:
            results.append(HPOA(**row))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return results

async def pubmed_search_pmids(ctx: RunContext[HPOADependencies], query: str, retmax: int = 20) -> list:
    """
    Search PubMed (via NCBI ESearch API) for PMIDs matching a text query.
    
    Args:
        query (str): Search query for PubMed.
        retmax (int): Maximum number of PMIDs to return. Default = 20.

    Returns:
        list: List of PMIDs in the format ["PMID:123456", ...].
    """
    config = ctx.deps or get_config()
    NCBI_API_KEY = config.ncbi_api_key

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
    }
    if NCBI_API_KEY:  # only include if non-empty
        params["api_key"] = NCBI_API_KEY

    headers = {"Accept": "application/json"}

    print(f"SEARCH PUBMED FOR PMIDs RELATED TO: {query}")

    client = get_client()
    r = await client.get(url, params=params, headers=headers)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ModelRetry(
            f"PubMed search failed: {e.response.status_code} {e.response.text[:200]}"
        )
    try:
        data = r.json()
    except ValueError:
        raise ModelRetry("PubMed search returned non-JSON response")

    pmids = data.get("esearchresult", {}).get("idlist", [])
    return [f"PMID:{p}" for p in pmids]

# Helper functions for dealing with ontology hierarchies
async def children_of(ctx: RunContext[HPOADependencies], parent: str) -> List[str]:
    """Return the direct child terms for the supplied ontology identifier.

    Args:
        ctx: Execution context with ontology adapters.
        parent: Ontology identifier (HP or MONDO) whose children should be listed.

    Returns:
        List[str]: Identifiers of immediate child terms.
    """
    config = ctx.deps or get_config()
    if "hp" in parent.lower():
        onto = config.get_hp_adapter()
    elif "mondo" in parent.lower():
        onto = config.get_mondo_adapter()
    else: 
        raise ValueError(f"Parent not an HP or MONDO ID: {parent}")
    try:
        return [s for s, p, o in onto.relationships(objects=[parent])]
    except Exception:
        return []

async def parents_of(ctx: RunContext[HPOADependencies], child: str) -> List[str]:
    """Return the direct parent terms for the supplied ontology identifier.

    Args:
        ctx: Execution context with ontology adapters.
        child: Ontology identifier (HP or MONDO) whose parents should be listed.

    Returns:
        List[str]: Identifiers of immediate parent terms.
    """
    config = ctx.deps or get_config()
    if "hp" in child.lower():
        onto = config.get_hp_adapter()
    elif "mondo" in child.lower():
        onto = config.get_mondo_adapter()
    else: 
        raise ValueError(f"Child not an HP or MONDO ID: {child}")
    try:
        return [o for s, p, o in onto.relationships(subjects=[child])]
    except Exception:
        return []

async def categorize_hpo(ctx: RunContext[HPOADependencies], term: str) -> List[str]:
    """Map an HPO term to the top-level organ system categories.

    Args:
        ctx: Execution context with ontology adapters.
        term: HPO identifier to categorize.

    Returns:
        List[str]: Category identifiers rendered as "ID (Label)" strings.
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    HP_SYSTEM_ROOT = "HP:0000118"  # Phenotypic abnormality
    systems = await children_of(ctx, HP_SYSTEM_ROOT)
    try:
        ancestors = set(hp.ancestors(term, reflexive=True) or [])
    except Exception:
        ancestors = set()
    return [f"{s} ({hp.label(s)})" for s in systems if s in ancestors]

async def categorize_mondo(ctx: RunContext[HPOADependencies], term: str) -> List[str]:
    """Map a MONDO term to high-level disease group categories.

    Args:
        ctx: Execution context with ontology adapters.
        term: MONDO identifier to categorize.

    Returns:
        List[str]: Category identifiers rendered as "ID (Label)" strings.
    """
    config = ctx.deps or get_config()
    mondo = config.get_mondo_adapter()
    MONDO_SYSTEM_ROOT = "MONDO:0700096"  # disease
    try:
        systems = [s for s, p, o in mondo.relationships(objects=[MONDO_SYSTEM_ROOT])]
    except Exception:
        systems = []
    try:
        ancestors = set(mondo.ancestors(term, reflexive=True) or [])
    except Exception:
        ancestors = set()
    return [f"{s} ({mondo.label(s)})" for s in systems if s in ancestors]