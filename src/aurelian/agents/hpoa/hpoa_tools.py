"""
Tools for interacting with MONDO, HPO, and HPOA files.
"""
import asyncio
from typing import Dict, List, Any, Optional
import requests
import httpx
import re, os, sqlite3
from pydantic_ai import RunContext, ModelRetry
from .hpoa_config import HPOADependencies, HPOA, get_config, get_client
from aurelian.agents.literature.literature_tools import (
    lookup_pmid as literature_lookup_pmid,
    literature_search_pmids as literature_search_pmids,
    )
from oaklib.datamodels.search import SearchConfiguration

# Simple in-memory caches to speed up oaklib calls within a run
_HP_LABEL_CACHE: Dict[str, Optional[str]] = {}
_HP_ANCESTORS_CACHE: Dict[str, List[str]] = {}
_HP_OUTGOING_CACHE: Dict[str, List[tuple]] = {}
_HP_CHILDREN_CACHE: Dict[str, List[str]] = {}
_CATEGORY_LABEL_TO_ID: Dict[str, str] = {}

CATEGORY_ROOT = "HP:0000118"  # Phenotypic abnormality

async def search_hp(ctx: RunContext[HPOADependencies], label: str) -> List[dict]:
    """Search the HPO for phenotypic abnormalities, qualifiers, or frequencies."""

    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    results = list(hp.basic_search(label, SearchConfiguration(is_partial=True)))
    data = []
    for curie in results:
        if not curie.startswith("HP:"):
            continue
        data.append({
                "id": curie,
                "label": hp.label(curie),
                "definition": hp.definition(curie),
            })
    return data

async def search_mondo(ctx: RunContext[HPOADependencies], label: str) -> List[dict]:
    """Search the MONDO Ontology for disease identifiers."""

    config = ctx.deps or get_config()
    mondo = config.get_mondo_adapter()

    HUMAN_DISEASE_ROOT = "MONDO:0700096"

    def is_human_disease(curie: str) -> bool:
        ancestors = set(mondo.ancestors(curie))
        return HUMAN_DISEASE_ROOT in ancestors

    results = list(mondo.basic_search(label, SearchConfiguration(is_partial=True)))
    data = []
    for curie in results:
        if not is_human_disease(curie):
            continue
        data.append({
            "id" : curie,
            "label" : mondo.label(curie),
            "definition": mondo.definition(curie),
        })
    return data

async def get_omim_terms(ctx: RunContext[HPOADependencies], label: str):
    """Search the OMIM DB for disease identifiers (async httpx)."""
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
    client = await get_client()
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
    """Search the OMIM DB for clinical synopses (async httpx).

    This can include other clinical DB terms e.g. SNOMED, HP, ICD10CM in the synopsis.
    """
    config = ctx.deps or get_config()
    OMIM_API_KEY = config.omim_api_key

    url = "https://api.omim.org/api/entry/search"
    params = {
        "search": label,
        "format": "json",
        "include": "clinicalSynopsis",
        "apiKey": OMIM_API_KEY or "",
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "aurelian-hpoa/1.0",
    }
    client = await get_client()
    r = await client.get(url, params=params, headers=headers)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ModelRetry(f"OMIM clinical search failed: {e.response.status_code} {e.response.text[:200]}")
    try:
        return r.json()
    except ValueError:
            raise ModelRetry("OMIM clinical search returned non-JSON response")

async def filter_hpoa(ctx: RunContext[HPOADependencies], label: str) -> List[HPOA]:
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
    """
    Return all phenotype.hpoa rows that cite a given PMID in the `reference` field.

    Accepts either "PMID:123456" or bare digits "123456".
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

async def lookup_pmid(pmid: str) -> str:
    """
    Lookup the text of a PubMed article by its PMID.
    A PMID should be of the form "PMID:nnnnnnn" (no underscores).
    Args:
        pmid: The PubMed ID to look up
        
    Returns:
        str: Full text if available, otherwise abstract
    """
    return await literature_lookup_pmid(pmid)

async def lookup_literature(query: str) -> List[str]:
    """
    Search PubMed for PMIDs matching a text query.
    Args:
        query: The search query
        
    Returns:
        List of matching PMIDs
    """
    return await literature_search_pmids(query)


# Alias for compatibility with hpoa_agent import
async def pubmed_search_pmids(query: str) -> List[str]:
    return await literature_search_pmids(query)


async def filter_hpoa_by_hp(ctx: RunContext[HPOADependencies], hp: str) -> List[HPOA]:
    """
    Return all phenotype.hpoa rows that have a given HPO term in `hpo_id`.

    Accepts either "HP:nnnnnnn" or a label; labels are resolved to an HP ID via
    an ontology search.
    """
    config = ctx.deps or get_config()
    await config.ensure_hpoa_db()

    hp_norm = hp.strip()
    # If not an HP:ID, attempt resolution via adapter search
    if not hp_norm.upper().startswith("HP:"):
        try:
            hp_adapter = config.get_hp_adapter()
            resolved = await _resolve_hp_term_by_label(hp_adapter, hp_norm)
            if resolved:
                hp_norm = resolved
        except Exception:
            pass

    con = sqlite3.connect(config.hpoa_db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
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


async def hierarchy_hp(ctx: RunContext[HPOADependencies], label: str) -> Dict[str, Any]:
    """
    Given an HPO label or CURIE, resolve to a term and return its immediate
    hierarchical parents via outgoing relationships, plus basic metadata and
    the transitive ancestor set.

    Returns:
      {
        "id": HP:xxxxx,
        "label": <label>,
        "parents": [
            {"rel": <CURIE>, "rel_label": <label>, "parent": <HP:xxxx>, "parent_label": <label>}, ...
        ]
        "ancestors": [HP:xxxx, ...]
      }
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()

    # Accept either CURIE or label
    term_id = label.strip() if label.strip().upper().startswith("HP:") else await _resolve_hp_term_by_label(hp, label)
    if term_id is None:
        return {"error": f"No HPO term found for label '{label}'"}

    parents: List[Dict[str, str]] = []
    try:
        for rel, parent in _hp_outgoing(hp, term_id):
            try:
                parents.append({
                    "rel": rel,
                    "rel_label": _hp_label(hp, rel),
                    "parent": parent,
                    "parent_label": _hp_label(hp, parent),
                })
            except Exception:
                parents.append({
                    "rel": rel,
                    "rel_label": None,
                    "parent": parent,
                    "parent_label": None,
                })
    except Exception as e:
        return {"id": term_id, "label": _hp_label(hp, term_id), "parents": [], "ancestors": [], "warning": f"relationships error: {e}"}

    # Transitive ancestors for quick category checks (e.g., neurological root)
    try:
        ancestors = _hp_ancestors(hp, term_id)
    except Exception:
        ancestors = []

    return {"id": term_id, "label": _hp_label(hp, term_id), "parents": parents, "ancestors": ancestors}


async def get_category_root(ctx: RunContext[HPOADependencies], category_label: str) -> Dict[str, Any]:
    """Resolve a top-level category under HP:0000118 (Phenotypic abnormality).

    Tries to match the label against the immediate children of HP:0000118.
    Returns {id, label}. If not found, returns {error}.
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    key = category_label.strip().lower()
    if key in _CATEGORY_LABEL_TO_ID:
        cid = _CATEGORY_LABEL_TO_ID[key]
        return {"id": cid, "label": _hp_label(hp, cid)}

    # Fetch immediate children of the category root (cache)
    try:
        children = _hp_children(hp, CATEGORY_ROOT)
    except Exception:
        children = []

    # Try exact label match among children
    for cid in children:
        lab = (_hp_label(hp, cid) or "").strip().lower()
        if lab and (lab == key or key in lab):
            _CATEGORY_LABEL_TO_ID[key] = cid
            return {"id": cid, "label": _hp_label(hp, cid)}

    # Fallback: resolve via general label search and then map to nearest category ancestor
    term = await _resolve_hp_term_by_label(hp, category_label)
    if term:
        # find first ancestor that is an immediate child of CATEGORY_ROOT
        try:
            anc = set(_hp_ancestors(hp, term))
            for cid in children:
                if cid in anc or term == cid:
                    _CATEGORY_LABEL_TO_ID[key] = cid
                    return {"id": cid, "label": _hp_label(hp, cid)}
        except Exception:
            pass

    return {"error": f"No category root found for '{category_label}' under {CATEGORY_ROOT}"}


async def is_hpo_in_category(ctx: RunContext[HPOADependencies], hpo_id: str, category_label: str) -> bool:
    """Check if a given HPO term is under a top-level category (immediate child of HP:0000118)."""
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    root = await get_category_root(ctx, category_label)
    cid = root.get("id") if isinstance(root, dict) else None
    if not cid:
        return False
    if hpo_id == cid:
        return True
    try:
        return cid in set(_hp_ancestors(hp, hpo_id))
    except Exception:
        return False


async def _resolve_hp_term_by_label(hp_adapter, label: str) -> Optional[str]:
    """Resolve an HPO label to a CURIE using oaklib search, preferring exact matches."""
    # Try exact label match by scanning results
    try:
        # First: non-partial search may behave like exact or broader depending on adapter
        results = list(hp_adapter.basic_search(label, SearchConfiguration(is_partial=False)))
    except Exception:
        results = []
    if not results:
        try:
            results = list(hp_adapter.basic_search(label, SearchConfiguration(is_partial=True)))
        except Exception:
            results = []

    # Prefer HP: terms and where label matches case-insensitively
    lc = label.strip().lower()
    hp_terms = [curie for curie in results if isinstance(curie, str) and curie.startswith("HP:")]
    for curie in hp_terms:
        try:
            if (_hp_label(hp_adapter, curie) or "").strip().lower() == lc:
                return curie
        except Exception:
            pass
    # fallback to first HP term
    if hp_terms:
        return hp_terms[0]
    return None


def _hp_label(hp_adapter, curie: str) -> Optional[str]:
    if curie in _HP_LABEL_CACHE:
        return _HP_LABEL_CACHE[curie]
    try:
        val = hp_adapter.label(curie)
    except Exception:
        val = None
    _HP_LABEL_CACHE[curie] = val
    return val


def _hp_ancestors(hp_adapter, curie: str) -> List[str]:
    if curie in _HP_ANCESTORS_CACHE:
        return _HP_ANCESTORS_CACHE[curie]
    try:
        vals = list(hp_adapter.ancestors(curie))
    except Exception:
        vals = []
    _HP_ANCESTORS_CACHE[curie] = vals
    return vals


def _hp_outgoing(hp_adapter, curie: str) -> List[tuple]:
    if curie in _HP_OUTGOING_CACHE:
        return _HP_OUTGOING_CACHE[curie]
    try:
        vals = list(hp_adapter.outgoing_relationships(curie))
    except Exception:
        vals = []
    _HP_OUTGOING_CACHE[curie] = vals
    return vals


def _hp_children(hp_adapter, curie: str) -> List[str]:
    if curie in _HP_CHILDREN_CACHE:
        return _HP_CHILDREN_CACHE[curie]
    vals: List[str] = []
    # Try adapter.children if available; fallback to scan outgoing relationships of all terms is too heavy, so skip
    try:
        if hasattr(hp_adapter, "children"):
            vals = list(hp_adapter.children(curie))  # type: ignore[attr-defined]
    except Exception:
        vals = []
    _HP_CHILDREN_CACHE[curie] = vals
    return vals

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
        "api_key": NCBI_API_KEY or "",
    }
    headers = {"Accept": "application/json"}

    print(f"SEARCH PUBMED FOR PMIDs RELATED TO: {query}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
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

    # Extract PMIDs from JSON
    pmids = data.get("esearchresult", {}).get("idlist", [])
    pmids = [f"PMID:{p}" for p in pmids]

    return pmids
