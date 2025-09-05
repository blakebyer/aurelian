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
import inspect as _inspect

async def search_hp(ctx: RunContext[HPOADependencies], term: str) -> List[dict]:
    """Search the HPO for phenotypic abnormalities by ID or label.

    - If `term` starts with "HP:", normalize and return that term (id/label/definition).
    - Otherwise, run a basic partial label search and return HP terms.
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
        if _inspect.iscoroutine(bs):
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
    """Search the MONDO ontology by ID or label.

    - If `term` starts with "MONDO:", normalize and return that term (id/label/definition).
    - Otherwise, run a basic partial label search and return MONDO terms.
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
        if _inspect.iscoroutine(bs):
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
    Search the Web for PMIDs matching a text query.
    Args:
        query: The search query
        
    Returns:
        List of matching PMIDs
    """
    return await literature_search_pmids(query)

async def filter_hpoa_by_hp(ctx: RunContext[HPOADependencies], hp: str) -> List[HPOA]:
    """
    Return all phenotype.hpoa rows that have a given HPO term in `hpo_id`.

    Accepts either an HP:ID (e.g., "HP:0001250") or a phenotype label.
    If a label is provided, resolves to the top HP:ID via search_hp.
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

# Helper functions for dealing with HPO hierarchy
HP_SYSTEM_ROOT = "HP:0000118"  # Phenotypic abnormality

def children_of(ctx: RunContext[HPOADependencies], parent: str) -> List[str]:
    """Direct children = subjects of subclass edges pointing to parent.

    Note: synchronous helper (no awaiting needed).
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    try:
        return [s for s, p, o in hp.relationships(objects=[parent])]
    except Exception:
        return []

def parents_of(ctx: RunContext[HPOADependencies], child: str) -> List[str]:
    """Direct parents = objects of subclass edges from child.

    Note: synchronous helper (no awaiting needed).
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    try:
        return [o for s, p, o in hp.relationships(subjects=[child])]
    except Exception:
        return []

async def categorize_hpo(ctx: RunContext[HPOADependencies], term: str) -> List[str]:
    """
    Categorize a term into top-level systems under HP:0000118.
    Returns list like: ["HP:xxxxxxx (Label)", ...].
    """
    config = ctx.deps or get_config()
    hp = config.get_hp_adapter()
    systems = children_of(ctx, HP_SYSTEM_ROOT)
    try:
        ancestors = set(hp.ancestors(term, reflexive=True) or [])
    except Exception:
        ancestors = set()
    return [f"{s} ({hp.label(s)})" for s in systems if s in ancestors]

async def categorize_mondo(ctx: RunContext[HPOADependencies], term: str) -> List[str]:
    """
    Categorize a MONDO term into top-level categories under MONDO:0700096 (human disease).
    Returns list like: ["MONDO:xxxxxxx (Label)", ...].
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
