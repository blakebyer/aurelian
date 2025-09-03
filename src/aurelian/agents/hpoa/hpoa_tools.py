"""
Tools for interacting with MONDO, HPO, and HPOA files.
"""
import asyncio
from typing import Dict, List, Any
import requests
import httpx
import re, os
from pydantic_ai import RunContext, ModelRetry
from .hpoa_config import HPOADependencies, HPOA, get_config
from aurelian.agents.literature.literature_tools import (
    lookup_pmid as literature_lookup_pmid,
    literature_search_pmids as literature_search_pmids,
    )
from oaklib.datamodels.search import SearchConfiguration

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
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
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
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
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
    rows = await config.fetch_and_parse_hpoa()

    q_raw = label.strip()
    q_norm = q_raw.upper().replace(" ", "")
    q_lower = q_raw.lower()

    # Detect CURIE-style disease IDs
    id_pattern = r"(OMIM|ORPHA|MONDO|DECIPHER):[A-Z0-9_.-]+"
    id_search = re.search(id_pattern, q_norm)
    q_id = id_search.group(0) if id_search else None

    results: List[HPOA] = []
    for row in rows:
        disease_name = (row.get("disease_name") or "")
        disease_id = (row.get("database_id") or "")

        try:
            if q_id:
                # Match directly against database_id
                if disease_id.upper().replace(" ", "") == q_id:
                    results.append(HPOA(**row))
            else:
                # Fallback: partial string match on disease_name
                if q_lower in disease_name.lower():
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
    rows = await config.fetch_and_parse_hpoa()
    pid = pmid.strip().replace("PMID:", "").strip()
    results: List[HPOA] = []
    for row in rows:
        ref = (row.get("reference") or "").upper()
        if f"PMID:{pid}" in ref:
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