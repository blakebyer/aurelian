"""
MCP tools for interacting with HPOA files.
"""
import os
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

import aurelian.agents.hpoa.hpoa_tools as ht
from aurelian.agents.hpoa.hpoa_agent import HPOA_SYSTEM_PROMPT
from aurelian.agents.hpoa.hpoa_config import HPOADependencies, HPOA, get_config
from pydantic_ai import RunContext

# Initialize FastMCP server
mcp = FastMCP("hpoa", instructions=HPOA_SYSTEM_PROMPT)    

from aurelian.dependencies.workdir import WorkDir

def deps() -> HPOADependencies:
    deps = get_config()
    # Set the location from environment variable or default
    loc = os.getenv("AURELIAN_WORKDIR", "/tmp/aurelian")
    deps.workdir = WorkDir(loc)
    return deps

def ctx() -> RunContext[HPOADependencies]:
    rc: RunContext[HPOADependencies] = RunContext[HPOADependencies](
        deps=deps(),
        model=None, usage=None, prompt=None,
    )
    return rc

@mcp.tool()
async def search_hp(term: str) -> List[dict]:
    """Search HPO by ID or label (HP:nnnnnnn or text). Returns top matches."""
    return await ht.search_hp(ctx(), term)


@mcp.tool()
async def search_mondo(term: str) -> List[dict]:
    """Search MONDO by ID or label (MONDO:nnnnnnn or text). Returns top matches."""
    return await ht.search_mondo(ctx(), term)


@mcp.tool()
async def get_omim_terms(label: str) -> dict:
    """
    Search the OMIM DB for disease identifiers.
    """
    return await ht.get_omim_terms(ctx(), label)

@mcp.tool()
async def get_omim_clinical(label: str) -> dict:
    """
    Search the OMIM DB for clinical synopses.
    """
    return await ht.get_omim_clinical(ctx(), label)

@mcp.tool()
async def filter_hpoa(label: str):
    """
    Filter HPOA to a disease or diseases of interest based on OMIM, ORPHA, or disease names.
    """
    return await ht.filter_hpoa(ctx(), label)

@mcp.tool()
async def filter_hpoa_by_pmid(pmid: str):
    """Return HPOA rows that cite a given PMID (PMID:nnnnnnn or digits)."""
    return await ht.filter_hpoa_by_pmid(ctx(), pmid)

@mcp.tool()
async def filter_hpoa_by_hp(hp: str):
    """Return HPOA rows with a given phenotype (HP:ID or label)."""
    return await ht.filter_hpoa_by_hp(ctx(), hp)

@mcp.tool()
async def categorize_hpo(hp: str) -> List[str]:
    """Categorize an HPO term into top-level organ-system buckets under HP:0000118.

    Args:
        hp: An HPO identifier (HP:nnnnnnn) or label

    Returns:
        A list of strings like "HP:xxxxxxx | Label" for matching system categories
    """
    return await ht.categorize_hpo(ctx(), hp)

@mcp.tool()
async def lookup_pmid(pmid: str) -> str:
    """
    Lookup a PubMed ID to get the article abstract.
    """
    return await ht.lookup_pmid(pmid)

@mcp.tool()
async def pubmed_search_pmids(query: str) -> List[str]:
    """Search PubMed (NCBI ESearch) for PMIDs matching a query. Returns ["PMID:nnnnnnn", ...]."""
    return await ht.pubmed_search_pmids(ctx(), query)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
