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
async def search_hp(label: str) -> List[dict]:
    """
    Find associations for a given gene or gene product.

    Args:
        gene_id: Gene identifier (e.g., HGNC symbol like "BRCA1")
        
    Returns:
        List of association objects containing subject, predicate, object details
    """
    return await ht.search_hp(ctx(), label)


@mcp.tool()
async def find_disease_associations(disease_id: str) -> List[Dict]:
    """
    Find associations for a given disease.

    Args:
        disease_id: Disease identifier (e.g., MONDO:0007254)
        
    Returns:
        List of association objects containing subject, predicate, object details
    """
    return await ht.find_disease_associations(ctx(), disease_id)

@mcp.tool()
async def search_mondo(label: str) -> List[dict]:
    """
    Search the MONDO Ontology for disease identifiers.
    """
    return await ht.search_mondo(ctx(), label)

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
async def lookup_pmid(pmid: str) -> str:
    """
    Lookup a PubMed ID to get the article abstract.
    """
    return await ht.lookup_pmid(pmid)

@mcp.tool()
async def search_literature_pmids(query: str) -> List[str]:
    """
    Search the web for scientific literature using a text query.
    
    Args:
        query: The search query (e.g., "alzheimer's disease genetics 2023")
        
    Returns:
        List of PMIDs matching the query
    """
    return await ht.literature_search_pmids(query)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')