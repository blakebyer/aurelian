"""
Basic eval tests for the HPOA agent/tools.
Run with: poetry run pytest -q src/aurelian/agents/hpoa/hpoa_evals.py
"""
from pathlib import Path
import os
import pytest
from pydantic_ai import RunContext

from aurelian.agents.hpoa.hpoa_config import HPOADependencies
import aurelian.agents.hpoa.hpoa_tools as tools

# --------------------------------------------------------------------
# Fixture builder for phenotype.hpoa
# --------------------------------------------------------------------
HEADER = "\t".join([
    "database_id","disease_name","qualifier","hpo_id","reference","evidence",
    "onset","frequency","sex","modifier","aspect","biocuration",
])

def write_hpoa_fixture(tmpdir: Path) -> Path:
    rows = [
        ["OMIM:123456","Foo syndrome","","HP:0000001","PMID:111","PCS","","","","","P","HPO:cur"],
        ["OMIM:123456","Foo syndrome","","HP:0000002","OMIM:123456","IEA","","","","","P","HPO:cur"],
        ["MONDO:0000001","X syndrome","","HP:0000001","PMID:222","TAS","","","","","P","HPO:cur"],
    ]
    content = HEADER + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    path = tmpdir / "phenotype.hpoa"
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _ctx() -> RunContext[HPOADependencies]:
    deps = HPOADependencies()
    return RunContext[HPOADependencies](deps=deps, model=None, usage=None, prompt=None)

def _ctx_with_fixture(fp: Path) -> RunContext[HPOADependencies]:
    db_path = fp.parent / 'hpoa-test.db'
    os.environ['HPOA_TSV'] = str(fp)
    os.environ['HPOA_DB'] = str(db_path)
    deps = HPOADependencies(hpoa_db_path=str(db_path))
    return RunContext[HPOADependencies](deps=deps, model=None, usage=None, prompt=None)


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_config_reads_fixture(tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    deps = HPOADependencies()
    rows = await deps.fetch_and_parse_hpoa(path=str(fp))
    assert len(rows) == 3
    assert rows[0]["disease_name"] == "Foo syndrome"


@pytest.mark.asyncio
async def test_filter_hpoa_by_name_and_id(tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    rc = _ctx_with_fixture(fp)

    res1 = await tools.filter_hpoa_by_disease(rc, "foo")
    assert len(res1) == 2
    assert all(r.disease_name == "Foo syndrome" for r in res1)

    res2 = await tools.filter_hpoa_by_disease(rc, "MONDO:0000001")
    assert len(res2) == 1
    assert res2[0].database_id == "MONDO:0000001"


@pytest.mark.asyncio
async def test_filter_hpoa_by_pmid(tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    rc = _ctx_with_fixture(fp)

    res = await tools.filter_hpoa_by_pmid(rc, "PMID:111")
    assert len(res) == 1
    assert res[0].hpo_id == "HP:0000001"


@pytest.mark.asyncio
async def test_filter_hpoa_by_hp(tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    rc = _ctx_with_fixture(fp)

    res = await tools.filter_hpoa_by_hp(rc, "HP:0000001")
    assert res and res[0].database_id == "OMIM:123456"


@pytest.mark.asyncio
async def test_search_hp_and_mondo():
    rc = _ctx()

    class DummyHP:
        def label(self, curie): return "Test HP"
        def definition(self, curie): return "Def"
        def basic_search(self, q, cfg): return ["HP:123456"]
        def ancestors(self, term, reflexive=True): return {"HP:0000118", term}
        def relationships(self, **kwargs): 
            return [("HP:123456", "subClassOf", kwargs.get("objects")[0])]

    class DummyMONDO:
        def label(self, curie): return "Test MONDO"
        def definition(self, curie): return "Def"
        def basic_search(self, q, cfg): return ["MONDO:1234"]
        def ancestors(self, term, reflexive=True): return {"MONDO:0700096", term}
        def relationships(self, **kwargs):
            return [("MONDO:1234", "subClassOf", kwargs.get("objects")[0])]

    # Patch adapters and their getters
    rc.deps._hp_adapter = DummyHP()
    rc.deps.get_hp_adapter = lambda: rc.deps._hp_adapter
    rc.deps._mondo_adapter = DummyMONDO()
    rc.deps.get_mondo_adapter = lambda: rc.deps._mondo_adapter

    hp_res = await tools.search_hp(rc, "x")
    assert hp_res and hp_res[0]["id"].startswith("HP:")

    mondo_res = await tools.search_mondo(rc, "y")
    assert mondo_res and mondo_res[0]["id"].startswith("MONDO:")

    cats = await tools.categorize_hpo(rc, "HP:123456")
    assert any("HP:" in c for c in cats)

    mondo_cats = await tools.categorize_mondo(rc, "MONDO:1234")
    assert any("MONDO:" in c for c in mondo_cats)

@pytest.mark.asyncio
async def test_pubmed_and_literature(monkeypatch):
    rc = _ctx()

    class DummyResp:
        def raise_for_status(self): return None
        def json(self): return {"esearchresult": {"idlist": ["12345"]}}

    class DummyClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def get(self, *a, **k): return DummyResp()

    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_tools.httpx.AsyncClient", lambda *a, **k: DummyClient())

    pmids = await tools.pubmed_search_pmids(rc, "alz", retmax=5)
    assert pmids == ["PMID:12345"]