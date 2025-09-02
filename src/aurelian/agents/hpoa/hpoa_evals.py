"""
Basic eval tests for the HPOA agent/tools.
Run with: pytest -q src/aurelian/agents/hpoa/hpoa_evals.py
"""
import os
from pathlib import Path
import pytest

from pydantic_ai import RunContext

from aurelian.agents.hpoa.hpoa_config import HPOADependencies
from aurelian.agents.hpoa.hpoa_tools import filter_hpoa, filter_hpoa_by_pmid


HEADER = "\t".join([
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
])


def write_hpoa_fixture(tmpdir: Path) -> Path:
    rows = [
        ["OMIM:123456", "Foo syndrome", "", "HP:0000001", "PMID:111", "PCS", "", "", "", "", "P", "HPO:cur"],
        ["OMIM:123456", "Foo syndrome", "", "HP:0000002", "OMIM:123456", "IEA", "", "", "", "", "P", "HPO:cur"],
        ["MONDO:0000001", "X syndrome", "", "HP:0000001", "PMID:222", "TAS", "", "", "", "", "P", "HPO:cur"],
    ]
    content = HEADER + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    path = tmpdir / "phenotype.hpoa"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_config_reads_env_path(monkeypatch, tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    monkeypatch.setenv("AURELIAN_HPOA_PATH", str(fp))
    deps = HPOADependencies()
    rows = await deps.fetch_and_parse_hpoa()
    assert len(rows) == 3
    assert rows[0]["disease_name"] == "Foo syndrome"


def _ctx_with_env() -> RunContext[HPOADependencies]:
    deps = HPOADependencies()
    return RunContext[HPOADependencies](deps=deps, model=None, usage=None, prompt=None)


@pytest.mark.asyncio
async def test_filter_hpoa_by_name_and_id(monkeypatch, tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    monkeypatch.setenv("AURELIAN_HPOA_PATH", str(fp))
    rc = _ctx_with_env()

    res1 = await filter_hpoa(rc, "foo")
    assert len(res1) == 2
    assert all(r.disease_name == "Foo syndrome" for r in res1)

    res2 = await filter_hpoa(rc, "MONDO:0000001")
    assert len(res2) == 1
    assert res2[0].database_id == "MONDO:0000001"


@pytest.mark.asyncio
async def test_filter_hpoa_by_pmid(monkeypatch, tmp_path: Path):
    fp = write_hpoa_fixture(tmp_path)
    monkeypatch.setenv("AURELIAN_HPOA_PATH", str(fp))
    rc = _ctx_with_env()

    res = await filter_hpoa_by_pmid(rc, "PMID:111")
    assert len(res) == 1
    assert res[0].hpo_id == "HP:0000001"
