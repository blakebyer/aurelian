""" Configuration file for HPOA Agent """
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import os, csv, subprocess, sqlite3
from io import StringIO
from typing import cast
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # fallback to csv module
import httpx
from typing import Optional, List, Dict, Any, TypedDict, Literal
from oaklib import get_adapter
from oaklib.interfaces import BasicOntologyInterface
from datetime import date

from aurelian.dependencies.workdir import HasWorkdir, WorkDir

class HPOA(BaseModel):
    database_id: str = Field(..., description="Refers to the database `disease_name` is drawn from. Must be formatted as a CURIE, e.g., OMIM:1547800 or MONDO:0021190")
    disease_name: str = Field(..., description="This is the name of the disease associated with the `database_id` in the database. Only the accepted name should be used, synonyms should not be listed here.")	
    qualifier: Optional[Literal["", "NOT"]] = Field(..., description="""This field is used to qualify the annotation shown in field `hpo_id`. The field can only be used to record `NOT` or is empty. A value of NOT indicates that the disease in question is not characterized by the indicated HPO term. This is used to record phenotypic features that can be of special differential diagnostic utility.""")
    hpo_id: str = Field(..., description="This field is for the HPO identifier for the term attributed to the `disease_name`.")
    reference: str = Field(..., description="""This field indicates the source of the information used for the annotation. This may be the clinical experience of the annotator, an article as indicated by a PMID, or an HPO collaborator ID, e.g. HPO:RefId. If a PMID cannot be found, default back to OMIM:mimNumber.""")	
    evidence: Literal["IEA", "PCS", "TAS"] = Field(..., description="""IEA (inferred from electronic annotation): annotations extracted from OMIM.
                                                   PCS (published clinical study): annotations extracted from articles in the medical literature.
                                                   TAS (traceable author statement): annotations extracted from knowledge bases such as OMIM or Orphanet that have derived the information from a published source..""")
    onset: Optional[str] = Field(..., description="""A term-id from the HPO-sub-ontology below the term `Age of onset` (HP:0003674). Note that if an HPO onset term is used in this field, it refers to the onset of the feature specified in field `hpo_id` in the disease being annotated. If an HPO term is used for age of onset in field `hpo_id` then it refers to the overall age of onset of the disease.""")
    frequency: Optional[str] = Field(..., description="""There are three allowed options for this field. (A) A term-id from the HPO-sub-ontology below the term `Frequency` (HP:0040279), (B) A count of patients affected within a cohort. For instance, 7/13 would indicate 7 of 13 patients with the disease in the `reference` field study were affected by the phenotype in the `hpo_id` field, and (C) A percentage value such as 17%.""")	
    sex: Optional[Literal["MALE", "FEMALE", ""]] = Field(..., description="""This field contains the strings MALE or FEMALE if the annotation in question is limited to males or females. This field refers to the phenotypic (and not the chromosomal) sex. If a phenotype is limited to one sex then a modifier from the clinical modifier subontology should be noted in the modifier field.""")	
    modifier: Optional[str]	= Field(..., description="A term-id from the HPO-sub-ontology below the term `Clinical modifier`.")
    aspect: Literal["P", "I", "C", "M"] = Field(..., description="""Terms with the P aspect are located in the Phenotypic abnormality subontology.
                              Terms with the I aspect are from the Inheritance subontology.
                              Terms with the C aspect are located in the Clinical course subontology, which includes onset, mortality, and other terms related to the temporal aspects of disease.
                              Terms with the M aspect are located in the Clinical Modifier subontology.""")	
    biocuration: str = Field(..., 
                             default_factory = lambda: f"HPO:Agent[{date.today().isoformat()}]",description="""This refers to the biocurator who made the annotation and the date on which the annotation was made; the date format is YYYY-MM-DD. The first entry in this field refers to the creation date. Any additional biocuration is recorded following a semicolon. So, if Joseph curated on July 5, 2012, and Suzanna curated on December 7, 2015, one might have a field like this: HPO:Joseph[2012-07-05];HPO:Suzanna[2015-12-07]. It is acceptable to use ORCID ids.""")

class HPOAResult(BaseModel):
    status: Literal["existing", "new", "updated", "removed"] = Field(
        ..., description="Whether this annotation was existing, new, updated, or suggested for removal from the phenotype.hpoa file."
    )
    rationale: Optional[str] = None
    annotation: HPOA

class HPOAResponse(BaseModel):
    explanation: str = Field(..., description="A brief natural language explanation of what was found and done.")
    annotations: List[HPOAResult]

class HPOAMixedResponse(BaseModel):
    """
    Flexible output for conversational + structured use.

    - text: free-form response for conversational answers and reasoning
    - annotations: optional structured block for curation actions; leave empty when not applicable
    """
    text: str = Field(..., description="Free text response and/or reasoning narrative.")
    annotations: List[HPOAResult] = Field(default_factory=list, description="Structured HPOA changes; empty when not proposing changes.")

@dataclass
class HPOADependencies(HasWorkdir):
    """Configuration for the HPOA agent."""
    openai_api_key: Optional[str] = None
    omim_api_key: Optional[str] = None
    ncbi_api_key: Optional[str] = None
    hpoa_db_path: Optional[str] = None
    _hp_adapter: Optional[BasicOntologyInterface] = field(default=None, init=False, repr=False)
    _mondo_adapter: Optional[BasicOntologyInterface] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the config with default values."""
        # HasWorkdir doesn't have a __post_init__ method, so we don't call super()
        if self.workdir is None:
            self.workdir = WorkDir()

        if self.openai_api_key is None:
            import os
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if self.omim_api_key is None:
            import os
            self.omim_api_key = os.environ.get("OMIM_API_KEY")
        
        if self.ncbi_api_key is None:
            import os
            self.ncbi_api_key = os.environ.get("NCBI_API_KEY")

        # establish default DB path
        if self.hpoa_db_path is None:
            # Prefer explicit env workdir; otherwise, use current working directory.
            base = os.environ.get("AURELIAN_WORKDIR") or os.getcwd()
            self.hpoa_db_path = os.path.join(base, "hpoa.db")

    def get_mondo_adapter(self) -> BasicOntologyInterface:
        """Get a configured Mondo adapter."""
        if self._mondo_adapter is None:
            self._mondo_adapter = get_adapter("sqlite:obo:mondo")
        return self._mondo_adapter
        #return get_adapter("ontobee:mondo")
    
    def get_hp_adapter(self) -> BasicOntologyInterface:
        """Get a configured HPO adapter."""
        if self._hp_adapter is None:
            self._hp_adapter = get_adapter("sqlite:obo:hp")
        return self._hp_adapter
        #return get_adapter("ontobee:hp")
    
    async def fetch_and_parse_hpoa(self, path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Load and parse phenotype.hpoa into a list of dicts.

        Resolution order:
        1) Explicit `path` argument if provided and exists
        2) Env var `AURELIAN_HPOA_PATH` if set and exists
        3) Cached file in workdir (if present): `${workdir}/phenotype.hpoa`
        4) Download latest release from GitHub and cache to workdir
        """
        # 1) Explicit path
        candidate_path = path
        if candidate_path and os.path.exists(candidate_path):
            if pd is not None:
                df = pd.read_csv(candidate_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
                self._persist_df_to_db(df)
                return cast(List[Dict[str, str]], df.to_dict("records"))
            else:
                rows = _read_hpoa_from_path(candidate_path)
                self._persist_hpoa_to_db(rows)
                return rows

        # 2) Environment variable
        env_path = os.environ.get("AURELIAN_HPOA_PATH")
        if env_path and os.path.exists(env_path):
            if pd is not None:
                df = pd.read_csv(env_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
                self._persist_df_to_db(df)
                return cast(List[Dict[str, str]], df.to_dict("records"))
            else:
                rows = _read_hpoa_from_path(env_path)
                self._persist_hpoa_to_db(rows)
                return rows

        # 3) Cached copy in workdir
        cached_path = None
        if self.workdir and self.workdir.location:
            cached_path = os.path.join(self.workdir.location, "phenotype.hpoa")
            if os.path.exists(cached_path):
                if pd is not None:
                    df = pd.read_csv(cached_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
                    self._persist_df_to_db(df)
                    return cast(List[Dict[str, str]], df.to_dict("records"))
                else:
                    rows = _read_hpoa_from_path(cached_path)
                    self._persist_hpoa_to_db(rows)
                    return rows

        # 4) Download latest and cache
        client = await get_client()
        r = await client.get("https://api.github.com/repos/obophenotype/human-phenotype-ontology/releases/latest")
        r.raise_for_status()
        url = next(
            a["browser_download_url"]
            for a in r.json().get("assets", [])
            if "phenotype.hpoa" in a.get("browser_download_url", "")
        )
        f = await client.get(url)
        f.raise_for_status()
        text = f.text

        # Cache if possible
        if cached_path:
            try:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                with open(cached_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except Exception:
                # Non-fatal; continue without cache
                pass

        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        if pd is not None:
            import pandas as _pd  # type: ignore
            df = _pd.read_csv(StringIO(text), sep="\t", comment="#", dtype=str, keep_default_na=False)
            self._persist_df_to_db(df)
            return cast(List[Dict[str, str]], df.to_dict("records"))
        else:
            reader = csv.DictReader(lines, delimiter="\t")
            rows = list(reader)
            self._persist_hpoa_to_db(rows)
            return rows

    async def ensure_hpoa_db(self, path: Optional[str] = None) -> None:
        """Ensure the SQLite DB is present and populated with HPOA rows.

        If the DB file/table is missing or empty, load TSV using fetch_and_parse_hpoa.
        """
        if not self.hpoa_db_path:
            # initialize default
            base = os.environ.get("AURELIAN_WORKDIR") or os.getcwd()
            self.hpoa_db_path = os.path.join(base, "hpoa.db")

        need_load = False
        if not os.path.exists(self.hpoa_db_path):
            need_load = True
        else:
            try:
                con = sqlite3.connect(self.hpoa_db_path)
                cur = con.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hpoa'")
                has_table = cur.fetchone() is not None
                if not has_table:
                    need_load = True
                else:
                    # lightweight presence check instead of full COUNT(*)
                    cur.execute("SELECT 1 FROM hpoa LIMIT 1")
                    if cur.fetchone() is None:
                        need_load = True
            finally:
                try:
                    con.close()
                except Exception:
                    pass

        if need_load:
            await self.fetch_and_parse_hpoa(path=path)

    def _persist_hpoa_to_db(self, rows: List[Dict[str, str]]) -> None:
        """Persist HPOA rows into SQLite DB (overwrites existing table)."""
        if not self.hpoa_db_path:
            base = os.environ.get("AURELIAN_WORKDIR") or os.getcwd()
            self.hpoa_db_path = os.path.join(base, "hpoa.db")

        db_dir = os.path.dirname(self.hpoa_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        con = sqlite3.connect(self.hpoa_db_path)
        try:
            # Set PRAGMAs BEFORE starting a transaction; changing these inside a transaction
            # causes: "Safety level may not be changed inside a transaction".
            con.execute("PRAGMA journal_mode = MEMORY")
            con.execute("PRAGMA synchronous = OFF")

            # Explicit transaction for bulk load
            con.execute("BEGIN IMMEDIATE")
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hpoa (
                    database_id TEXT,
                    disease_name TEXT,
                    qualifier TEXT,
                    hpo_id TEXT,
                    reference TEXT,
                    evidence TEXT,
                    onset TEXT,
                    frequency TEXT,
                    sex TEXT,
                    modifier TEXT,
                    aspect TEXT,
                    biocuration TEXT
                )
                """
            )
            # overwrite existing table contents
            cur.execute("DELETE FROM hpoa")

            to_tuples = [
                (
                    row.get("database_id"),
                    row.get("disease_name"),
                    row.get("qualifier"),
                    row.get("hpo_id"),
                    row.get("reference"),
                    row.get("evidence"),
                    row.get("onset"),
                    row.get("frequency"),
                    row.get("sex"),
                    row.get("modifier"),
                    row.get("aspect"),
                    row.get("biocuration"),
                )
                for row in rows
            ]
            cur.executemany(
                "INSERT INTO hpoa (database_id, disease_name, qualifier, hpo_id, reference, evidence, onset, frequency, sex, modifier, aspect, biocuration) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                to_tuples,
            )
            # indexes to accelerate lookups (including expression indexes used in queries)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid ON hpoa(database_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid_norm ON hpoa(UPPER(REPLACE(database_id,' ','')))")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dname_nocase ON hpoa(disease_name COLLATE NOCASE)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_ref_upper ON hpoa(UPPER(reference))")
            con.commit()
        finally:
            con.close()

    def _persist_df_to_db(self, df):
        """Persist a pandas DataFrame to SQLite, replacing the hpoa table and adding indexes."""
        if not self.hpoa_db_path:
            base = os.environ.get("AURELIAN_WORKDIR") or os.getcwd()
            self.hpoa_db_path = os.path.join(base, "hpoa.db")

        db_dir = os.path.dirname(self.hpoa_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        con = sqlite3.connect(self.hpoa_db_path)
        try:
            con.execute("PRAGMA journal_mode = MEMORY")
            con.execute("PRAGMA synchronous = OFF")
            # Replace table using pandas in a single efficient transaction
            df.to_sql("hpoa", con, if_exists="replace", index=False, chunksize=5000, method=None)
            cur = con.cursor()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid ON hpoa(database_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid_norm ON hpoa(UPPER(REPLACE(database_id,' ','')))")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dname_nocase ON hpoa(disease_name COLLATE NOCASE)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_ref_upper ON hpoa(UPPER(reference))")
            con.commit()
        finally:
            con.close()
    
    
def get_config() -> HPOADependencies:
    """Get the HPOA configuration from environment variables or defaults."""
    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    workdir = WorkDir(location=workdir_path) if workdir_path else None
    
    return HPOADependencies(
        workdir=workdir
    )    


def _read_hpoa_from_path(path: str) -> List[Dict[str, str]]:
    """Read a local phenotype.hpoa TSV file preserving columns."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    return list(reader)


# Shared Async HTTP client for session reuse
_async_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """Get or create a shared AsyncClient for HTTP requests."""
    global _async_client
    if _async_client is None:
        # Enable HTTP/2 when available; otherwise, gracefully fall back to HTTP/1.1
        try:
            import h2  # noqa: F401
            http2_flag = True
        except Exception:
            http2_flag = False
        _async_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True, http2=http2_flag)
    return _async_client


async def close_client() -> None:
    """Close the shared AsyncClient and reset it."""
    global _async_client
    if _async_client is not None:
        try:
            await _async_client.aclose()
        finally:
            _async_client = None
