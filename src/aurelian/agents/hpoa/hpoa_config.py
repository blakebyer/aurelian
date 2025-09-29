""" Configuration file for HPOA Agent """
import csv
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import httpx
from oaklib import get_adapter
from oaklib.interfaces import BasicOntologyInterface
from pydantic import BaseModel, Field, model_validator
from aurelian.dependencies.workdir import HasWorkdir, WorkDir

# Module-level singletons for ontology adapters to avoid repeated loads
HP_ADAPTER_SINGLETON: Optional[BasicOntologyInterface] = None
MONDO_ADAPTER_SINGLETON: Optional[BasicOntologyInterface] = None
client: Optional[httpx.AsyncClient] = None

def get_client() -> httpx.AsyncClient:
    """Return a shared AsyncClient for connection reuse (faster API calls)."""
    global client
    if client is None:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    return client

class HPOA(BaseModel):
    database_id: str = Field(..., description="Refers to the database `disease_name` is drawn from. Must be formatted as a CURIE, e.g., OMIM:1547800 or MONDO:0021190")
    disease_name: str = Field(..., description="This is the name of the disease associated with the `database_id` in the database. Only the accepted name should be used, synonyms should not be listed here.")	
    qualifier: Optional[Literal["NOT", ""]] = Field(..., description="""This field is used to qualify the annotation shown in field `hpo_id`. The field can only be used to record `NOT` or is empty. A value of NOT indicates that the disease in question is not characterized by the indicated HPO term. This is used to record phenotypic features that can be of special differential diagnostic utility.""")
    hpo_id: str = Field(..., description="This field is for the HPO identifier for the term attributed to the `disease_name`.")
    reference: str = Field(..., description="""This field indicates the source of the information used for the annotation. This may be the clinical experience of the annotator, an article as indicated by a PMID, or an HPO collaborator ID, e.g. HPO:RefId. If a PMID cannot be found, default back to OMIM:mimNumber.""")	
    evidence: Literal["IEA", "PCS", "TAS"] = Field(..., description="""IEA (inferred from electronic annotation): annotations extracted from OMIM.
                                                   PCS (published clinical study): annotations extracted from articles in the medical literature (including PubMed).
                                                   TAS (traceable author statement): annotations extracted from knowledge bases such as OMIM or Orphanet.""")
    onset: Optional[str] = Field(..., description="""A term-id from the HPO-sub-ontology below the term `Age of onset` (HP:0003674). Note that if an HPO onset term is used in this field, it refers to the onset of the feature specified in field `hpo_id` in the disease being annotated. If an HPO term is used for age of onset in field `hpo_id` then it refers to the overall age of onset of the disease.""")
    frequency: Optional[str] = Field(..., description="""Must be a fraction (e.g., 7/13), a percentage (e.g., 17%), or HPO frequency term (below the term HP:0040279). Leave empty if unspecified.""")	
    sex: Optional[Literal["MALE", "FEMALE", ""]] = Field(..., description="""This field contains the strings MALE or FEMALE if the annotation in question is limited to males or females. This field refers to the phenotypic (and not the chromosomal) sex. If a phenotype is limited to one sex then a modifier from the clinical modifier subontology should be noted in the modifier field.""")	
    modifier: Optional[str]	= Field(..., description="A term-id from the HPO-sub-ontology below the term `Clinical modifier`.")
    aspect: Literal["P", "I", "C", "M"] = Field(..., description="""Terms with the P aspect are located in the Phenotypic abnormality subontology.
                              Terms with the I aspect are from the Inheritance subontology.
                              Terms with the C aspect are located in the Clinical course subontology, which includes onset, mortality, and other terms related to the temporal aspects of disease.
                              Terms with the M aspect are located in the Clinical Modifier subontology.""")	
    biocuration: str = Field(..., default_factory=lambda: f"HPO:Agent[{date.today().isoformat()}]", description="""This refers to the biocurator who made the annotation and the date on which the annotation was made; the date format is YYYY-MM-DD. The first entry in this field refers to the creation date. Any additional biocuration is recorded following a semicolon. So, if Joseph curated on July 5, 2012, and Suzanna curated on December 7, 2015, one might have a field like this: HPO:Joseph[2012-07-05];HPO:Suzanna[2015-12-07]. It is acceptable to use ORCID ids.""")
    @model_validator(mode="after")
    def build_biocuration(self):
        config = get_config()
        today = date.today().isoformat()

        tags = []
        existing = config.get_biocurator_metadata(
            self.database_id,
            self.disease_name,
            self.qualifier,
            self.hpo_id,
            self.onset,
            self.frequency,
            self.sex,
            self.modifier,
        )
        # populate with existing entries first, return without HPO Agent and curator
        if existing:
            if isinstance(existing, (list, tuple)):
                existing_str = ";".join(r[0] for r in existing if r and r[0])
            else:
                existing_str = str(existing)
            tags.append(existing_str)
            return self

        # append curator + agent if it's a new or edited curation
        if config.curator_id:
            tags.append(f"HPO:{config.curator_id}[{today}]")
        tags.append(f"HPO:Agent[{today}]")

        self.biocuration = ";".join(tags)
        return self
    
    @model_validator(mode="after")
    def build_reference(self):
        """Ensure reference field is valid; fallback to curator ID if not PMID/OMIM/dbid."""
        ref = (self.reference or "").strip()

        if ref.startswith("PMID:") or ref.startswith("OMIM:") or ref == self.database_id:
            return self

        config = get_config()
        curator_tag = config.curator_id if config.curator_id else "Agent"
        self.reference = f"HPO:{curator_tag}"
        return self

    @model_validator(mode="after")
    def validate_terms(self):
        config = get_config()
        hp_adapter = config.get_hp_adapter()
        mondo_adapter = config.get_mondo_adapter()
    
        # HPO term checks
        if not (self.hpo_id and hp_adapter.label(self.hpo_id)):
            raise ValueError(f"HPO term not found: {self.hpo_id}")

        # optional checks
        if self.onset and not (
            hp_adapter.label(self.onset) and
            self.onset in [s for s in hp_adapter.descendants("HP:0003674")] # onset descendants
            ):
            raise ValueError(f"HPO onset term not found: {self.onset}")

        if self.modifier and not (
            hp_adapter.label(self.modifier) and 
            self.modifier in [s for s in hp_adapter.descendants("HP:0012823")] # clinical modifier descendants
            ):
            raise ValueError(f"HPO modifier term not found: {self.modifier}")

        # database_id is in phenotype.hpoa or exists in mondo
        if not (config.database_id_exists(self.database_id) or mondo_adapter.label(self.database_id)):
            raise ValueError(f"database_id is not in phenotype.hpoa or valid MONDO term: {self.database_id}")

        return self

class HPOAResult(BaseModel):
    status: Literal["existing", "add", "edit", "remove"] = Field(
        ..., description="If an HPOA row is existing, or the agent suggests to add, edit, or remove an HPOA row."
    )
    rationale: Optional[str] = None
    annotation: HPOA

    @model_validator(mode="after")
    def mark_duplicates(self):
        """If annotation already exists in DB, mark status=existing to prevent errant curation."""
        config = get_config()
        ann = self.annotation

        with sqlite3.connect(config.hpoa_db_path) as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT 1 FROM hpoa
                WHERE database_id = ?
                  AND disease_name = ?
                  AND COALESCE(qualifier, '') = COALESCE(?, '')
                  AND hpo_id = ?
                  AND COALESCE(onset, '') = COALESCE(?, '')
                  AND COALESCE(frequency, '') = COALESCE(?, '')
                  AND COALESCE(sex, '') = COALESCE(?, '')
                  AND COALESCE(modifier, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (
                    ann.database_id,
                    ann.disease_name,
                    ann.qualifier,
                    ann.hpo_id,
                    ann.onset,
                    ann.frequency,
                    ann.sex,
                    ann.modifier,
                ),
            )
            if cur.fetchone():
                self.status = "existing"
        return self

class HPOAResponse(BaseModel):
    """
    Flexible output for conversational + structured use.

    - explanation: free-form response for conversational answers, non HPOA JSON, or reasoning narrative
    - annotations: optional structured HPOA block for curation actions; leave empty when not applicable
    """
    explanation: str = Field(..., description="A brief natural language explanation of what was found and done.")
    annotations: List[HPOAResult]

@dataclass
class HPOADependencies(HasWorkdir):
    """Configuration for the HPOA agent."""

    openai_api_key: Optional[str] = None
    omim_api_key: Optional[str] = None
    ncbi_api_key: Optional[str] = None
    curator_id: Optional[str] = None
    cache_dir: Optional[str] = None
    hpoa_db_path: Optional[str] = None
    hpoa_tsv_path: Optional[str] = None
    hp_adapter: Optional[BasicOntologyInterface] = field(default=None, init=False, repr=False)
    mondo_adapter: Optional[BasicOntologyInterface] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the config with default values."""
        if self.workdir is None:
            self.workdir = WorkDir()

        cache_root = Path(self.cache_dir).expanduser().resolve() if self.cache_dir else default_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_root)
        os.environ.setdefault("HPOA_CACHE_DIR", self.cache_dir)

        cache_db = cache_root / "hpoa.db"
        cache_tsv = cache_root / "phenotype.hpoa"

        search_roots = [Path.cwd().resolve()]
        if self.workdir and getattr(self.workdir, "location", None):
            workdir_path = Path(self.workdir.location).expanduser().resolve()
            if workdir_path not in search_roots:
                search_roots.append(workdir_path)

        self._cache_existing_asset("phenotype.hpoa", cache_tsv, search_roots)
        self._cache_existing_asset("hpoa.db", cache_db, search_roots)

        self.hpoa_db_path = str(cache_db)
        self.hpoa_tsv_path = str(cache_tsv)

        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        if self.omim_api_key is None:
            self.omim_api_key = os.environ.get("OMIM_API_KEY")

        if self.ncbi_api_key is None:
            self.ncbi_api_key = os.environ.get("NCBI_API_KEY")

        if self.curator_id is None:
            self.curator_id = os.environ.get("CURATOR_ID")
        if self.curator_id:
            self.curator_id = self.curator_id.strip() or None

    def _cache_existing_asset(self, filename: str, destination: Path, search_roots: List[Path]) -> None:
        for root in search_roots:
            if root is None:
                continue
            candidate = root / filename
            if candidate.exists():
                if destination.resolve() == candidate.resolve():
                    return
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
                return

    def get_mondo_adapter(self) -> BasicOntologyInterface:
        """Get a configured Mondo adapter."""
        global MONDO_ADAPTER_SINGLETON
        if MONDO_ADAPTER_SINGLETON is None:
            MONDO_ADAPTER_SINGLETON = get_adapter("sqlite:obo:mondo")
        return MONDO_ADAPTER_SINGLETON

    def get_hp_adapter(self) -> BasicOntologyInterface:
        """Get a configured HPO adapter."""
        global HP_ADAPTER_SINGLETON
        if HP_ADAPTER_SINGLETON is None:
            HP_ADAPTER_SINGLETON = get_adapter("sqlite:obo:hp")
        return HP_ADAPTER_SINGLETON
    
    def database_id_exists(self, dbid: str) -> bool:
        """Check if a database_id exists in the cached HPOA file."""
        if not self.hpoa_db_path or not os.path.exists(self.hpoa_db_path):
            return False
        con = sqlite3.connect(self.hpoa_db_path)
        try:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM hpoa WHERE database_id = ? LIMIT 1", (dbid,))
            return cur.fetchone() is not None
        finally:
            con.close()

    def get_biocurator_metadata(
        self,
        dbid: str,
        disease_name: str,
        qualifier: Optional[str],
        hpo_id: str,
        onset: Optional[str],
        frequency: Optional[str],
        sex: Optional[str],
        modifier: Optional[str],
    ) -> str:
        """From all identifying fields, return existing biocurator metadata."""
        if not self.hpoa_db_path or not os.path.exists(self.hpoa_db_path):
            return ""

        con = sqlite3.connect(self.hpoa_db_path)
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT biocuration
                FROM hpoa
                WHERE database_id = ?
                AND disease_name = ?
                AND COALESCE(qualifier, '') = COALESCE(?, '')
                AND hpo_id = ?
                AND COALESCE(onset, '') = COALESCE(?, '')
                AND COALESCE(frequency, '') = COALESCE(?, '')
                AND COALESCE(sex, '') = COALESCE(?, '')
                AND COALESCE(modifier, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (dbid, disease_name, qualifier, hpo_id, onset, frequency, sex, modifier),
            )
            row = cur.fetchone()
            return row[0] if row else ""
        finally:
            con.close()

    async def fetch_and_parse_hpoa(self, path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Load phenotype.hpoa into cache and refresh the SQLite database.
            If a local path is given, use that. Otherwise, always fetch the current
            stable HPOA release from the PURL.
        """
        source_path: Optional[Path] = None
        if path:
            candidate = Path(path).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"phenotype.hpoa not found at {candidate}")
            source_path = candidate
        elif self.hpoa_tsv_path and os.path.exists(self.hpoa_tsv_path):
            source_path = Path(self.hpoa_tsv_path)

        # --- local file already available ---
        if source_path:
            rows = read_hpoa_from_path(str(source_path))
            if source_path != Path(self.hpoa_tsv_path):
                shutil.copyfile(str(source_path), self.hpoa_tsv_path)
            self.persist_hpoa_to_db(rows)
            return rows

        # --- fetch from stable PURL ---
        http_client = get_client()
        download_resp = await http_client.get("https://purl.obolibrary.org/obo/hp/phenotype.hpoa")
        download_resp.raise_for_status()
        text = download_resp.text

        with open(self.hpoa_tsv_path, "w", encoding="utf-8") as fh:
            fh.write(text)

        rows = parse_hpoa_text(text)
        self.persist_hpoa_to_db(rows)
        return rows

    async def ensure_hpoa_db(self, path: Optional[str] = None) -> None:
        """Ensure the SQLite cache has rows; download or rebuild if needed."""
        if self.database_has_rows():
            with sqlite3.connect(self.hpoa_db_path) as con:
                self.ensure_indexes(con)
            return
        await self.fetch_and_parse_hpoa(path=path)

    def database_has_rows(self) -> bool:
        if not self.hpoa_db_path or not os.path.exists(self.hpoa_db_path):
            return False
        con = sqlite3.connect(self.hpoa_db_path)
        try:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hpoa'")
            if cur.fetchone() is None:
                return False
            cur.execute("SELECT 1 FROM hpoa LIMIT 1")
            return cur.fetchone() is not None
        finally:
            con.close()

    def persist_hpoa_to_db(self, rows: List[Dict[str, str]]) -> None:
        if not rows:
            raise ValueError("Cannot persist empty HPOA rows")
        con = sqlite3.connect(self.hpoa_db_path)
        try:
            con.execute("PRAGMA journal_mode = MEMORY")
            con.execute("PRAGMA synchronous = OFF")
            con.execute("PRAGMA temp_store = MEMORY")
            con.execute("PRAGMA cache_size = -64000")

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
            cur.execute("DELETE FROM hpoa")
            cur.executemany(
                "INSERT INTO hpoa (database_id, disease_name, qualifier, hpo_id, reference, evidence, onset, frequency, sex, modifier, aspect, biocuration) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
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
                ],
            )
            self.ensure_indexes(con)
            con.commit()
        finally:
            con.close()

    def ensure_indexes(self, con: sqlite3.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid ON hpoa(database_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dbid_norm ON hpoa(UPPER(REPLACE(database_id,' ','')))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_dname_nocase ON hpoa(disease_name COLLATE NOCASE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_ref_upper ON hpoa(UPPER(reference))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hpoa_hp_upper ON hpoa(UPPER(hpo_id))")


def default_cache_root() -> Path:
    """Resolve the directory used to cache phenotype.hpoa and hpoa.db."""
    cache_dir = os.environ.get("HPOA_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    return (Path.home() / ".aurelian" / "hpoa").resolve()
def get_config() -> HPOADependencies:
    """Create default dependencies, respecting any CLI-provided workdir."""
    workdir_path = os.environ.get("AURELIAN_WORKDIR")
    workdir = WorkDir(location=workdir_path) if workdir_path else WorkDir()
    return HPOADependencies(workdir=workdir)

def read_hpoa_from_path(path: str) -> List[Dict[str, str]]:
    """Read a local phenotype.hpoa TSV file preserving columns."""
    with open(path, "r", encoding="utf-8") as fh:
        return parse_hpoa_text(fh.read())

def parse_hpoa_text(text: str) -> List[Dict[str, str]]:
    """Parse a phenotype.hpoa TSV string into dictionaries."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    reader = csv.DictReader(lines, delimiter="	")
    return list(reader)