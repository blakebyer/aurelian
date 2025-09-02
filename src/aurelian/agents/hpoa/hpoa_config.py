""" Configuration file for HPOA Agent """
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dataclasses import dataclass, field
import os, csv, httpx, subprocess
from typing import Optional, List, Dict, Any
from oaklib import get_adapter
from oaklib.interfaces import BasicOntologyInterface

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
    biocuration: str = Field(..., description="""This refers to the biocurator who made the annotation and the date on which the annotation was made; the date format is YYYY-MM-DD. The first entry in this field refers to the creation date. Any additional biocuration is recorded following a semicolon. So, if Joseph curated on July 5, 2012, and Suzanna curated on December 7, 2015, one might have a field like this: HPO:Joseph[2012-07-05];HPO:Suzanna[2015-12-07]. It is acceptable to use ORCID ids.""")

class HPOAResult(BaseModel):
    status: Literal["existing", "new"] = Field(
        ..., description="Whether this annotation was already present in phenotype.hpoa or is newly suggested."
    )
    rationale: Optional[str] = None
    annotation: HPOA

class HPOAResponse(BaseModel):
    explanation: str = Field(..., description="A brief natural language explanation of what was found and done.")
    annotations: List[HPOAResult] = Field(default_factory=list)

@dataclass
class HPOADependencies(HasWorkdir):
    """Configuration for the HPOA agent."""
    openai_api_key: Optional[str] = None
    omim_api_key: Optional[str] = None
    
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

    def get_mondo_adapter(self) -> BasicOntologyInterface:
        """Get a configured Mondo adapter."""
        return get_adapter("sqlite:obo:mondo")
        #return get_adapter("ontobee:mondo")
    
    def get_hp_adapter(self) -> BasicOntologyInterface:
        """Get a configured HPO adapter."""
        return get_adapter("sqlite:obo:hp")
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
            return _read_hpoa_from_path(candidate_path)

        # 2) Environment variable
        env_path = os.environ.get("AURELIAN_HPOA_PATH")
        if env_path and os.path.exists(env_path):
            return _read_hpoa_from_path(env_path)

        # 3) Cached copy in workdir
        cached_path = None
        if self.workdir and self.workdir.location:
            cached_path = os.path.join(self.workdir.location, "phenotype.hpoa")
            if os.path.exists(cached_path):
                return _read_hpoa_from_path(cached_path)

        # 4) Download latest and cache
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
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
        reader = csv.DictReader(lines, delimiter="\t")
        return list(reader)
    
    
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
