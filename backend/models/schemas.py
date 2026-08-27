"""Shared JSON contracts for MedOrchestrate.

Every agent and API route imports its data shapes from here rather than
redefining them, so the pipeline stays consistent end to end.
"""

from pydantic import BaseModel


class Patient(BaseModel):
    age: int
    sex: str


class CaseInput(BaseModel):
    case_id: str
    patient: Patient
    clinical_text: str | None = None
    medical_report: str | None = None
    image_path: str | None = None
    labs: dict = {}
    history: list[str] = []


class ClinicalFindings(BaseModel):
    demographics: dict
    symptoms: list[str]
    history: list[str]
    labs: dict
    imaging_text_findings: list[str]


class ImagingFindings(BaseModel):
    modality: str
    findings: list[str]
    confidence: float


class EvidenceItem(BaseModel):
    text: str
    source: str
    score: float


class RAGEvidence(BaseModel):
    evidence: list[EvidenceItem]


class LiveEvidenceSource(BaseModel):
    title: str
    source: str
    url: str
    summary: str
    evidence_level: str
    publication_date: str | None = None


class LiveEvidence(BaseModel):
    sources: list[LiveEvidenceSource]


class FusionResult(BaseModel):
    diagnoses: list[dict]
    overall_confidence: float
    conflicts: list[str]


class FinalReport(BaseModel):
    case_id: str
    diagnoses: list[dict]
    confidence: float
    evidence: list[dict]
    review_required: bool
    conflicts: list[str] = []
    iteration_count: int = 0
    live_evidence_available: bool = True
