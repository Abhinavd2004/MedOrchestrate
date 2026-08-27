"""Candidate label sets for BiomedCLIP zero-shot classification.

Both lists are plain Python lists of short clinical phrases used as the
text side of zero-shot classification (image embedding vs. each label's
text embedding). They are intentionally small and scoped to this
prototype's demo cases — not a comprehensive radiology ontology.

To extend either list: add another short, clinically phrased string.
Prefer phrases in the same style/length as the existing entries, since
BiomedCLIP's text encoder was trained on short biomedical captions —
long or multi-clause phrases embed less reliably.
"""

# Candidate imaging modalities. Extend this list to add support for a
# new modality (e.g. "MRI spine scan", "abdominal ultrasound") — the
# imaging agent classifies against whichever labels are here, no code
# changes needed elsewhere.
MODALITY_LABELS = [
    "MRI brain scan",
    "CT scan",
    "chest X-ray",
    "unspecified medical image",
]

# Candidate findings, scoped to the neuro / headache-dizziness demo
# cases this prototype targets for the Sept 1 demo. This is NOT a
# comprehensive radiology ontology — it is a small, hand-picked label
# set for zero-shot classification against BiomedCLIP. Expanding this
# to cover more modalities/pathologies is future work.
FINDING_LABELS = [
    "normal brain MRI, no acute findings",
    "white matter hyperintensity",
    "mass lesion",
    "midline shift",
    "ventricular enlargement",
    "acute infarct",
    "chronic infarct",
    "hemorrhage",
    "sinus abnormality",
    "cerebral atrophy",
    "edema",
    "vascular malformation",
    "no significant abnormality",
]
