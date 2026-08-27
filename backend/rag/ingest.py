"""Ingestion pipeline: chunk a small starter corpus, embed it with
BioBERT (embed_text), and upsert it into a Qdrant collection.

Corpus: CORPUS below is a small (18-document) set of SYNTHETIC reference
passages -- written for this prototype, not pulled from any live
source, and explicitly labeled source="synthetic" in their metadata
rather than attributed to a real citation. Live fetching of real
guideline/abstract text is not wired up yet; when it is, real documents
should be added to CORPUS (or a separate loader) with an honest
source field (e.g. the actual citation/URL), never mislabeled as
synthetic or vice versa.

Qdrant connection: reads QDRANT_URL / QDRANT_API_KEY from the
environment (via .env). If QDRANT_URL is set, connects to that server
-- this covers both a local docker-compose Qdrant (QDRANT_URL=
http://localhost:6333, no API key) and Qdrant Cloud (QDRANT_URL=
https://<cluster>.cloud.qdrant.io, QDRANT_API_KEY=<key>). If QDRANT_URL
is not set, falls back to an embedded, on-disk local Qdrant instance
(no server required) at data/qdrant_local, so this pipeline is runnable
out of the box for local dev/demo.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.rag.embeddings import EMBEDDING_DIM, embed_text

load_dotenv()

COLLECTION_NAME = "medorchestrate_kb"

# Word-count is used as a cheap approximation of token count for
# chunking purposes (real subword tokenization would require loading
# the model's tokenizer just to size chunks, which is unnecessary
# precision for a prototype-scoped corpus this small).
CHUNK_SIZE_TOKENS = 300
MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = 400

DEFAULT_LOCAL_QDRANT_PATH = "data/qdrant_local"

# --------------------------------------------------------------------------
# Starter corpus: small, real (not placeholder) SYNTHETIC reference
# passages covering the symptoms/differentials in our test cases
# (headache, dizziness, hypertension, and common neuro differentials).
# Each entry's "source" field is honest: "synthetic" means this text was
# authored for this prototype and is NOT a quotation of any real
# publication.
# --------------------------------------------------------------------------
CORPUS = [
    {
        "doc_id": "headache_tension_type",
        "source": "synthetic",
        "text": (
            "Tension-type headache is the most common primary headache "
            "disorder, typically described as a bilateral, pressing or "
            "tightening sensation of mild to moderate intensity, often "
            "likened to a band around the head. It is not usually "
            "aggravated by routine physical activity and lacks the "
            "nausea, vomiting, or marked photophobia seen with migraine. "
            "Episodes can be triggered by stress, poor posture, "
            "dehydration, eye strain, or sleep disruption, and can occur "
            "episodically or, less commonly, as a chronic daily pattern "
            "when present on more days than not over several months. "
            "Most cases resolve with rest, hydration, stress management, "
            "and simple analgesia such as acetaminophen or nonsteroidal "
            "anti-inflammatory drugs, though frequent use of these "
            "medications carries its own risk of medication-overuse "
            "headache if not monitored. Tension-type headache becomes a "
            "diagnosis of exclusion when new neurological symptoms, a "
            "sudden severe onset, or a change in a patient's usual "
            "headache pattern are present, since these features raise "
            "concern for a secondary cause -- such as a vascular, "
            "structural, or metabolic process -- that warrants further "
            "imaging or laboratory evaluation rather than routine "
            "symptomatic management alone. Clinicians should also ask "
            "about associated dizziness, visual change, or weakness, "
            "none of which are expected features of straightforward "
            "tension-type headache."
        ),
    },
    {
        "doc_id": "migraine_overview",
        "source": "synthetic",
        "text": (
            "Migraine is a recurrent primary headache disorder "
            "characterized by moderate to severe, often unilateral, "
            "throbbing pain lasting hours to days, commonly accompanied "
            "by nausea, vomiting, photophobia, and phonophobia, and "
            "often worsened by routine physical activity such as "
            "climbing stairs. A subset of patients experience an aura -- "
            "transient visual, sensory, or speech disturbances, most "
            "often visual zigzag lines or expanding blind spots -- "
            "preceding or accompanying the headache phase. Dizziness and "
            "vertiginous sensations are reported in a meaningful "
            "proportion of migraine patients, sometimes as vestibular "
            "migraine, where episodic vertigo occurs with or without a "
            "concurrent headache, and can be mistaken for an inner-ear "
            "disorder if the headache history is not specifically asked "
            "about. Management includes identifying and avoiding "
            "individual triggers (poor sleep, skipped meals, certain "
            "foods, hormonal changes), abortive therapy such as "
            "triptans or NSAIDs taken early in an attack, and preventive "
            "daily medication for patients with frequent or disabling "
            "attacks. Red flags that should prompt reconsideration of a "
            "migraine diagnosis -- and evaluation for a secondary cause "
            "instead -- include a first severe headache after age 50, a "
            "thunderclap onset reaching peak intensity within minutes, "
            "or new focal neurological deficits that do not resolve "
            "within the expected aura timeframe."
        ),
    },
    {
        "doc_id": "dizziness_vertigo_overview",
        "source": "synthetic",
        "text": (
            "Dizziness is a broad symptom category that clinicians "
            "generally divide into vertigo (a false sensation of "
            "spinning or motion), presyncope (a feeling of near-fainting "
            "often from transient reduced cerebral perfusion), "
            "disequilibrium (a sense of imbalance mainly during standing "
            "or walking), and nonspecific lightheadedness that does not "
            "cleanly fit the other categories. Taking a careful history "
            "of exactly what the patient means by 'dizzy', how long "
            "episodes last, and what triggers them is often more useful "
            "than any single test in narrowing the differential. Vertigo "
            "is most often peripheral in origin, arising from the inner "
            "ear or vestibular nerve, and is typically intense but "
            "self-limited or positionally triggered. Central causes "
            "originating in the brainstem or cerebellum are less common "
            "but more dangerous, and must be considered, particularly "
            "when vertigo is accompanied by headache, double vision, "
            "slurred speech, limb weakness, or gait instability, since "
            "these combinations can indicate a posterior circulation "
            "stroke rather than a benign inner-ear process. A basic "
            "neurological examination focusing on eye movements, gait, "
            "and limb coordination helps distinguish peripheral from "
            "central causes at the bedside before imaging is ordered."
        ),
    },
    {
        "doc_id": "bppv",
        "source": "synthetic",
        "text": (
            "Benign paroxysmal positional vertigo (BPPV) is the most "
            "common cause of peripheral vertigo, resulting from "
            "displaced calcium carbonate crystals (otoconia) that have "
            "migrated into one of the semicircular canals of the inner "
            "ear, most often the posterior canal. It classically "
            "presents as brief, intense episodes of spinning vertigo "
            "lasting less than a minute, triggered by specific head "
            "movements such as rolling over in bed, looking upward, or "
            "bending forward, and is often accompanied by nausea during "
            "the episode itself. Between episodes patients are typically "
            "entirely asymptomatic, which is an important distinguishing "
            "feature from conditions causing continuous dizziness. "
            "Diagnosis relies on positional testing, most commonly the "
            "Dix-Hallpike maneuver, which reproduces the characteristic "
            "torsional, upbeating nystagmus and vertigo, and treatment "
            "with canalith repositioning maneuvers such as the Epley "
            "maneuver is usually curative within one or a few sessions. "
            "BPPV does not typically cause headache, hearing loss, or "
            "tinnitus, and the presence of headache alongside positional "
            "dizziness should prompt consideration of alternative or "
            "coexisting diagnoses rather than being attributed to BPPV "
            "by default."
        ),
    },
    {
        "doc_id": "hypertension_overview",
        "source": "synthetic",
        "text": (
            "Hypertension is a chronic elevation of systemic arterial "
            "blood pressure that is usually asymptomatic until it causes "
            "end-organ damage, which is why it is often described as a "
            "silent condition detected mainly through routine screening "
            "rather than symptoms. Longstanding, poorly controlled "
            "hypertension is a major risk factor for stroke, myocardial "
            "infarction, heart failure, chronic kidney disease, and "
            "cerebral small vessel disease, and its cumulative vascular "
            "damage develops silently over years to decades. Occasional "
            "symptoms popularly attributed to hypertension -- headache, "
            "dizziness, or lightheadedness -- more often reflect very "
            "high or rapidly changing blood pressure, medication "
            "effects, or an unrelated coexisting cause, rather than "
            "mild-to-moderate chronic hypertension itself, and clinicians "
            "should avoid anchoring on hypertension as the automatic "
            "explanation for nonspecific symptoms without further "
            "assessment. Management centers on lifestyle modification -- "
            "dietary sodium reduction, weight management, regular "
            "physical activity, and limiting alcohol -- combined with "
            "antihypertensive medication individualized to the patient's "
            "overall cardiovascular risk profile, comorbidities, and "
            "tolerance of side effects."
        ),
    },
    {
        "doc_id": "hypertensive_emergency",
        "source": "synthetic",
        "text": (
            "A hypertensive emergency is a severe elevation in blood "
            "pressure, generally above 180/120 mmHg, accompanied by "
            "evidence of acute end-organ damage -- including "
            "hypertensive encephalopathy, which can present with severe "
            "headache, altered mental status, visual disturbance, "
            "seizures, and focal neurological signs due to cerebral "
            "edema and impaired autoregulation of cerebral blood flow. "
            "Other end-organ manifestations include acute pulmonary "
            "edema, acute kidney injury, and aortic dissection, meaning "
            "the evaluation of a hypertensive emergency extends well "
            "beyond the neurological exam alone. This differs "
            "importantly from asymptomatic severe hypertension (often "
            "called hypertensive urgency), which does not require the "
            "same urgent, closely monitored intravenous blood pressure "
            "reduction and can typically be managed with oral "
            "medication adjustment in an outpatient or observation "
            "setting. A patient with a known history of hypertension who "
            "presents with new severe headache and dizziness should have "
            "their blood pressure measured promptly as a first step, "
            "since a hypertensive emergency is an important and "
            "time-sensitive differential in this presentation that "
            "changes the urgency of the entire subsequent workup."
        ),
    },
    {
        "doc_id": "orthostatic_hypotension",
        "source": "synthetic",
        "text": (
            "Orthostatic hypotension is a sustained drop in blood "
            "pressure -- conventionally at least 20 mmHg systolic or 10 "
            "mmHg diastolic -- occurring within a few minutes of "
            "standing from a sitting or lying position, and is a common "
            "cause of presyncope-type dizziness, lightheadedness, and "
            "occasional falls, particularly in older adults with reduced "
            "baroreflex sensitivity. It can result from dehydration, "
            "autonomic dysfunction (including diabetic or Parkinsonian "
            "autonomic neuropathy), prolonged bed rest, or medications, "
            "with antihypertensive drugs being a particularly common "
            "culprit since they can overshoot the normal compensatory "
            "blood pressure response on standing. Symptoms are typically "
            "worst immediately after standing and improve within a "
            "minute or two as the body compensates, which is a useful "
            "clue distinguishing it from other causes of dizziness. "
            "Because it is a common, readily testable cause of dizziness "
            "in hypertensive patients on treatment, orthostatic vital "
            "signs -- blood pressure and heart rate measured lying and "
            "then standing -- are a simple, low-cost step worth checking "
            "early in the workup of a hypertensive patient reporting "
            "dizziness, before proceeding to more resource-intensive "
            "testing."
        ),
    },
    {
        "doc_id": "stroke_tia_red_flags",
        "source": "synthetic",
        "text": (
            "Stroke and transient ischemic attack (TIA) should be "
            "considered whenever headache or dizziness occurs alongside "
            "sudden focal neurological symptoms: unilateral weakness or "
            "numbness, facial droop, slurred or garbled speech, sudden "
            "vision loss or double vision, severe imbalance, or a "
            "markedly abnormal gait that is new for the patient. The "
            "FAST mnemonic (Face, Arms, Speech, Time) is commonly used "
            "for rapid public and clinical recognition of these warning "
            "signs. A sudden, severe 'thunderclap' headache reaching "
            "maximum intensity within seconds to minutes -- often "
            "described by patients as the worst headache of their life "
            "-- is a classic warning sign of subarachnoid hemorrhage and "
            "requires urgent imaging, typically a non-contrast CT head "
            "as the first-line test. Hypertension is among the strongest "
            "modifiable risk factors for both ischemic and hemorrhagic "
            "stroke, alongside atrial fibrillation, diabetes, smoking, "
            "and hyperlipidemia, so a hypertensive patient presenting "
            "with new headache and dizziness warrants a lower threshold "
            "for neuroimaging than a normotensive patient with the same "
            "complaint, especially if any focal deficit is present on "
            "examination."
        ),
    },
    {
        "doc_id": "white_matter_hyperintensities",
        "source": "synthetic",
        "text": (
            "White matter hyperintensities are areas of increased signal "
            "on T2-weighted and FLAIR brain MRI sequences, most commonly "
            "reflecting chronic cerebral small vessel disease affecting "
            "the deep and periventricular white matter. They are "
            "strongly associated with age and with vascular risk "
            "factors, chief among them longstanding hypertension, which "
            "accelerates small vessel injury and impairs normal "
            "cerebral blood flow autoregulation over time. Radiologists "
            "often grade their extent using scales such as the Fazekas "
            "scale, ranging from mild punctate foci to large, confluent "
            "areas of signal change. While scattered small white matter "
            "hyperintensities are a frequent, often incidental finding "
            "on brain MRI in middle-aged and older adults and do not by "
            "themselves indicate an acute problem, an extensive or "
            "rapidly progressive burden is associated with an increased "
            "risk of stroke, cognitive decline, and gait disturbance, "
            "and should prompt attention to vascular risk factor "
            "control, particularly blood pressure management, smoking "
            "cessation, and glycemic control where relevant."
        ),
    },
    {
        "doc_id": "vestibular_neuritis_labyrinthitis",
        "source": "synthetic",
        "text": (
            "Vestibular neuritis and labyrinthitis are inner-ear "
            "conditions, usually presumed post-viral in origin, that "
            "cause acute, continuous vertigo lasting days, often "
            "accompanied by nausea, vomiting, and unsteady gait severe "
            "enough to make walking difficult during the acute phase. "
            "Labyrinthitis additionally involves the cochlea and so can "
            "cause hearing loss or tinnitus, which is not typical of "
            "vestibular neuritis alone since that condition spares the "
            "cochlear branch of the eighth cranial nerve. Unlike BPPV, "
            "symptoms are constant rather than triggered by specific "
            "head positions, though head movement does worsen the "
            "underlying vertigo, and unlike a central (stroke-related) "
            "cause, patients generally lack other brainstem signs such "
            "as double vision, dysarthria, or limb weakness -- bedside "
            "eye-movement testing (the HINTS exam) is often used by "
            "specialists to help distinguish the two. Headache is not a "
            "prominent feature of either condition; when significant "
            "headache accompanies acute vertigo, a central cause should "
            "be more strongly considered and evaluated accordingly "
            "rather than assuming a peripheral vestibular process."
        ),
    },
    {
        "doc_id": "menieres_disease",
        "source": "synthetic",
        "text": (
            "Meniere's disease is an inner-ear disorder attributed to "
            "excess fluid (endolymphatic hydrops) within the labyrinth, "
            "producing recurrent episodes of vertigo typically lasting "
            "20 minutes to several hours, accompanied by fluctuating "
            "hearing loss, tinnitus, and a sensation of aural fullness "
            "or pressure in the affected ear. The classic tetrad of "
            "vertigo, fluctuating hearing loss, tinnitus, and aural "
            "fullness helps distinguish it from other vestibular "
            "disorders, though not all four features are always present "
            "at diagnosis, particularly early in the disease course. "
            "Episodes can be disabling and are often followed by "
            "prolonged fatigue and residual imbalance lasting hours to "
            "days afterward. Diagnosis is largely clinical, supported by "
            "audiometry showing low-frequency sensorineural hearing "
            "loss that may fluctuate with symptom flares over time. "
            "Headache is not a defining feature of Meniere's disease, "
            "and its consistent presence alongside vertigo should "
            "prompt consideration of migraine-associated vertigo or "
            "another concurrent process rather than being attributed to "
            "Meniere's disease alone, since misattribution can delay "
            "appropriate treatment."
        ),
    },
    {
        "doc_id": "secondary_headache_red_flags",
        "source": "synthetic",
        "text": (
            "Certain headache features -- often summarized with "
            "mnemonics like SNNOOP10 -- should prompt evaluation for a "
            "secondary cause rather than a primary headache disorder: "
            "systemic symptoms such as fever or weight loss, a "
            "history of Neoplasm, a new headache in an Older patient "
            "(generally over 50), progressively worsening Pattern, "
            "sudden 'thunderclap' onset, positional worsening, "
            "headache triggered by exertion, cough, or Valsalva, "
            "Papilledema on exam, and new neurological deficits "
            "including altered consciousness, seizure, or focal "
            "weakness. Each of these features shifts the pretest "
            "probability toward a structural, vascular, infectious, or "
            "inflammatory cause that primary headache disorders like "
            "migraine or tension-type headache do not explain. A "
            "patient presenting for the first time with headache and "
            "dizziness in mid-to-late adulthood, especially with a "
            "background of hypertension or other vascular risk factors, "
            "sits in a population where these red flags should be "
            "actively screened for before defaulting to a benign "
            "primary headache diagnosis, since the consequences of "
            "missing a secondary cause in this group can be severe."
        ),
    },
    {
        "doc_id": "anemia_dizziness",
        "source": "synthetic",
        "text": (
            "Anemia reduces the blood's oxygen-carrying capacity and "
            "commonly presents with fatigue, pallor, exertional "
            "dyspnea, palpitations, and dizziness or lightheadedness, "
            "particularly on standing or exertion, due to reduced "
            "cerebral oxygen delivery and compensatory changes in "
            "cardiac output. Headache can also occur, especially with "
            "more significant or rapidly developing anemia, and some "
            "patients additionally report tinnitus or a 'whooshing' "
            "sound related to increased cardiac flow. Common causes "
            "include iron deficiency, chronic disease, vitamin B12 or "
            "folate deficiency, and occult gastrointestinal blood loss, "
            "each of which points to a different subsequent workup. A "
            "basic complete blood count, including hemoglobin, is a "
            "simple and inexpensive test that should be considered "
            "early in the workup of unexplained dizziness, since anemia "
            "is a common, readily identifiable, and generally treatable "
            "contributor that is easy to miss if clinical attention "
            "focuses only on neurological or cardiovascular causes and "
            "skips basic laboratory screening."
        ),
    },
    {
        "doc_id": "antihypertensive_medication_dizziness",
        "source": "synthetic",
        "text": (
            "Antihypertensive medications -- including diuretics, "
            "alpha-blockers, beta-blockers, and some vasodilators -- can "
            "cause dizziness or lightheadedness, most often through "
            "excessive blood pressure lowering or orthostatic "
            "hypotension, particularly soon after starting a new agent "
            "or a dose increase, and especially in older patients on "
            "multiple agents where effects can compound. Diuretics can "
            "additionally contribute to dizziness through volume "
            "depletion or electrolyte disturbances such as hyponatremia, "
            "which independently cause lethargy and unsteadiness. This "
            "is an important consideration in any hypertensive patient "
            "reporting new dizziness: a recent medication change, dose "
            "titration, or drug interaction (including with other "
            "prescribed or over-the-counter medications) should be "
            "reviewed alongside other causes, since medication-induced "
            "dizziness is common, usually reversible with dose "
            "adjustment or a medication switch, and easy to overlook if "
            "clinical focus goes straight to structural or neurological "
            "explanations without first reviewing the medication list."
        ),
    },
    {
        "doc_id": "chronic_subdural_hematoma",
        "source": "synthetic",
        "text": (
            "Chronic subdural hematoma is a collection of old blood "
            "between the dura and the brain surface that develops "
            "gradually, often after a minor or even unrecalled head "
            "injury weeks earlier, and disproportionately affects older "
            "adults -- due to age-related brain atrophy that stretches "
            "bridging veins -- and patients on anticoagulant or "
            "antiplatelet therapy, who are at higher risk of both "
            "developing and re-accumulating the collection. Presentation "
            "is often subtle and nonspecific -- headache, dizziness, "
            "mild cognitive changes, personality change, or gait "
            "unsteadiness -- developing over days to weeks rather than "
            "suddenly, which can make it easy to mistake for a benign or "
            "age-related cause, dementia, or simple deconditioning. It "
            "is an important differential in an older or hypertensive "
            "patient with persistent headache and dizziness, since "
            "brain imaging (CT or MRI) is required for diagnosis and the "
            "condition is generally treatable, ranging from observation "
            "for small collections to surgical drainage for larger or "
            "symptomatic ones, if identified before significant mass "
            "effect develops."
        ),
    },
    {
        "doc_id": "brain_mri_findings_overview",
        "source": "synthetic",
        "text": (
            "Brain MRI is a key tool for evaluating patients with "
            "headache and dizziness when a structural or vascular cause "
            "is suspected, offering better soft-tissue and posterior "
            "fossa resolution than CT for many of these conditions. "
            "Findings that clinicians and radiologists look for include "
            "mass lesions (space-occupying tumors, abscesses, or "
            "hematomas), midline shift indicating raised intracranial "
            "pressure or significant mass effect, ventricular "
            "enlargement suggestive of hydrocephalus or age-related "
            "atrophy, acute or chronic infarcts reflecting ischemic "
            "stroke of differing ages, hemorrhage of varying location "
            "and cause, sinus abnormality incidentally noted on brain "
            "sequences, and white matter hyperintensities associated "
            "with chronic small vessel disease. Findings are always "
            "interpreted alongside the clinical history, since the same "
            "imaging appearance can carry very different significance "
            "in a young healthy patient versus an older hypertensive "
            "one. Many brain MRIs in patients with nonspecific symptoms "
            "show no significant abnormality, which is itself a "
            "clinically useful and reassuring finding rather than an "
            "inconclusive one, and helps redirect the workup toward "
            "non-structural causes."
        ),
    },
    {
        "doc_id": "sinusitis_headache",
        "source": "synthetic",
        "text": (
            "Sinusitis-related headache arises from inflammation and "
            "pressure within the paranasal sinuses, typically presenting "
            "as facial pain or pressure over the affected sinus (frontal, "
            "maxillary, ethmoid, or sphenoid), worsened by bending "
            "forward or changes in position, alongside nasal congestion, "
            "purulent nasal discharge, or reduced sense of smell, often "
            "in the setting of an upper respiratory infection lasting "
            "more than seven to ten days. True sinusitis-related "
            "headache is far less common than patients and even some "
            "clinicians sometimes assume; many headaches attributed to "
            "'sinus' symptoms are in fact migraine or tension-type "
            "headache with overlapping features such as facial pressure "
            "or nasal congestion during an attack, leading to frequent "
            "misdiagnosis and unnecessary antibiotic use. Dizziness is "
            "not a typical feature of sinusitis, and its presence "
            "alongside headache should point toward an alternative or "
            "additional cause rather than being folded into a sinus "
            "headache diagnosis without further evaluation."
        ),
    },
    {
        "doc_id": "cerebral_small_vessel_disease",
        "source": "synthetic",
        "text": (
            "Cerebral small vessel disease refers to a group of "
            "pathological processes affecting the small arteries, "
            "arterioles, capillaries, and venules of the brain, most "
            "commonly driven by chronic hypertension and aging, with "
            "diabetes and smoking as additional contributing risk "
            "factors. It manifests on imaging as white matter "
            "hyperintensities, lacunar infarcts, cerebral microbleeds, "
            "and enlarged perivascular spaces, which together form a "
            "recognizable radiological pattern even when individually "
            "nonspecific. Clinically it can contribute to gait "
            "disturbance, cognitive decline (including vascular "
            "dementia), mood changes such as apathy or depression, and "
            "an increased risk of both ischemic stroke and intracerebral "
            "hemorrhage, since the same fragile vessels are prone to "
            "both blockage and rupture. Because hypertension is the "
            "single most important modifiable risk factor, sustained "
            "blood pressure control is central to slowing progression "
            "of the disease, making it a key consideration when a "
            "hypertensive patient's brain MRI shows these changes, and "
            "an opportunity to reinforce the importance of adherence to "
            "antihypertensive treatment."
        ),
    },
]


def _approx_token_count(text: str) -> int:
    """Cheap word-count approximation of token count for chunk sizing."""
    return len(text.split())


def chunk_document(doc: dict, chunk_size_tokens: int = CHUNK_SIZE_TOKENS) -> list[dict]:
    """Split one corpus document into ~200-400 token passages.

    Splits on whitespace-delimited words into windows of
    `chunk_size_tokens`. For documents already within the ~200-400
    token band (true for most of this small starter corpus) this
    produces a single chunk; longer documents are split into multiple
    sequential, non-overlapping chunks with incrementing `page` numbers.
    """
    words = doc["text"].split()
    chunks = []
    for start in range(0, len(words), chunk_size_tokens):
        chunk_words = words[start : start + chunk_size_tokens]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "source": doc["source"],
                "doc_id": doc["doc_id"],
                "page": len(chunks) + 1,
                "chunk_length": len(chunk_words),
                "text": chunk_text,
            }
        )
    return chunks


def get_qdrant_client() -> QdrantClient:
    """Build a Qdrant client from QDRANT_URL / QDRANT_API_KEY, with a
    local on-disk fallback when no URL is configured (see module docstring)."""
    qdrant_url = os.getenv("QDRANT_URL") or None
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return QdrantClient(path=DEFAULT_LOCAL_QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def build_chunks() -> list[dict]:
    chunks = []
    for doc in CORPUS:
        chunks.extend(chunk_document(doc))
    return chunks


def ingest(client: QdrantClient | None = None) -> int:
    """Chunk CORPUS, embed each chunk, and upsert into Qdrant.

    Returns the number of chunks upserted.
    """
    client = client or get_qdrant_client()
    ensure_collection(client)

    chunks = build_chunks()
    points = [
        PointStruct(id=i, vector=embed_text(chunk["text"]), payload=chunk)
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks from {len(CORPUS)} documents into '{COLLECTION_NAME}'.")
