"""Synthetic clinical case layer used by document renderers.

Synthea should feed the clinical story when available. This module provides the
same shape for built-in synthetic cases, so renderers do not depend on where the
clinical content came from.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from meddeid_language_nl.date_age_variants import (
    age_text_variant as shared_age_text_variant,
)
from meddeid_language_nl.date_age_variants import (
    birthdate_days_before,
    birthdate_for_age,
    birthdate_months_before,
    birthdate_weeks_before,
    date_context_prefix,
    format_date_variant,
    format_named_date_profile,
    sample_encounter_date,
)

from .identifiers import (
    caregiver_id,
    email,
    hard_negative_codes,
    his_patient_id,
    internal_phone,
    material_lot_number,
    national_register,
    patient_identifier_bundle,
    patient_number,
    phone,
    study_protocol_identifier,
    study_protocol_name,
)
from .lookups import LookupSampler, full_name

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources" / "clinical"


def _require_resource(filename: str) -> Path:
    path = RESOURCE_ROOT / filename
    if not path.exists():
        raise RuntimeError(f"missing packaged clinical resource: {path}")
    return path


def _load_jsonl_resource(filename: str) -> list[dict[str, Any]]:
    path = _require_resource(filename)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = list(handle)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read packaged clinical resource: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON in packaged clinical resource {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(
                f"expected an object in packaged clinical resource {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"packaged clinical resource is empty: {path}")
    return rows


def _load_tsv_resource(filename: str) -> list[dict[str, str]]:
    path = _require_resource(filename)
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = list(handle)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read packaged clinical resource: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t", 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"invalid TSV in packaged clinical resource {path}:{line_number}"
            )
        cnk, name = (" ".join(part.split()) for part in parts)
        if not cnk.isdigit() or not name:
            raise RuntimeError(
                f"invalid medicine in packaged clinical resource {path}:{line_number}"
            )
        rows.append({"cnk": cnk, "name": name, "source": str(path)})
    if not rows:
        raise RuntimeError(f"packaged clinical resource is empty: {path}")
    return rows


def _load_eponym_resource(filename: str) -> list[dict[str, str]]:
    path = _require_resource(filename)
    rows: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read packaged clinical resource: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t", 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"invalid TSV in packaged clinical resource {path}:{line_number}"
            )
        slug, title = (" ".join(part.split()) for part in parts)
        if not slug or not title:
            raise RuntimeError(
                f"invalid eponym in packaged clinical resource {path}:{line_number}"
            )
        rows.append({"slug": slug, "title": title, "source": str(path)})
    if not rows:
        raise RuntimeError(f"packaged clinical resource is empty: {path}")
    return rows


def _condition_record(
    name: str,
    departments: list[str],
    symptoms: list[str],
    medications: list[str],
    labs: list[tuple[str, str, str]],
) -> dict:
    return {
        "name": name,
        "departments": departments,
        "symptoms": symptoms,
        "medications": medications,
        "labs": labs,
    }


def _genetic_finding(
    gene: str,
    variant: str,
    protein: str,
    rsid: str,
    interpretation: str,
) -> dict:
    return {
        "gene": gene,
        "variant": variant,
        "protein": protein,
        "rsid": rsid,
        "interpretation": interpretation,
    }


DEPARTMENTS = [
    "cardiologie",
    "oncologie",
    "spoedgevallen",
    "geriatrie",
    "pneumologie",
    "endocrinologie",
    "neurologie",
    "psychiatrie",
    "nefrologie",
    "genetica",
    "revalidatie",
    "infectiologie",
    "huisartsgeneeskunde",
    "pathologie",
    "radiologie",
    "gastro-enterologie",
    "hepatologie",
    "hematologie",
    "dermatologie",
    "reumatologie",
    "orthopedie",
    "traumatologie",
    "urologie",
    "gynaecologie",
    "verloskunde",
    "pediatrie",
    "neonatologie",
    "intensieve zorgen",
    "anesthesiologie",
    "pijnkliniek",
    "oftalmologie",
    "neus-keel-oor",
    "handchirurgie",
    "audiologie",
    "vaatchirurgie",
    "algemene chirurgie",
    "thoraxchirurgie",
    "plastische chirurgie",
    "neurochirurgie",
    "hartchirurgie",
    "kinderchirurgie",
    "geriatrisch dagziekenhuis",
    "palliatieve zorg",
    "thuiszorg",
    "wondzorgkliniek",
    "diabetologie",
    "obesitaskliniek",
    "slaapkliniek",
    "fertiliteitscentrum",
    "menopauzekliniek",
    "borstkliniek",
    "dagoncologie",
    "radiotherapie",
    "nucleaire geneeskunde",
    "klinische biologie",
    "microbiologie",
    "immunologie",
    "allergologie",
    "arbeidsgeneeskunde",
    "sportgeneeskunde",
    "reisgeneeskunde",
    "tuberculosekliniek",
    "hepatobiliaire chirurgie",
    "transplantatiegeneeskunde",
    "dialysecentrum",
    "hartfalenkliniek",
    "pacemakerkliniek",
    "stroke unit",
    "geheugenkliniek",
    "epilepsiecentrum",
    "MS-kliniek",
    "pijnrevalidatie",
    "locomotorische revalidatie",
    "cardiale revalidatie",
    "respiratoire revalidatie",
    "psychiatrisch dagziekenhuis",
    "crisisinterventie",
    "verslavingszorg",
    "liaisonpsychiatrie",
    "kinderpsychiatrie",
    "psychosomatiek",
    "klinische psychologie",
    "logopedie",
    "ergotherapie",
    "kinesitherapie",
    "dietetiek",
    "sociale dienst",
    "geriatrisch supportteam",
    "valkliniek",
    "osteoporosekliniek",
    "bekkenbodemkliniek",
    "continentiekliniek",
    "stomazorg",
    "endoscopie",
    "functieonderzoek",
    "prenatale diagnostiek",
    "medische beeldvorming",
    "nucleaire cardiologie",
    "urgentie observatie",
    "dagziekenhuis algemeen",
]

_CONDITION_ROWS = [
    (
        "acuut coronair syndroom",
        ["cardiologie", "spoedgevallen"],
        ["drukkende thoracale pijn", "dyspneu bij inspanning", "nausea"],
        ["acetylsalicylzuur", "atorvastatine", "bisoprolol"],
        [("troponine T", "84", "ng/L"), ("CK-MB", "11.2", "ug/L")],
    ),
    (
        "type 2 diabetes mellitus",
        ["endocrinologie", "huisartsgeneeskunde"],
        ["polyurie", "vermoeidheid", "wazig zicht"],
        ["metformine", "insuline glargine", "empagliflozine"],
        [("HbA1c", "8.4", "%"), ("glucose nuchter", "168", "mg/dL")],
    ),
    (
        "COPD exacerbatie",
        ["pneumologie", "spoedgevallen"],
        ["toegenomen dyspneu", "productieve hoest", "piepen"],
        ["salbutamol", "prednisolone", "amoxicilline/clavulaanzuur"],
        [("CRP", "62", "mg/L"), ("pCO2", "48", "mmHg")],
    ),
    (
        "mammacarcinoom",
        ["oncologie", "pathologie"],
        ["vermoeidheid", "neuropathie", "misselijkheid na kuur"],
        ["paclitaxel", "ondansetron", "pegfilgrastim"],
        [("CA 15-3", "48", "U/mL"), ("neutrofielen", "1.1", "10^9/L")],
    ),
    (
        "chronische nierinsufficientie",
        ["nefrologie", "geriatrie"],
        ["enkeloedeem", "jeuk", "nachtelijke krampen"],
        ["furosemide", "calciumcarbonaat", "epoetine alfa"],
        [("creatinine", "2.1", "mg/dL"), ("eGFR", "31", "mL/min/1.73m2")],
    ),
    (
        "epilepsie",
        ["neurologie", "spoedgevallen"],
        ["tonisch-clonisch insult", "postictale verwardheid", "tongbeet"],
        ["levetiracetam", "lamotrigine", "diazepam"],
        [("natrium", "139", "mmol/L"), ("levetiracetam spiegel", "18", "mg/L")],
    ),
    (
        "depressieve episode",
        ["psychiatrie", "huisartsgeneeskunde"],
        ["slaapstoornis", "anhedonie", "concentratieproblemen"],
        ["sertraline", "trazodon", "vitamine D"],
        [("TSH", "1.7", "mU/L"), ("vitamine B12", "312", "ng/L")],
    ),
    (
        "familiaire hypercholesterolemie",
        ["genetica", "cardiologie"],
        ["xanthelasmata", "familiale belasting", "geen thoracale pijn"],
        ["rosuvastatine", "ezetimibe", "evolocumab"],
        [("LDL cholesterol", "212", "mg/dL"), ("ApoB", "143", "mg/dL")],
    ),
    (
        "hartfalen decompensatie",
        ["cardiologie", "hartfalenkliniek"],
        ["orthopneu", "enkeloedeem", "snelle gewichtstoename"],
        ["furosemide", "sacubitril/valsartan", "spironolacton"],
        [("NT-proBNP", "2460", "ng/L"), ("kalium", "4.8", "mmol/L")],
    ),
    (
        "atriumfibrilleren",
        ["cardiologie", "spoedgevallen"],
        ["palpitaties", "duizeligheid", "kortademigheid"],
        ["apixaban", "metoprolol", "digoxine"],
        [("TSH", "2.3", "mU/L"), ("magnesium", "1.9", "mg/dL")],
    ),
    (
        "arteriele hypertensie",
        ["cardiologie", "huisartsgeneeskunde"],
        ["hoofdpijn", "oorsuizen", "druk op de borst"],
        ["amlodipine", "perindopril", "hydrochloorthiazide"],
        [("creatinine", "0.9", "mg/dL"), ("kalium", "4.1", "mmol/L")],
    ),
    (
        "diep veneuze trombose",
        ["vaatchirurgie", "spoedgevallen"],
        ["eenzijdige kuitzwelling", "drukpijn", "warm been"],
        ["rivaroxaban", "paracetamol", "compressietherapie"],
        [("D-dimeren", "1860", "ng/mL"), ("bloedplaatjes", "241", "10^9/L")],
    ),
    (
        "longembolie",
        ["pneumologie", "spoedgevallen"],
        ["plotse dyspneu", "pleuritische pijn", "tachycardie"],
        ["heparine", "apixaban", "zuurstof"],
        [("D-dimeren", "2440", "ng/mL"), ("troponine T", "19", "ng/L")],
    ),
    (
        "community-acquired pneumonie",
        ["pneumologie", "infectiologie"],
        ["koorts", "productieve hoest", "thoracale pijn"],
        ["amoxicilline", "claritromycine", "paracetamol"],
        [("CRP", "118", "mg/L"), ("leukocyten", "13.6", "10^9/L")],
    ),
    (
        "astma exacerbatie",
        ["pneumologie", "huisartsgeneeskunde"],
        ["piepende ademhaling", "nachtelijke hoest", "benauwdheid"],
        ["salbutamol", "budesonide/formoterol", "prednisolone"],
        [("eosinofielen", "0.62", "10^9/L"), ("FeNO", "48", "ppb")],
    ),
    (
        "slaapapneu",
        ["slaapkliniek", "pneumologie"],
        ["snurken", "dagmoeheid", "ochtendhoofdpijn"],
        ["CPAP", "mometason neusspray", "gewichtsreductie"],
        [("AHI", "31", "/uur"), ("zuurstofsaturatie minimum", "82", "%")],
    ),
    (
        "interstitiele longziekte",
        ["pneumologie", "radiologie"],
        ["progressieve dyspneu", "droge hoest", "inspanningsbeperking"],
        ["nintedanib", "omeprazol", "zuurstof"],
        [("FVC", "68", "% voorspeld"), ("DLCO", "51", "% voorspeld")],
    ),
    (
        "pulmonale hypertensie",
        ["pneumologie", "cardiologie"],
        ["syncope", "dyspneu", "druk op de borst"],
        ["sildenafil", "furosemide", "macitentan"],
        [("NT-proBNP", "980", "ng/L"), ("urinezuur", "7.8", "mg/dL")],
    ),
    (
        "COVID-19 pneumonie",
        ["infectiologie", "pneumologie"],
        ["koorts", "droge hoest", "hypoxemie"],
        ["dexamethason", "remdesivir", "enoxaparine"],
        [("CRP", "94", "mg/L"), ("lymfocyten", "0.7", "10^9/L")],
    ),
    (
        "tuberculose screening",
        ["tuberculosekliniek", "infectiologie"],
        ["nachtzweten", "gewichtsverlies", "chronische hoest"],
        ["isoniazide", "rifampicine", "pyridoxine"],
        [("IGRA", "positief", ""), ("CRP", "22", "mg/L")],
    ),
    (
        "inflammatoire darmziekte",
        ["gastro-enterologie", "immunologie"],
        ["buikpijn", "diarree", "gewichtsverlies"],
        ["mesalazine", "azathioprine", "adalimumab"],
        [("calprotectine", "780", "ug/g"), ("CRP", "34", "mg/L")],
    ),
    (
        "ziekte van Crohn opstoot",
        ["gastro-enterologie", "endoscopie"],
        ["rechteronderbuikpijn", "diarree", "subfebrilitas"],
        ["prednisolone", "ustekinumab", "vitamine B12"],
        [("calprotectine", "1120", "ug/g"), ("albumine", "3.3", "g/dL")],
    ),
    (
        "colitis ulcerosa opstoot",
        ["gastro-enterologie", "endoscopie"],
        ["bloederige diarree", "tenesmen", "krampen"],
        ["mesalazine", "methylprednisolone", "vedolizumab"],
        [("hemoglobine", "10.8", "g/dL"), ("CRP", "41", "mg/L")],
    ),
    (
        "levercirrose",
        ["hepatologie", "gastro-enterologie"],
        ["ascites", "spider naevi", "verwardheid"],
        ["spironolacton", "lactulose", "propranolol"],
        [("INR", "1.6", ""), ("bilirubine", "2.8", "mg/dL")],
    ),
    (
        "galsteenlijden",
        ["algemene chirurgie", "gastro-enterologie"],
        ["koliekpijn", "misselijkheid", "pijn rechterbovenbuik"],
        ["diclofenac", "butylscopolamine", "paracetamol"],
        [("gamma-GT", "146", "U/L"), ("ALAT", "86", "U/L")],
    ),
    (
        "acute pancreatitis",
        ["gastro-enterologie", "spoedgevallen"],
        ["epigastrische pijn", "braken", "uitstraling naar rug"],
        ["piritramide", "ondansetron", "Ringer lactaat"],
        [("lipase", "1260", "U/L"), ("CRP", "58", "mg/L")],
    ),
    (
        "gastro-oesofageale reflux",
        ["gastro-enterologie", "huisartsgeneeskunde"],
        ["pyrosis", "zure oprispingen", "nachtelijke hoest"],
        ["pantoprazol", "alginaat", "domperidon"],
        [("hemoglobine", "13.9", "g/dL"), ("ferritine", "42", "ug/L")],
    ),
    (
        "maagzweer",
        ["gastro-enterologie", "endoscopie"],
        ["epigastrische pijn", "melena", "nausea"],
        ["pantoprazol", "amoxicilline", "claritromycine"],
        [("hemoglobine", "9.8", "g/dL"), ("ureum", "54", "mg/dL")],
    ),
    (
        "diverticulitis",
        ["algemene chirurgie", "gastro-enterologie"],
        ["linkeronderbuikpijn", "koorts", "obstipatie"],
        ["amoxicilline/clavulaanzuur", "paracetamol", "polyethyleenglycol"],
        [("CRP", "136", "mg/L"), ("leukocyten", "14.2", "10^9/L")],
    ),
    (
        "prikkelbare darm syndroom",
        ["gastro-enterologie", "huisartsgeneeskunde"],
        ["wisselende stoelgang", "opgeblazen gevoel", "buikkrampen"],
        ["mebeverine", "psyllium", "simeticon"],
        [("CRP", "2", "mg/L"), ("TSH", "1.4", "mU/L")],
    ),
    (
        "acute pyelonefritis",
        ["nefrologie", "spoedgevallen"],
        ["flankpijn", "koorts", "dysurie"],
        ["ceftriaxon", "ciprofloxacine", "paracetamol"],
        [("CRP", "188", "mg/L"), ("nitriet urine", "positief", "")],
    ),
    (
        "niersteenlijden",
        ["urologie", "spoedgevallen"],
        ["koliekpijn", "hematurie", "bewegingsdrang"],
        ["diclofenac", "tamsulosine", "paracetamol"],
        [("erytrocyten urine", "3+", ""), ("creatinine", "1.2", "mg/dL")],
    ),
    (
        "nefrotisch syndroom",
        ["nefrologie", "immunologie"],
        ["oedeem", "schuimende urine", "gewichtstoename"],
        ["furosemide", "ramipril", "prednisolone"],
        [("proteinurie", "6.2", "g/dag"), ("albumine", "2.4", "g/dL")],
    ),
    (
        "urineweginfectie",
        ["huisartsgeneeskunde", "urologie"],
        ["dysurie", "pollakisurie", "suprapubische pijn"],
        ["nitrofurantoine", "fosfomycine", "paracetamol"],
        [("leukocyten urine", "3+", ""), ("nitriet urine", "positief", "")],
    ),
    (
        "prostaatcarcinoom",
        ["urologie", "oncologie"],
        ["mictieklachten", "botpijn", "gewichtsverlies"],
        ["leuproreline", "enzalutamide", "calcium/vitamine D"],
        [("PSA", "18.6", "ug/L"), ("alkalische fosfatase", "168", "U/L")],
    ),
    (
        "benigne prostaathyperplasie",
        ["urologie", "huisartsgeneeskunde"],
        ["nycturie", "zwakke straal", "nadruppelen"],
        ["tamsulosine", "finasteride", "solifenacine"],
        [("PSA", "3.4", "ug/L"), ("creatinine", "1.0", "mg/dL")],
    ),
    (
        "stressincontinentie",
        ["urologie", "bekkenbodemkliniek"],
        ["urineverlies bij hoesten", "urgentie", "schaamte"],
        ["bekkenbodemtherapie", "duloxetine", "vaginale oestrogenen"],
        [("urinekweek", "negatief", ""), ("residu urine", "35", "mL")],
    ),
    (
        "endometriose",
        ["gynaecologie", "pijnkliniek"],
        ["dysmenorroe", "dyspareunie", "chronische bekkenpijn"],
        ["dienogest", "naproxen", "tramadol"],
        [("CA-125", "42", "U/mL"), ("hemoglobine", "12.1", "g/dL")],
    ),
    (
        "pre-eclampsie",
        ["verloskunde", "gynaecologie"],
        ["hoofdpijn", "visusklachten", "bovenbuikpijn"],
        ["labetalol", "magnesiumsulfaat", "nifedipine"],
        [("proteinurie", "1.8", "g/dag"), ("bloedplaatjes", "118", "10^9/L")],
    ),
    (
        "zwangerschapsdiabetes",
        ["verloskunde", "diabetologie"],
        ["dorst", "vermoeidheid", "macrosomie risico"],
        ["dieetadvies", "insuline aspart", "insuline detemir"],
        [("OGTT 2u", "178", "mg/dL"), ("HbA1c", "5.9", "%")],
    ),
    (
        "infertiliteit",
        ["fertiliteitscentrum", "gynaecologie"],
        ["uitblijvende zwangerschap", "onregelmatige cyclus", "ovulatiepijn"],
        ["letrozol", "foliumzuur", "progesteron"],
        [("AMH", "1.4", "ug/L"), ("FSH", "8.9", "IU/L")],
    ),
    (
        "ovariumcyste",
        ["gynaecologie", "radiologie"],
        ["bekkenpijn", "opgeblazen gevoel", "cyclusgebonden pijn"],
        ["naproxen", "paracetamol", "orale anticonceptie"],
        [("CA-125", "18", "U/mL"), ("hemoglobine", "12.9", "g/dL")],
    ),
    (
        "menopauzale klachten",
        ["menopauzekliniek", "gynaecologie"],
        ["warmteopwellingen", "nachtzweten", "slaapstoornis"],
        ["estradiol", "progesteron", "clonidine"],
        [("FSH", "62", "IU/L"), ("TSH", "1.8", "mU/L")],
    ),
    (
        "otitis media",
        ["neus-keel-oor", "pediatrie"],
        ["oorpijn", "koorts", "gehoorsdaling"],
        ["amoxicilline", "ibuprofen", "xylometazoline"],
        [("CRP", "18", "mg/L"), ("leukocyten", "9.4", "10^9/L")],
    ),
    (
        "sinusitis",
        ["neus-keel-oor", "huisartsgeneeskunde"],
        ["aangezichtspijn", "purulente rinorroe", "drukgevoel"],
        ["mometason neusspray", "amoxicilline", "paracetamol"],
        [("CRP", "28", "mg/L"), ("eosinofielen", "0.22", "10^9/L")],
    ),
    (
        "gehoorverlies",
        ["neus-keel-oor", "audiologie"],
        ["verminderd spraakverstaan", "tinnitus", "drukgevoel"],
        ["betahistine", "prednisolone", "hoorapparaat proef"],
        [("spraakdiscriminatie", "72", "%"), ("toon-audiogram verlies", "38", "dB")],
    ),
    (
        "cataract",
        ["oftalmologie", "dagziekenhuis algemeen"],
        ["wazig zicht", "lichtverstrooiing", "nachtelijke rijproblemen"],
        ["kunsttranen", "prednisolon oogdruppels", "ofloxacine oogdruppels"],
        [("visus rechts", "0.4", ""), ("oogdruk", "15", "mmHg")],
    ),
    (
        "glaucoom",
        ["oftalmologie", "functieonderzoek"],
        ["gezichtsveldverlies", "oogdruk", "hoofdpijn"],
        ["latanoprost", "timolol", "brinzolamide"],
        [("oogdruk", "26", "mmHg"), ("cup-disc ratio", "0.7", "")],
    ),
    (
        "diabetische retinopathie",
        ["oftalmologie", "diabetologie"],
        ["wazig zicht", "floaters", "microaneurysmata"],
        ["anti-VEGF injectie", "metformine", "insuline glargine"],
        [("HbA1c", "9.1", "%"), ("albumine/creatinine ratio", "86", "mg/g")],
    ),
    (
        "psoriasis",
        ["dermatologie", "immunologie"],
        ["schilferende plaques", "jeuk", "nagelputjes"],
        ["calcipotriol/betamethason", "methotrexaat", "ustekinumab"],
        [("PASI", "14", ""), ("ALAT", "32", "U/L")],
    ),
    (
        "atopisch eczeem",
        ["dermatologie", "allergologie"],
        ["jeuk", "droge huid", "lichenificatie"],
        ["mometason zalf", "tacrolimus zalf", "cetirizine"],
        [("IgE totaal", "460", "kU/L"), ("eosinofielen", "0.58", "10^9/L")],
    ),
    (
        "cellulitis",
        ["dermatologie", "infectiologie"],
        ["roodheid", "warmte", "lokale pijn"],
        ["flucloxacilline", "clindamycine", "paracetamol"],
        [("CRP", "76", "mg/L"), ("leukocyten", "12.8", "10^9/L")],
    ),
    (
        "melanoom follow-up",
        ["dermatologie", "oncologie"],
        ["nieuwe pigmentvlek", "jeuk", "bloeding"],
        ["nivolumab", "pembrolizumab", "zonnebescherming"],
        [("LDH", "248", "U/L"), ("S100", "0.11", "ug/L")],
    ),
    (
        "reumatoide artritis",
        ["reumatologie", "immunologie"],
        ["ochtendstijfheid", "MCP-zwelling", "vermoeidheid"],
        ["methotrexaat", "foliumzuur", "adalimumab"],
        [("anti-CCP", "164", "U/mL"), ("CRP", "24", "mg/L")],
    ),
    (
        "jicht",
        ["reumatologie", "huisartsgeneeskunde"],
        ["acute grote teen pijn", "roodheid", "zwelling"],
        ["colchicine", "allopurinol", "naproxen"],
        [("urinezuur", "8.9", "mg/dL"), ("CRP", "38", "mg/L")],
    ),
    (
        "polymyalgia rheumatica",
        ["reumatologie", "geriatrie"],
        ["schoudergordelpijn", "heupstijfheid", "ochtendstijfheid"],
        ["prednisolone", "calcium/vitamine D", "pantoprazol"],
        [("BSE", "68", "mm/u"), ("CRP", "54", "mg/L")],
    ),
    (
        "osteoporose",
        ["osteoporosekliniek", "geriatrie"],
        ["rugpijn", "lengteverlies", "fractuurangst"],
        ["alendronaat", "calcium/vitamine D", "denosumab"],
        [("vitamine D", "18", "ng/mL"), ("calcium", "9.2", "mg/dL")],
    ),
    (
        "artrose knie",
        ["orthopedie", "revalidatie"],
        ["kniepijn", "startstijfheid", "crepitaties"],
        ["paracetamol", "diclofenac gel", "glucosamine"],
        [("CRP", "3", "mg/L"), ("urinezuur", "5.4", "mg/dL")],
    ),
    (
        "heupfractuur",
        ["orthopedie", "geriatrie"],
        ["heuppijn", "onvermogen tot steunen", "verkort been"],
        ["paracetamol", "morfine", "enoxaparine"],
        [("hemoglobine", "11.2", "g/dL"), ("vitamine D", "16", "ng/mL")],
    ),
    (
        "lumbale radiculopathie",
        ["neurochirurgie", "pijnkliniek"],
        ["uitstralende beenpijn", "paresthesie", "positieve Lasegue"],
        ["pregabaline", "naproxen", "paracetamol"],
        [("CRP", "2", "mg/L"), ("vitamine B12", "288", "ng/L")],
    ),
    (
        "schouderimpingement",
        ["orthopedie", "kinesitherapie"],
        ["pijnlijke abductie", "nachtpijn", "krachtsverlies"],
        ["ibuprofen", "paracetamol", "corticosteroid infiltratie"],
        [("CRP", "4", "mg/L"), ("bezinking", "12", "mm/u")],
    ),
    (
        "carpale tunnelsyndroom",
        ["neurologie", "handchirurgie"],
        ["nachtelijke tintelingen", "duimzwakte", "gevoelsstoornis"],
        ["polsspalk", "methylprednisolon infiltratie", "paracetamol"],
        [("geleidingssnelheid n. medianus", "38", "m/s"), ("TSH", "2.0", "mU/L")],
    ),
    (
        "migraine",
        ["neurologie", "huisartsgeneeskunde"],
        ["eenzijdige hoofdpijn", "fotofobie", "misselijkheid"],
        ["sumatriptan", "topiramaat", "naproxen"],
        [("CRP", "1", "mg/L"), ("hemoglobine", "13.6", "g/dL")],
    ),
    (
        "TIA",
        ["neurologie", "stroke unit"],
        ["kortdurende afasie", "hemiparese", "duizeligheid"],
        ["clopidogrel", "atorvastatine", "amlodipine"],
        [("LDL cholesterol", "128", "mg/dL"), ("HbA1c", "6.1", "%")],
    ),
    (
        "ischemisch CVA",
        ["neurologie", "stroke unit"],
        ["facialisparese", "armzwakte", "spraakstoornis"],
        ["alteplase", "acetylsalicylzuur", "atorvastatine"],
        [("INR", "1.0", ""), ("glucose", "142", "mg/dL")],
    ),
    (
        "ziekte van Parkinson",
        ["neurologie", "geriatrie"],
        ["rusttremor", "bradykinesie", "rigiditeit"],
        ["levodopa/carbidopa", "rasagiline", "domperidon"],
        [("vitamine B12", "356", "ng/L"), ("TSH", "2.1", "mU/L")],
    ),
    (
        "multiple sclerose",
        ["MS-kliniek", "neurologie"],
        ["sensibiliteitsstoornis", "visusdaling", "vermoeidheid"],
        ["ocrelizumab", "methylprednisolone", "vitamine D"],
        [("oligoklonale banden", "positief", ""), ("vitamine D", "22", "ng/mL")],
    ),
    (
        "perifere neuropathie",
        ["neurologie", "endocrinologie"],
        ["brandende voeten", "gevoelsverlies", "evenwichtsproblemen"],
        ["duloxetine", "pregabaline", "vitamine B12"],
        [("HbA1c", "7.8", "%"), ("vitamine B12", "198", "ng/L")],
    ),
    (
        "dementie",
        ["geheugenkliniek", "geriatrie"],
        ["geheugenverlies", "desorientatie", "woordvindproblemen"],
        ["donepezil", "memantine", "melatonine"],
        [("TSH", "1.6", "mU/L"), ("vitamine B12", "284", "ng/L")],
    ),
    (
        "angststoornis",
        ["psychiatrie", "klinische psychologie"],
        ["paniekaanvallen", "hyperventilatie", "vermijdingsgedrag"],
        ["escitalopram", "alprazolam", "propranolol"],
        [("TSH", "1.2", "mU/L"), ("vitamine D", "24", "ng/mL")],
    ),
    (
        "bipolaire stoornis",
        ["psychiatrie", "psychosomatiek"],
        ["verminderde slaapnood", "verhoogde spraakzaamheid", "prikkelbaarheid"],
        ["lithium", "quetiapine", "valproaat"],
        [("lithium spiegel", "0.72", "mmol/L"), ("creatinine", "0.95", "mg/dL")],
    ),
    (
        "alcoholafhankelijkheid",
        ["verslavingszorg", "psychiatrie"],
        ["craving", "tremor", "slaapproblemen"],
        ["acamprosaat", "thiamine", "diazepam"],
        [("gamma-GT", "182", "U/L"), ("MCV", "101", "fL")],
    ),
    (
        "psychose eerste episode",
        ["crisisinterventie", "psychiatrie"],
        ["wanen", "auditieve hallucinaties", "achterdocht"],
        ["risperidon", "lorazepam", "olanzapine"],
        [("CRP", "5", "mg/L"), ("TSH", "0.9", "mU/L")],
    ),
    (
        "suicide risico evaluatie",
        ["crisisinterventie", "psychiatrisch dagziekenhuis"],
        ["hopeloosheid", "insomnie", "doodswens"],
        ["sertraline", "trazodon", "lorazepam"],
        [("TSH", "1.5", "mU/L"), ("alcohol bloed", "0.0", "g/L")],
    ),
    (
        "anorexia nervosa",
        ["psychiatrie", "dietetiek"],
        ["gewichtsverlies", "amenorroe", "lichaamsbeeldverstoring"],
        ["olanzapine", "fosfaat suppletie", "thiamine"],
        [("fosfaat", "2.1", "mg/dL"), ("kalium", "3.3", "mmol/L")],
    ),
    (
        "ADHD volwassen leeftijd",
        ["psychiatrie", "klinische psychologie"],
        ["concentratieproblemen", "impulsiviteit", "innerlijke onrust"],
        ["methylfenidaat", "atomoxetine", "melatonine"],
        [("bloeddruk systolisch", "128", "mmHg"), ("hartfrequentie", "84", "/min")],
    ),
    (
        "acute appendicitis",
        ["algemene chirurgie", "spoedgevallen"],
        ["rechteronderbuikpijn", "koorts", "misselijkheid"],
        ["cefuroxim", "metronidazol", "paracetamol"],
        [("CRP", "92", "mg/L"), ("leukocyten", "15.1", "10^9/L")],
    ),
    (
        "liesbreuk",
        ["algemene chirurgie", "dagziekenhuis algemeen"],
        ["lieszwelling", "drukpijn", "toename bij hoesten"],
        ["paracetamol", "ibuprofen", "lactulose"],
        [("hemoglobine", "14.1", "g/dL"), ("INR", "1.0", "")],
    ),
    (
        "cholecystitis",
        ["hepatobiliaire chirurgie", "spoedgevallen"],
        ["rechterbovenbuikpijn", "koorts", "Murphy positief"],
        ["ceftriaxon", "metronidazol", "piritramide"],
        [("CRP", "154", "mg/L"), ("bilirubine", "1.9", "mg/dL")],
    ),
    (
        "colorectaal carcinoom",
        ["oncologie", "algemene chirurgie"],
        ["rectaal bloedverlies", "gewichtsverlies", "veranderd stoelgangspatroon"],
        ["capecitabine", "oxaliplatine", "ondansetron"],
        [("CEA", "18.4", "ug/L"), ("hemoglobine", "10.6", "g/dL")],
    ),
    (
        "schildkliernodus",
        ["endocrinologie", "radiologie"],
        ["halszwelling", "sliklast", "heesheid"],
        ["levothyroxine", "paracetamol", "selenium"],
        [("TSH", "0.42", "mU/L"), ("vrij T4", "1.6", "ng/dL")],
    ),
    (
        "hypothyreoidie",
        ["endocrinologie", "huisartsgeneeskunde"],
        ["koude-intolerantie", "gewichtstoename", "obstipatie"],
        ["levothyroxine", "vitamine D", "macrogol"],
        [("TSH", "12.8", "mU/L"), ("vrij T4", "0.7", "ng/dL")],
    ),
    (
        "hyperthyreoidie",
        ["endocrinologie", "nucleaire geneeskunde"],
        ["palpitaties", "gewichtsverlies", "tremor"],
        ["thiamazol", "propranolol", "prednisolone"],
        [("TSH", "0.01", "mU/L"), ("vrij T4", "3.4", "ng/dL")],
    ),
    (
        "bijnierincidentaloom",
        ["endocrinologie", "radiologie"],
        ["hypertensie", "spierzwakte", "toevallige CT-vondst"],
        ["spironolacton", "dexamethason test", "kaliumsuppletie"],
        [("cortisol na suppressie", "2.4", "ug/dL"), ("aldosteron", "21", "ng/dL")],
    ),
    (
        "obesitas",
        ["obesitaskliniek", "dietetiek"],
        ["gewichtstoename", "kniepijn", "snurken"],
        ["liraglutide", "orlistat", "vitamine D"],
        [("BMI", "36.8", "kg/m2"), ("HbA1c", "6.4", "%")],
    ),
    (
        "vitamine D deficientie",
        ["huisartsgeneeskunde", "geriatrie"],
        ["spierpijn", "vermoeidheid", "botpijn"],
        ["cholecalciferol", "calciumcarbonaat", "magnesium"],
        [("vitamine D", "11", "ng/mL"), ("calcium", "8.8", "mg/dL")],
    ),
    (
        "ijzergebreksanemie",
        ["hematologie", "huisartsgeneeskunde"],
        ["moeheid", "kortademigheid", "pica"],
        ["ijzerfumaraat", "ferricarboxymaltose", "foliumzuur"],
        [("hemoglobine", "8.9", "g/dL"), ("ferritine", "6", "ug/L")],
    ),
    (
        "lymfoom",
        ["hematologie", "oncologie"],
        ["nachtzweten", "lymfeklierzwelling", "gewichtsverlies"],
        ["rituximab", "cyclofosfamide", "prednisolone"],
        [("LDH", "418", "U/L"), ("beta-2-microglobuline", "4.1", "mg/L")],
    ),
    (
        "trombocytopenie",
        ["hematologie", "spoedgevallen"],
        ["petechien", "neusbloedingen", "blauwe plekken"],
        ["prednisolone", "intraveneus immunoglobuline", "tranexaminezuur"],
        [("bloedplaatjes", "24", "10^9/L"), ("hemoglobine", "12.4", "g/dL")],
    ),
    (
        "multipel myeloom",
        ["hematologie", "nefrologie"],
        ["botpijn", "anemie", "recidiverende infecties"],
        ["bortezomib", "lenalidomide", "dexamethason"],
        [("M-proteine", "31", "g/L"), ("calcium", "11.4", "mg/dL")],
    ),
    (
        "sepsis",
        ["intensieve zorgen", "infectiologie"],
        ["koorts", "hypotensie", "verwardheid"],
        ["piperacilline/tazobactam", "noradrenaline", "Ringer lactaat"],
        [("lactaat", "3.8", "mmol/L"), ("CRP", "226", "mg/L")],
    ),
    (
        "bacteriele meningitis",
        ["infectiologie", "neurologie"],
        ["nekstijfheid", "koorts", "fotofobie"],
        ["ceftriaxon", "vancomycine", "dexamethason"],
        [("CSV leukocyten", "1240", "/uL"), ("CSV glucose", "28", "mg/dL")],
    ),
    (
        "HIV follow-up",
        ["infectiologie", "immunologie"],
        ["nachtzweten", "moeheid", "therapietrouwvragen"],
        ["bictegravir/emtricitabine/tenofovir", "cotrimoxazol", "vitamine D"],
        [("HIV RNA", "niet detecteerbaar", ""), ("CD4", "486", "/uL")],
    ),
    (
        "hepatitis C",
        ["hepatologie", "infectiologie"],
        ["vermoeidheid", "rechterbovenbuikpijn", "jeuk"],
        ["sofosbuvir/velpatasvir", "ribavirine", "ondansetron"],
        [("HCV RNA", "620000", "IU/mL"), ("ALAT", "114", "U/L")],
    ),
    (
        "Lyme neuroborreliose",
        ["infectiologie", "neurologie"],
        ["radiculaire pijn", "facialisparese", "hoofdpijn"],
        ["ceftriaxon", "doxycycline", "paracetamol"],
        [("Borrelia IgG", "positief", ""), ("CSV eiwit", "88", "mg/dL")],
    ),
    (
        "diabetische voetwonde",
        ["wondzorgkliniek", "diabetologie"],
        ["voetulcus", "eeltvorming", "verminderde sensibiliteit"],
        ["amoxicilline/clavulaanzuur", "insuline glargine", "wondverband"],
        [("HbA1c", "9.6", "%"), ("CRP", "64", "mg/L")],
    ),
    (
        "decubituswonde",
        ["wondzorgkliniek", "geriatrie"],
        ["sacrale wonde", "pijn", "exsudaat"],
        ["zilververband", "paracetamol", "proteinesupplement"],
        [("albumine", "3.0", "g/dL"), ("CRP", "44", "mg/L")],
    ),
    (
        "palliatieve symptoomcontrole",
        ["palliatieve zorg", "thuiszorg"],
        ["pijn", "dyspneu", "angst"],
        ["morfine", "midazolam", "haloperidol"],
        [("creatinine", "1.1", "mg/dL"), ("natrium", "136", "mmol/L")],
    ),
    (
        "pacemakercontrole",
        ["pacemakerkliniek", "cardiologie"],
        ["duizeligheid", "palpitaties", "inspanningstolerantie"],
        ["bisoprolol", "apixaban", "atorvastatine"],
        [("batterijstatus", "8.2", "jaar"), ("ventriculaire pacing", "62", "%")],
    ),
    (
        "cochleair implantaat controle",
        ["neus-keel-oor", "audiologie"],
        ["spraakverstaan", "tinnitus", "druk rond processor"],
        ["ofloxacine oordruppels", "paracetamol", "hoorrevalidatie"],
        [("spraakscore", "68", "%"), ("impedantie elektrode", "normaal", "")],
    ),
]

CONDITIONS = [_condition_record(*row) for row in _CONDITION_ROWS]

_GENETIC_FINDING_ROWS = [
    ("BRCA1", "c.5266dupC", "p.Gln1756Profs*74", "rs80357906", "pathogeen"),
    ("CFTR", "c.1521_1523delCTT", "p.Phe508del", "rs113993960", "pathogeen"),
    ("EGFR", "c.2573T>G", "p.Leu858Arg", "rs121434568", "klinisch relevant"),
    ("VHL", "c.500G>A", "p.Arg167Gln", "rs5030821", "waarschijnlijk pathogeen"),
    ("BRCA2", "c.5946delT", "p.Ser1982Argfs*22", "rs80359550", "pathogeen"),
    (
        "MLH1",
        "c.1852_1854delAAG",
        "p.Lys618del",
        "rs63750362",
        "waarschijnlijk pathogeen",
    ),
    ("MSH2", "c.942+3A>T", "p.?", "rs63750447", "pathogeen"),
    ("MSH6", "c.3261dupC", "p.Phe1088Leufs*5", "rs63751310", "pathogeen"),
    ("PMS2", "c.137G>T", "p.Ser46Ile", "rs587782523", "variant van onzekere betekenis"),
    ("APC", "c.3927_3931delAAAGA", "p.Glu1309Aspfs*4", "rs121913332", "pathogeen"),
    ("MUTYH", "c.536A>G", "p.Tyr179Cys", "rs34612342", "pathogeen"),
    ("TP53", "c.743G>A", "p.Arg248Gln", "rs11540652", "pathogeen"),
    ("PTEN", "c.388C>T", "p.Arg130*", "rs121909229", "pathogeen"),
    ("STK11", "c.842delC", "p.Pro281Leufs*6", "rs587776642", "pathogeen"),
    ("RET", "c.1901G>A", "p.Cys634Tyr", "rs75996173", "pathogeen"),
    ("MEN1", "c.784-9G>A", "p.?", "rs794728278", "waarschijnlijk pathogeen"),
    ("SDHB", "c.423+1G>A", "p.?", "rs587777560", "pathogeen"),
    ("SDHD", "c.242C>T", "p.Pro81Leu", "rs104894294", "pathogeen"),
    ("CHEK2", "c.1100delC", "p.Thr367Metfs*15", "rs555607708", "pathogeen"),
    ("PALB2", "c.3113G>A", "p.Trp1038*", "rs180177143", "pathogeen"),
    ("ATM", "c.7271T>G", "p.Val2424Gly", "rs28904921", "waarschijnlijk pathogeen"),
    ("NBN", "c.657_661delACAAA", "p.Lys219Asnfs*16", "rs587776650", "pathogeen"),
    (
        "CDH1",
        "c.1901C>T",
        "p.Ala634Val",
        "rs587778534",
        "variant van onzekere betekenis",
    ),
    ("SMAD4", "c.1244G>A", "p.Arg415His", "rs121912578", "pathogeen"),
    ("BMPR1A", "c.1435C>T", "p.Arg479*", "rs587779807", "pathogeen"),
    ("TSC1", "c.1831C>T", "p.Arg611*", "rs397514625", "pathogeen"),
    ("TSC2", "c.1832G>A", "p.Arg611Gln", "rs45517134", "waarschijnlijk pathogeen"),
    ("NF1", "c.6855C>A", "p.Tyr2285*", "rs587780666", "pathogeen"),
    ("NF2", "c.784C>T", "p.Arg262*", "rs587780280", "pathogeen"),
    ("RB1", "c.958C>T", "p.Arg320*", "rs121913300", "pathogeen"),
    ("WT1", "c.1180C>T", "p.Arg394Trp", "rs121907907", "pathogeen"),
    ("ALK", "c.3520T>C", "p.Phe1174Leu", "rs121913378", "klinisch relevant"),
    ("BRAF", "c.1799T>A", "p.Val600Glu", "rs113488022", "klinisch relevant"),
    ("KRAS", "c.35G>A", "p.Gly12Asp", "rs121913529", "klinisch relevant"),
    ("NRAS", "c.181C>A", "p.Gln61Lys", "rs121913254", "klinisch relevant"),
    ("KIT", "c.2447A>T", "p.Asp816Val", "rs121913507", "klinisch relevant"),
    ("PDGFRA", "c.2525A>T", "p.Asp842Val", "rs121908599", "klinisch relevant"),
    ("JAK2", "c.1849G>T", "p.Val617Phe", "rs77375493", "klinisch relevant"),
    ("CALR", "c.1092_1143del", "p.Leu367Thrfs*46", "rs1555769767", "klinisch relevant"),
    ("MPL", "c.1544G>T", "p.Trp515Leu", "rs121913615", "klinisch relevant"),
    ("FLT3", "c.2503G>T", "p.Asp835Tyr", "rs121913488", "klinisch relevant"),
    ("IDH1", "c.395G>A", "p.Arg132His", "rs121913500", "klinisch relevant"),
    ("IDH2", "c.419G>A", "p.Arg140Gln", "rs121913503", "klinisch relevant"),
    (
        "NPM1",
        "c.860_863dupTCTG",
        "p.Trp288Cysfs*12",
        "rs587776806",
        "klinisch relevant",
    ),
    ("DNMT3A", "c.2645G>A", "p.Arg882His", "rs147001633", "klinisch relevant"),
    ("TET2", "c.3964C>T", "p.Arg1322*", "rs759949069", "klinisch relevant"),
    ("ASXL1", "c.1934dupG", "p.Gly646Trpfs*12", "rs750765331", "klinisch relevant"),
    ("RUNX1", "c.958C>T", "p.Arg320*", "rs121912499", "waarschijnlijk pathogeen"),
    ("CEBPA", "c.68dupC", "p.His24Alafs*84", "rs778288150", "klinisch relevant"),
    ("GATA2", "c.1061C>T", "p.Thr354Met", "rs387906936", "pathogeen"),
    ("F5", "c.1601G>A", "p.Arg534Gln", "rs6025", "trombofilie-risico"),
    ("F2", "c.*97G>A", "p.?", "rs1799963", "trombofilie-risico"),
    ("HFE", "c.845G>A", "p.Cys282Tyr", "rs1800562", "pathogeen"),
    ("SERPINA1", "c.1096G>A", "p.Glu366Lys", "rs28929474", "pathogeen"),
    ("LDLR", "c.662A>G", "p.Asp221Gly", "rs121908028", "pathogeen"),
    ("APOB", "c.10580G>A", "p.Arg3527Gln", "rs5742904", "pathogeen"),
    ("PCSK9", "c.1120G>T", "p.Asp374Tyr", "rs137852912", "pathogeen"),
    ("MYH7", "c.1208G>A", "p.Arg403Gln", "rs121913641", "pathogeen"),
    ("MYBPC3", "c.1504C>T", "p.Arg502Trp", "rs375882485", "waarschijnlijk pathogeen"),
    ("TNNT2", "c.518G>A", "p.Arg173Gln", "rs121964857", "pathogeen"),
    (
        "TNNI3",
        "c.470C>T",
        "p.Ala157Val",
        "rs397516366",
        "variant van onzekere betekenis",
    ),
    ("LMNA", "c.1130G>A", "p.Arg377His", "rs267607571", "pathogeen"),
    (
        "KCNQ1",
        "c.1032G>A",
        "p.Ser344Ser",
        "rs1057128",
        "variant van onzekere betekenis",
    ),
    (
        "KCNH2",
        "c.1898A>G",
        "p.Asn633Ser",
        "rs1805123",
        "variant van onzekere betekenis",
    ),
    ("SCN5A", "c.3823G>A", "p.Asp1275Asn", "rs199473199", "waarschijnlijk pathogeen"),
    ("RYR2", "c.12530T>A", "p.Val4177Asp", "rs794728683", "waarschijnlijk pathogeen"),
    ("FBN1", "c.7585C>T", "p.Arg2529Cys", "rs137854468", "pathogeen"),
    ("TGFBR1", "c.1459C>T", "p.Arg487Trp", "rs121909120", "pathogeen"),
    ("TGFBR2", "c.1582C>T", "p.Arg528Cys", "rs121918170", "pathogeen"),
    ("COL3A1", "c.1662+1G>A", "p.?", "rs587779621", "pathogeen"),
    ("COL1A1", "c.769G>A", "p.Gly257Arg", "rs72658152", "pathogeen"),
    ("GLA", "c.640-801G>A", "p.?", "rs2071225", "variant van onzekere betekenis"),
    ("GBA", "c.1226A>G", "p.Asn409Ser", "rs76763715", "pathogeen"),
    ("LRRK2", "c.6055G>A", "p.Gly2019Ser", "rs34637584", "pathogeen"),
    ("HTT", "c.52CAG[42]", "p.Gln18[42]", "rs362331", "pathogeen"),
    ("DMD", "c.8713C>T", "p.Arg2905*", "rs398123847", "pathogeen"),
    ("SMN1", "c.840C>T", "p.?", "rs116432568", "pathogeen"),
    ("MECP2", "c.473C>T", "p.Thr158Met", "rs61751362", "pathogeen"),
    ("FMR1", "c.-129CGG[90]", "p.?", "rs193922937", "pathogeen"),
    ("PAH", "c.1222C>T", "p.Arg408Trp", "rs5030858", "pathogeen"),
    ("GALT", "c.563A>G", "p.Gln188Arg", "rs75391579", "pathogeen"),
    ("GAA", "c.-32-13T>G", "p.?", "rs386834236", "pathogeen"),
    ("HEXA", "c.1274_1277dupTATC", "p.Tyr427Ilefs*5", "rs387906309", "pathogeen"),
    ("ATP7B", "c.3207C>A", "p.His1069Gln", "rs76151636", "pathogeen"),
    ("CYP21A2", "c.293-13C>G", "p.?", "rs6467", "pathogeen"),
    ("HBB", "c.20A>T", "p.Glu7Val", "rs334", "pathogeen"),
    ("HBA2", "c.427T>C", "p.Ter143Glnext*31", "rs63751267", "pathogeen"),
    ("F8", "c.6046C>T", "p.Arg2016Trp", "rs28935206", "pathogeen"),
    ("F9", "c.1025C>T", "p.Thr342Met", "rs137852248", "pathogeen"),
    ("COL4A5", "c.1871G>A", "p.Gly624Asp", "rs104886118", "pathogeen"),
    ("PKD1", "c.5014C>T", "p.Arg1672Trp", "rs397515811", "pathogeen"),
    ("UMOD", "c.857G>A", "p.Cys286Tyr", "rs121908818", "pathogeen"),
    ("SLC2A1", "c.680G>A", "p.Arg227His", "rs121908143", "pathogeen"),
    ("TTR", "c.424G>A", "p.Val142Ile", "rs76992529", "pathogeen"),
    ("PRNP", "c.598G>A", "p.Glu200Lys", "rs28933385", "pathogeen"),
    ("HNF1A", "c.872dupC", "p.Gly292Argfs*25", "rs587776933", "pathogeen"),
    ("KCNJ11", "c.602G>A", "p.Arg201His", "rs80356610", "pathogeen"),
    ("ABCC8", "c.3992-9G>A", "p.?", "rs757110", "variant van onzekere betekenis"),
    ("CYP2C19", "c.681G>A", "p.Pro227Pro", "rs4244285", "farmacogenetisch relevant"),
    ("DPYD", "c.1905+1G>A", "p.?", "rs3918290", "farmacogenetisch relevant"),
]

GENETIC_FINDINGS = [_genetic_finding(*row) for row in _GENETIC_FINDING_ROWS]

DOCUMENT_TYPES = [
    "ai_scribe_note",
    "discharge_summary",
    "ed_note",
    "consult_letter",
    "lab_report",
    "genetics_report",
    "oncology_mdo",
    "medication_reconciliation",
    "nursing_note",
    "radiology_summary",
    "referral_letter",
    "home_care_report",
    "rehab_progress",
    "pathology_report",
    "device_implant_note",
    "secure_email",
    "anesthesia_operating_grid",
    "pulmonary_calibration_report",
]

NOTE_STYLES = [
    "compacte SOAP-nota",
    "uitgebreide narratieve verslagbrief",
    "gestructureerd probleem-georienteerd verslag",
    "chronologisch consultatieverslag",
    "kort verpleegkundig overdrachtsverslag",
    "multidisciplinair overlegverslag",
    "telefonische opvolgnota",
    "dagziekenhuis behandelnota",
    "klinisch rapport met tabelachtige resultaten",
    "ontslagplanning met sociale context",
    "urgentie-observatie met triagefocus",
    "medicatiegerichte reconciliatienota",
    "revalidatievoortgang met functionele doelen",
    "genetische interpretatiebrief",
    "beeldvormingssamenvatting",
    "pathologieverslag met klinische correlatie",
    "thuiszorgcommunicatie",
    "preoperatieve evaluatie",
    "postoperatieve controle",
    "palliatieve symptoomnota",
    "zorgpad-opvolging",
    "korte verwijsbrief",
    "lange specialistische terugbrief",
    "administratieve attestaanvraag",
    "risico-inschatting voor opvolging",
    "familiale contextnota",
    "patientinstructies na contact",
    "technisch procedureverslag",
    "labogerichte interpretatienota",
    "interprofessioneel zorgplan",
]

STYLE_PROFILES = _load_jsonl_resource("style_profiles.jsonl")
MEDICAL_DETAIL_POOLS = _load_jsonl_resource("medical_detail_pools.jsonl")
PIXELPHARMA_MEDICINES = _load_tsv_resource("pixelpharma_medicines.tsv")
ENSIE_MEDICAL_EPONYMS = _load_eponym_resource("ensie_medical_eponyms.tsv")

LANGUAGES = ["nl"]

PROFESSIONS = [
    "gepensioneerde postbode",
    "zelfstandige kapster",
    "software-ingenieur",
    "leerkracht lager onderwijs",
    "student verpleegkunde",
    "kok",
    "2de leerjaar",
    "5de middelbaar",
    "opleiding communicatiewetenschappen",
    "zelfstandige",
    "buschauffeur",
    "poetshulp",
    "administratief bediende",
    "boekhouder",
    "magazijnier",
    "vrachtwagenchauffeur",
    "verpleegkundige",
    "zorgkundige",
    "kinesitherapeut",
    "apotheker",
    "huisarts in opleiding",
    "tandartsassistente",
    "laborant",
    "radiologisch technoloog",
    "maatschappelijk werker",
    "politie-inspecteur",
    "brandweerman",
    "militair",
    "havenarbeider",
    "procesoperator",
    "elektricien",
    "loodgieter",
    "schrijnwerker",
    "metser",
    "dakwerker",
    "schilder",
    "tuinaannemer",
    "landbouwer",
    "melkveehouder",
    "bloemenkweker",
    "slager",
    "bakker",
    "winkelbediende",
    "kassamedewerker",
    "verkoopster",
    "horeca-uitbater",
    "hotelreceptionist",
    "poetsteamleider",
    "kinderbegeleider",
    "onthaalouder",
    "kleuterjuf",
    "docent hogeschool",
    "onderzoeker biomedische wetenschappen",
    "doctoraatsstudent",
    "grafisch ontwerper",
    "journalist",
    "vertaler",
    "advocaat",
    "notarieel medewerker",
    "gerechtsdeurwaarder",
    "verzekeringsmakelaar",
    "bankbediende",
    "vastgoedmakelaar",
    "architect",
    "interieurarchitect",
    "projectleider bouw",
    "werfleider",
    "veiligheidscoordinator",
    "IT-consultant",
    "data-analist",
    "helpdeskmedewerker",
    "callcenteragent",
    "telecomtechnicus",
    "treinbegeleider",
    "loketbediende NMBS",
    "piloot",
    "steward",
    "taxichauffeur",
    "fietskoerier",
    "student geneeskunde",
    "student rechten",
    "bachelorstudent informatica",
    "masterstudent psychologie",
    "secundair onderwijs derde graad",
    "1ste middelbaar",
    "3de kleuterklas",
    "universitair docent",
    "gepensioneerde verpleegkundige",
    "invaliditeitsuitkering",
    "werkzoekende",
    "huisman",
    "mantelzorger",
    "vrijwilliger sportclub",
    "professioneel muzikant",
    "theatertechnicus",
    "fotograaf",
    "webwinkeluitbater",
    "zelfstandige boekhoudconsulent",
    "kwaliteitsmanager",
    "productieplanner",
]

OTHER_ORGS = [
    "BASF Antwerpen",
    "Korfbalclub Riviera",
    "Sint-Jozefscollege",
    "ING Bank",
    "AP Hogeschool",
    "Basisschool Regenboog",
    "Rode Kruis Vlaanderen",
    "restaurant De Gouden Lepel",
    "Atlas Copco Wilrijk",
    "Port of Antwerp-Bruges",
    "Colruyt Group",
    "Delhaize filiaal Mechelen",
    "Carrefour Market Leuven",
    "Bpost sorteercentrum Gent",
    "NMBS werkplaats Melle",
    "De Lijn stelplaats Hasselt",
    "Brussels Airport Company",
    "Proximus contactcenter",
    "Telenet Business",
    "KBC Verzekeringen",
    "Belfius kantoor Aalst",
    "BNP Paribas Fortis Turnhout",
    "Universiteit Gent",
    "KU Leuven",
    "Vrije Universiteit Brussel",
    "Universiteit Antwerpen",
    "UHasselt",
    "Thomas More Hogeschool",
    "Arteveldehogeschool",
    "Odisee Hogeschool",
    "Howest",
    "PXL Hogeschool",
    "GO! Atheneum Etterbeek",
    "Vrije Basisschool De Horizon",
    "Gemeenteschool De Beuk",
    "Sint-Lievenscollege",
    "Don Bosco Technisch Instituut",
    "CLB Noord Antwerpen",
    "Kinderdagverblijf Zonnetje",
    "Chiro Sint-Martinus",
    "Scouts De Klauwaards",
    "KSA Sint-Paulus",
    "Voetbalclub KFC Linden",
    "Basketbalclub Gembo",
    "Zwemclub Brabo",
    "Tennisclub Den Brandt",
    "Turnkring Olympia",
    "Harmonie De Eendracht",
    "Bibliotheek Permeke",
    "Cultuurcentrum De Werf",
    "Stad Antwerpen",
    "Stad Gent",
    "Stad Leuven",
    "Gemeente Brasschaat",
    "Gemeente Wetteren",
    "OCMW Brugge",
    "VDAB kantoor Kortrijk",
    "RVA kantoor Luik",
    "FOD Financien",
    "Vlaams Agentschap Wegen en Verkeer",
    "Agentschap Natuur en Bos",
    "Vlaamse Milieumaatschappij",
    "De Watergroep",
    "Fluvius netbeheer",
    "Elia onderhoudsteam",
    "Aquafin projectdienst",
    "Jan De Nul Group",
    "DEME Offshore",
    "Besix",
    "Willemen Construct",
    "Matexi",
    "Immoweb Services",
    "VRT NWS",
    "DPG Media",
    "Mediahuis",
    "Roularta Media Group",
    "Studio Brussel",
    "Ancienne Belgique",
    "Sportpaleis Antwerpen",
    "Kinepolis Gent",
    "Plopsaland De Panne",
    "Pairi Daiza",
    "Zoo Antwerpen",
    "Sport Vlaanderen Herentals",
    "Fitnessclub Basic-Fit",
    "Yoga Studio Lotus",
    "Fietsenwinkel Velodroom",
    "Garage Peeters",
    "Bakkerij Het Broodhuis",
    "Slagerij De Markt",
    "Apotheek De Vaart",
    "Notariskantoor Vermeulen",
    "Advocatenkantoor Lexnova",
    "Accountantskantoor Fiducia",
    "Brouwerij De Halve Maan",
    "Chocolaterie Van Hoorebeke",
    "Boerenbond",
    "Natuurpunt afdeling Dijleland",
    "buurtcomite De Lindeboom",
    "theatergezelschap Het Gevolg",
]

PEDIATRIC_ACTIVITY_BY_GROUP = {
    "infant": [
        "kinderdagverblijf",
        "onthaaloudertraject",
        "consultatie Kind en Gezin",
        "babyzwemmen",
    ],
    "toddler": [
        "peuterklas",
        "kinderdagverblijf peutergroep",
        "onthaaloudertraject",
        "psychomotoriek peuters",
    ],
    "preschool_child": [
        "eerste kleuterklas",
        "tweede kleuterklas",
        "derde kleuterklas",
        "turnclub kleuters",
    ],
    "school_age_child": [
        "2de leerjaar",
        "5de leerjaar",
        "leerling lagere school",
        "jeugdvoetbal U11",
    ],
    "adolescent": [
        "leerling secundair onderwijs",
        "student 5de middelbaar",
        "duaal leren",
        "jeugdbeweging leiding in opleiding",
    ],
}

PEDIATRIC_ORGS_BY_GROUP = {
    "infant": [
        "Kind en Gezin consultatiebureau",
        "Kinderdagverblijf De Speelboom",
        "Onthaalouderdienst Ferm",
        "Babyzwemmen De Plons",
    ],
    "toddler": [
        "Kinderdagverblijf De Speelboom",
        "Peuterspeelpunt De Tuimelaar",
        "Onthaalouderdienst Ferm",
        "Psychomotoriek De Klimtoren",
    ],
    "preschool_child": [
        "Vrije Kleuterschool Sint-Jan",
        "Stedelijke Kleuterschool De Regenboog",
        "Turnclub De Springertjes",
        "Buitenschoolse opvang De Speelvogel",
    ],
    "school_age_child": [
        "Basisschool De Horizon",
        "Voetbalclub U11 KFC Wilrijk",
        "Academie voor Muziek en Woord",
        "Scouts Sint-Paulus",
    ],
    "adolescent": [
        "GO! Atheneum Antwerpen",
        "Sint-Jozefscollege",
        "Jeugdhuis De Linie",
        "Basketbalclub Gembo",
    ],
}

PEDIATRIC_FALLBACK_CONDITIONS = {
    "infant": _condition_record(
        "preventieve zuigelingencontrole",
        ["pediatrie", "huisartsgeneeskunde", "neonatologie"],
        ["voedingsvragen", "groeicontrole", "slaappatroon wisselend"],
        ["cholecalciferol", "fysiologisch serum", "paracetamol op gewicht"],
        [("gewicht", "6.8", "kg"), ("lengte", "64", "cm")],
    ),
    "preschool_child": _condition_record(
        "bovensteluchtweginfectie bij kleuter",
        ["pediatrie", "huisartsgeneeskunde", "neus-keel-oor"],
        ["koorts", "neusloop", "hoest"],
        ["paracetamol op gewicht", "fysiologisch serum", "ibuprofen zo nodig"],
        [("temperatuur", "38.4", "°C"), ("zuurstofsaturatie", "98", "%")],
    ),
    "toddler": _condition_record(
        "acute otitis media bij peuter",
        ["pediatrie", "huisartsgeneeskunde", "neus-keel-oor"],
        ["oorpijn", "koorts", "prikkelbaarheid"],
        ["paracetamol op gewicht", "amoxicilline", "fysiologisch serum"],
        [("temperatuur", "38.7", "°C"), ("gewicht", "12.4", "kg")],
    ),
    "school_age_child": _condition_record(
        "astma-opvolging bij kind",
        ["pediatrie", "pneumologie", "huisartsgeneeskunde"],
        [
            "piepende ademhaling",
            "nachtelijke hoest",
            "inspanningstolerantie verminderd",
        ],
        ["salbutamol inhalatie", "fluticason inhalatie", "cetirizine"],
        [("FEV1", "82", "% voorspeld"), ("zuurstofsaturatie", "99", "%")],
    ),
    "adolescent": _condition_record(
        "sportletsel bij adolescent",
        ["pediatrie", "orthopedie", "spoedgevallen"],
        ["kniepijn", "zwelling na training", "manken"],
        ["paracetamol", "ibuprofen", "ijsapplicatie"],
        [("pijnscore", "6", "/10"), ("CRP", "4", "mg/L")],
    ),
}

PEDIATRIC_AGE_GROUPS = frozenset(PEDIATRIC_FALLBACK_CONDITIONS)
PEDIATRIC_TARGET_RATIO = 0.20
PEDIATRIC_GROUP_ORDER = (
    "infant",
    "toddler",
    "preschool_child",
    "school_age_child",
    "adolescent",
)


@dataclass(frozen=True)
class PersonProfile:
    name: str
    source_paths: dict


@dataclass(frozen=True)
class AddressProfile:
    text: str
    source_paths: dict


@dataclass(frozen=True)
class ClinicalCase:
    document_type: str
    language: str
    department: str
    condition: dict
    patient: PersonProfile
    caregiver: PersonProfile
    secondary_caregiver: PersonProfile
    relative: PersonProfile
    patient_address: AddressProfile
    hospital: tuple[str, str]
    healthcare_institution: tuple[str, str]
    caregiver_locality: tuple[str, str]
    other_location: tuple[str, str]
    profession: str
    other_org: str
    age_text: str
    age_context: str
    birthdate: str
    birthdate_prefix: str
    encounter_date: str
    followup_date: str
    genetic_finding: dict
    identifiers: dict
    contact: dict
    hard_negatives: list[str]
    note_style: str = "compacte SOAP-nota"
    synthea_source: dict | None = None
    coverage_targets: list[str] | None = None
    style_profile: dict | None = None
    medical_details: dict | None = None
    date_overview: dict[str, str] | None = None
    date_times: dict[str, str] | None = None
    date_periods: dict[str, str] | None = None
    date_focus: dict[str, str] | None = None
    date_focus_template: str | None = None
    date_focus_style: int | None = None
    administrative_gender: str = ""


def _address(sampler: LookupSampler, rng: random.Random) -> AddressProfile:
    street, street_source = sampler.street()
    postal, postal_source = sampler.postal_locality()
    return AddressProfile(
        text=f"{street} {rng.randrange(1, 180)}, {postal}",
        source_paths={"street": street_source, "postal_locality": postal_source},
    )


def _person(sampler: LookupSampler) -> PersonProfile:
    name, sources = full_name(sampler)
    return PersonProfile(name=name, source_paths=sources)


def person_name_variants(name: str) -> dict[str, str]:
    parts = [part for part in name.split() if part]
    if not parts:
        return {}
    first = parts[0]
    surname = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    return {
        key: value
        for key, value in {
            "first": first,
            "surname": surname,
        }.items()
        if value
    }


def caregiver_name_variants(name: str) -> dict[str, str]:
    variants = person_name_variants(name)
    if not variants:
        return {}
    parts = [part for part in name.split() if part]
    first = variants["first"]
    surname = variants["surname"]
    initials = [f"{part[0].upper()}." for part in parts if part]
    initials_nodot = "".join(part[0].upper() for part in parts if part)
    surname_first = f"{surname}, {first}" if surname != first else name
    caregiver_variants = {
        "first": first,
        "initial_surname": f"{first[0].upper()}. {surname}",
        "initial_surname_nodot": f"{first[0].upper()} {surname}",
        "initials_compact": "".join(initials),
        "initials_compact_nodot": initials_nodot,
        "initials_spaced": " ".join(initials),
        "initials_spaced_nodot": " ".join(initials_nodot),
        "first_last_initial": f"{first} {surname[0].upper()}.",
        "first_last_initial_nodot": f"{first} {surname[0].upper()}",
        "surname": surname,
        "surname_first": surname_first,
        "surname_first_dr": f"{surname_first}, Dr.",
    }
    return {
        key: value
        for key, value in caregiver_variants.items()
        if value and value != name
    }


def _format_date_variant(value: date, index: int, slot: str) -> str:
    return format_date_variant(value, index, slot)


def _date_context_prefix(index: int, slot: str) -> str:
    return date_context_prefix(index, slot)


def _encounter_date(rng: random.Random) -> date:
    return sample_encounter_date(rng)


def _followup_date(rng: random.Random, encounter_date: date) -> date:
    return encounter_date + timedelta(days=rng.randrange(3, 121))


DATE_FOCUS_INTERVAL = 7
DATE_FOCUS_TEMPLATES = (
    "compact_timeline",
    "screening_rows",
    "functional_rows",
    "history_list",
    "categorized_antecedents",
    "checklist",
    "measurement_rows",
)
DATE_FOCUS_SCHEDULE: tuple[tuple[str, str, int], ...] = (
    ("aanmelding_numeric_slash_long", "numeric_slash_long", -28),
    ("telefonisch_numeric_slash_short", "numeric_slash_short", -21),
    ("opname_numeric_dash_long", "numeric_dash_long", -14),
    ("controle_numeric_dash_short", "numeric_dash_short", -7),
    ("staal_numeric_dot_long", "numeric_dot_long", -2),
    ("consult_textual_full", "textual_full", 0),
    ("brief_textual_abbr_dot", "textual_abbr_dot", 3),
    ("ingreep_textual_hyphen", "textual_hyphen", 9),
    ("mdo_weekday_numeric", "weekday_numeric", 14),
    ("verpleeg_weekday_textual", "weekday_textual", 21),
    ("dagplan_day_month_numeric", "day_month_numeric", 28),
    ("thuiszorg_day_month_textual", "day_month_textual", 35),
    ("revalidatie_month_year", "month_year", 63),
    ("antecedenten_month_year_1", "numeric_month_year_slash", -(365 * 12)),
    ("antecedenten_month_year_2", "numeric_month_year_slash", -(365 * 7)),
    ("antecedenten_month_year_3", "numeric_month_year_slash", -(365 * 3)),
    ("voorgeschiedenis_year_1", "year_only", -(365 * 34)),
    ("voorgeschiedenis_year_2", "year_only", -(365 * 28)),
    ("voorgeschiedenis_year_3", "year_only", -(365 * 21)),
    ("voorgeschiedenis_year_4", "year_only", -(365 * 15)),
    ("voorgeschiedenis_year_5", "year_only", -(365 * 9)),
    ("lange_termijn_year_only", "year_only", 365),
)


def _is_date_focus_case(index: int) -> bool:
    return index % DATE_FOCUS_INTERVAL == 3


def _date_focus(encounter_date: date, index: int) -> dict[str, str] | None:
    """Dense Date slots for notes dedicated to date-format annotation coverage."""

    if not _is_date_focus_case(index):
        return None
    values: dict[str, str] = {}
    for offset, (key, profile_name, day_delta) in enumerate(DATE_FOCUS_SCHEDULE):
        value = encounter_date + timedelta(days=day_delta)
        values[key] = format_named_date_profile(
            value,
            profile_name,
            variant=index + 20_000 + offset,
        )
    return values


def _date_focus_template(index: int) -> str | None:
    if not _is_date_focus_case(index):
        return None
    focus_ordinal = index // DATE_FOCUS_INTERVAL
    return DATE_FOCUS_TEMPLATES[focus_ordinal % len(DATE_FOCUS_TEMPLATES)]


def _date_focus_style(index: int) -> int | None:
    if not _is_date_focus_case(index):
        return None
    focus_ordinal = index // DATE_FOCUS_INTERVAL
    return focus_ordinal // len(DATE_FOCUS_TEMPLATES)


def _date_overview(
    encounter_date: date, followup_date: date, index: int
) -> dict[str, str]:
    """Extra Date slots for compact histories and timeline/plan summaries."""

    history_start = date(encounter_date.year - (2 + index % 9), 1, 1)
    history_review = date(encounter_date.year - (index % 4), 1, 1)
    plan_year = date(followup_date.year + (1 if index % 6 == 0 else 0), 1, 1)
    completed_date = encounter_date - timedelta(days=7 + (index % 35))
    return {
        "history_start_year": format_date_variant(
            history_start, index + 10_001, "date", precision="year"
        ),
        "history_review_year": format_date_variant(
            history_review, index + 10_002, "date", precision="year"
        ),
        "plan_year": format_date_variant(
            plan_year, index + 10_003, "date", precision="year"
        ),
        "completed_date": format_date_variant(
            completed_date, index + 10_004, "date", precision="month_day"
        ),
        "todo_date": format_date_variant(
            followup_date, index + 10_005, "date", precision="month_day"
        ),
    }


RELATIVE_PERIOD_POOLS = {
    "future_appointment": [
        "over 1 maand",
        "over een maand",
        "over 6 weken",
        "binnen 3 maanden",
        "binnen drie maanden",
        "na 4 weken",
        "volgende week",
        "over enkele dagen",
    ],
    "deadline_control": [
        "binnen 48 uur",
        "binnen 72 uur",
        "voor het einde van de week",
        "tegen eind volgende maand",
        "uiterlijk binnen 2 weken",
        "ten laatste over 10 dagen",
        "binnen 5 werkdagen",
        "in de loop van volgende maand",
    ],
    "recurrence_interval": [
        "om de 2 jaar",
        "om de twee jaar",
        "elke 6 maanden",
        "elke drie maanden",
        "3-maandelijks",
        "halfjaarlijks",
        "jaarlijks",
        "om de 14 dagen",
        "om de andere week",
        "tweewekelijks",
    ],
    "treatment_duration": [
        "gedurende 2 weken",
        "voor 3 maanden",
        "10 dagen lang",
        "tijdens de komende 4 weken",
        "voor een periode van 6 weken",
        "de eerste 5 dagen",
        "nog 2 maanden",
        "tot over 1 week",
    ],
    "lookback_window": [
        "de afgelopen 2 weken",
        "laatste 6 maanden",
        "voorbije 48 uur",
        "sinds 3 dagen",
        "al 2 weken",
        "reeds 4 jaar",
        "in de voorbije 3 maanden",
        "sinds vorige week",
    ],
    "monitoring_window": [
        "na 1 week",
        "na twee weken",
        "na 3 maanden",
        "na 1 jaar",
        "na een half jaar",
        "over 2 tot 3 weken",
        "binnen 6 tot 8 weken",
        "na 7 dagen",
    ],
    "pregnancy_duration": [
        "zwangerschapsduur 32w5d",
        "zwangerschapsduur 32 w 5 d",
        "amenorroeduur 28w3d",
        "graviditeitsduur 12 weken en 4 dagen",
        "AD 32+5 weken",
        "GA 20w0d",
        "termijn 39w0d",
        "32 weken 5 dagen zwanger",
        "32 5/7 weken",
        "PML 16w2d",
    ],
}


def _date_periods(index: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, pool in RELATIVE_PERIOD_POOLS.items():
        seed = _stable_int("date-period", index, key)
        values[key] = pool[seed % len(pool)]
    return values


def _date_times(index: int) -> dict[str, str]:
    keys = [
        "encounter_time",
        "followup_time",
        "collection_time",
        "review_time",
    ]
    values = {}
    minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    for key in keys:
        seed = _stable_int("date-time", index, key)
        hour = 7 + seed % 12
        minute = minutes[(seed // 12) % len(minutes)]
        style = (seed // 144) % 3
        if style == 0:
            values[key] = f"{hour:02d}:{minute:02d}"
        elif style == 1:
            values[key] = f"{hour}u{minute:02d}"
        else:
            values[key] = f"{hour:02d}.{minute:02d}"
    return values


def _birthdate_for_age(rng: random.Random, age: int, encounter_date: date) -> date:
    return birthdate_for_age(rng, age, encounter_date)


def _birthdate_months_before(
    rng: random.Random, encounter_date: date, months: int
) -> date:
    return birthdate_months_before(rng, encounter_date, months)


def _age_profile(
    rng: random.Random,
    index: int,
    encounter_date: date,
    synthea_seed: dict | None,
) -> tuple[str, date]:
    age_group = (synthea_seed or {}).get("synthea_age_group")
    if age_group == "infant":
        infant_style = index % 8
        if infant_style == 6:
            return "1 dag", birthdate_days_before(encounter_date, 1)
        if infant_style == 7:
            return "0 dagen", birthdate_days_before(encounter_date, 0)
        if infant_style in {0, 3}:
            days = 2 + ((index * 5 + rng.randrange(0, 27)) % 27)
            return f"{days} dagen", birthdate_days_before(encounter_date, days)
        if infant_style in {1, 4}:
            weeks = 2 + ((index * 3 + rng.randrange(0, 11)) % 11)
            return f"{weeks} weken", birthdate_weeks_before(rng, encounter_date, weeks)
        months = 2 + ((index * 7 + rng.randrange(0, 10)) % 10)
        return f"{months} maanden", _birthdate_months_before(
            rng, encounter_date, months
        )

    if age_group == "toddler" and index % 3 != 0:
        months = 12 + ((index * 11 + rng.randrange(0, 24)) % 24)
        return f"{months} maanden", _birthdate_months_before(
            rng, encounter_date, months
        )

    if age_group == "preschool_child" and index % 4 in {0, 2}:
        months = 36 + ((index * 13 + rng.randrange(0, 36)) % 36)
        return f"{months} maanden", _birthdate_months_before(
            rng, encounter_date, months
        )

    if age_group == "school_age_child" and index % 12 in {0, 8}:
        months = 72 + ((index * 17 + rng.randrange(0, 24)) % 24)
        return f"{months} maanden", _birthdate_months_before(
            rng, encounter_date, months
        )

    age_ranges = {
        "toddler": (1, 2),
        "preschool_child": (3, 5),
        "school_age_child": (6, 12),
        "adolescent": (13, 17),
        "young_adult": (18, 39),
        "adult": (40, 64),
        "older_adult": (65, 99),
    }
    lower, upper = age_ranges.get(age_group, (18, 90))
    age = lower + (
        (index * 31 + rng.randrange(0, upper - lower + 1)) % (upper - lower + 1)
    )
    return f"{age} jaar", _birthdate_for_age(rng, age, encounter_date)


def _age_text_variant(age_text: str, index: int, synthea_seed: dict | None) -> str:
    age_group = (synthea_seed or {}).get("synthea_age_group")
    return shared_age_text_variant(age_text, index, age_group)


def _age_context_variant(age_text: str, index: int, synthea_seed: dict | None) -> str:
    if "-jarige" in age_text:
        return ""
    if index % 17 == 4:
        return _cycle_pick(
            ["ongeveer", "bijna", "ca.", "rond", "+/-", "±"], index, stride=7
        )
    age_group = (synthea_seed or {}).get("synthea_age_group")
    if age_group == "infant":
        variants = ["", "leeftijd", "lft."]
    elif age_group == "toddler":
        variants = ["", "leeftijd", "lft."]
    else:
        variants = ["", "leeftijd", "lft."]
    return variants[(index + _stable_offset(age_text)) % len(variants)]


def _cycle_pick(values: list, index: int, stride: int = 1):
    if not values:
        raise ValueError("Cannot sample from an empty list.")
    return values[(index * stride) % len(values)]


def _stable_offset(*values: str) -> int:
    text = "|".join(values)
    return sum(
        (position + 1) * ord(character) for position, character in enumerate(text)
    )


def _stable_int(*values: object) -> int:
    text = "|".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


_BLOOD_PRESSURE_RE = re.compile(r"^(?P<systolic>\d{2,3})/(?P<diastolic>\d{2,3})$")
_SCALAR_NUMBER_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def _bounded(
    value: float, minimum: float | None = None, maximum: float | None = None
) -> float:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _deterministic_delta(index: int, key: str, spread: int) -> int:
    return ((index + _stable_offset(key)) % (spread * 2 + 1)) - spread


def _format_numeric_like(original: str, value: float) -> str:
    normalized = original.replace(",", ".")
    decimals = len(normalized.split(".", 1)[1]) if "." in normalized else 0
    if decimals:
        return f"{value:.{decimals}f}"
    return str(int(round(value)))


def _varied_numeric_value(value: str, *, index: int, key: str, unit: str = "") -> str:
    text = str(value)
    blood_pressure = _BLOOD_PRESSURE_RE.fullmatch(text)
    if blood_pressure:
        systolic = int(blood_pressure.group("systolic"))
        diastolic = int(blood_pressure.group("diastolic"))
        systolic += _deterministic_delta(index, f"{key}|sys", 4) * 2
        diastolic += _deterministic_delta(index, f"{key}|dia", 3)
        return f"{int(_bounded(systolic, 85, 230))}/{int(_bounded(diastolic, 45, 135))}"

    if not _SCALAR_NUMBER_RE.fullmatch(text):
        return text

    normalized = text.replace(",", ".")
    base = float(normalized)
    decimals = len(normalized.split(".", 1)[1]) if "." in normalized else 0
    absolute = abs(base)
    if decimals:
        step = 0.1 if absolute < 20 else 0.5
    elif absolute >= 100:
        step = 5
    elif absolute >= 20:
        step = 2
    else:
        step = 1
    varied = base + _deterministic_delta(index, f"{key}|scalar", 3) * step
    lowered = f"{key} {unit}".lower()
    if "/10" in lowered or "score" in lowered:
        varied = _bounded(varied, 0, 10)
    elif unit == "%" and 0 <= base <= 100:
        varied = _bounded(varied, 0, 100)
    elif base >= 0:
        varied = _bounded(varied, 0, None)
    return _format_numeric_like(text, varied)


def _varied_numeric_row(item: dict, *, index: int, offset: int) -> dict:
    row = deepcopy(item)
    if "value" not in row:
        return row
    name = str(row.get("name", "value"))
    unit = str(row.get("unit", ""))
    row["value"] = _varied_numeric_value(
        str(row.get("value", "")),
        index=index + offset,
        key=f"{name}|{unit}",
        unit=unit,
    )
    return row


_PROCEDURE_NAME_VARIANTS = {
    "urinesediment": ["urinesediment", "urinecontrole", "dipstick urine"],
    "rx thorax": ["RX thorax", "thoraxfoto", "controle RX thorax"],
    "wondzorg": ["wondzorg", "verbandzorg", "lokale wondzorg"],
    "controlefoto": ["controlefoto", "postprocedurele RX", "beeldcontrole"],
    "doorverwijzing kinesitherapie": [
        "doorverwijzing kinesitherapie",
        "kineverwijzing",
        "kinesitherapieplan",
    ],
    "controlelabo": ["controlelabo", "herhaallabo", "labo-opvolging"],
    "time-out": ["time-out", "preprocedurele check", "veiligheidscheck"],
    "materiaalcontrole": [
        "materiaalcontrole",
        "telling materiaal",
        "instrumentencontrole",
    ],
}

_PROCEDURE_SNIPPET_VARIANTS = {
    "nitriet positief, leukocyten 2+": [
        "nitriet positief met leukocyten 2+",
        "sediment toont nitrietpositiviteit en leukocyturie 2+",
        "dipstick positief voor nitriet; leukocyten 2+",
    ],
    "geen pleuravocht, discreet basaal infiltraat": [
        "geen pleuravocht; beperkt basaal infiltraat",
        "discreet basaal infiltraat zonder pleuravocht",
        "basaal beperkt infiltraat, geen pleurale vochtcollectie",
    ],
    "spoelen met NaCl en absorberend verband": [
        "spoelen met fysiologisch zout en absorberend verband",
        "NaCl-spoeling gevolgd door absorberend verband",
        "wonde gereinigd en afgedekt met absorberend materiaal",
    ],
    "positie materiaal conform, geen pneumothorax": [
        "materiaalpositie correct; geen pneumothorax",
        "positie van materiaal conform zonder pneumothorax",
        "controlebeeld toont correcte ligging, geen klaplong",
    ],
    "focus op uithouding en valpreventie": [
        "nadruk op conditieopbouw en valpreventie",
        "oefenplan rond uithouding met valpreventie",
        "functionele training met aandacht voor valrisico",
    ],
    "nierfunctie en elektrolyten via huisarts": [
        "nierfunctie en ionogram via huisarts",
        "creatinine en elektrolytencontrole bij huisarts",
        "huisarts volgt nierfunctie en elektrolyten op",
    ],
    "kompressen en instrumenten volledig geteld": [
        "telling van kompressen en instrumenten volledig",
        "instrumenten- en kompressentelling klopt",
        "materiaal geteld zonder discrepantie",
    ],
    "patient, zijde en procedure bevestigd": [
        "identiteit, zijde en procedure bevestigd",
        "patientidentificatie en procedurezijde gecontroleerd",
        "check van patient, zijde en ingreep afgerond",
    ],
}


def _variant_text(
    value: str, variants: dict[str, list[str]], *, index: int, offset: int
) -> str:
    options = (
        variants.get(value)
        or variants.get(value.lower())
        or variants.get(value.casefold())
    )
    if not options:
        return value
    return _cycle_pick(options, index + offset, stride=5)


def _varied_procedure_item(item: dict, *, index: int, offset: int) -> dict:
    row = deepcopy(item)
    if "procedure" in row:
        row["procedure"] = _variant_text(
            str(row["procedure"]), _PROCEDURE_NAME_VARIANTS, index=index, offset=offset
        )
    if "snippet" in row:
        row["snippet"] = _variant_text(
            str(row["snippet"]), _PROCEDURE_SNIPPET_VARIANTS, index=index, offset=offset
        )
    return row


def _medication_display_name(raw_value: str, *, index: int, offset: int) -> str:
    text = str(raw_value)
    lowered = text.lower()
    variant_rules = [
        (
            ("acetaminophen", "hydrocodone"),
            [
                "paracetamol/hydrocodon",
                "paracetamol met hydrocodon",
                "paracetamol-hydrocodon",
            ],
        ),
        (
            ("acetaminophen", "codeine"),
            ["paracetamol/codeine", "paracetamol met codeine", "codeine-paracetamol"],
        ),
        (
            ("sodium fluoride",),
            ["natriumfluoride gel", "fluoride tandgel", "fluoridegel lokaal"],
        ),
        (
            ("amoxicillin", "clavulanate"),
            ["amoxicilline/clavulaanzuur", "amoxicilline-clavulaanzuur", "amoxiclav"],
        ),
        (
            ("amoxicillin",),
            ["amoxicilline", "amoxicilline p.o.", "amoxicilline schema"],
        ),
        (
            ("acetaminophen",),
            ["paracetamol", "paracetamol p.o.", "paracetamol zo nodig"],
        ),
        (("ibuprofen",), ["ibuprofen", "ibuprofen p.o.", "ibuprofen zo nodig"]),
        (
            ("sodium chloride",),
            ["fysiologisch serum", "NaCl-oplossing", "zoutoplossing i.v."],
        ),
        (
            ("oxycodone",),
            ["oxycodon", "oxycodon vertraagde vrijstelling", "opioid oxycodon"],
        ),
        (("cefuroxime",), ["cefuroxim", "cefuroxim p.o.", "cefuroxim schema"]),
        (
            ("cholecalciferol",),
            ["vitamine D3", "cholecalciferol", "vitamine D onderhoud"],
        ),
        (
            ("penicillin",),
            ["penicilline V", "fenoxymethylpenicilline", "penicilline schema"],
        ),
        (
            ("lisinopril",),
            ["lisinopril", "lisinopril lage dosis", "ACE-remmer lisinopril"],
        ),
        (
            ("hydrochlorothiazide",),
            [
                "hydrochloorthiazide",
                "thiazidediureticum",
                "hydrochloorthiazide onderhoud",
            ],
        ),
        (("fluoxetine",), ["fluoxetine", "SSRI fluoxetine", "fluoxetine onderhoud"]),
        (
            ("loratadine",),
            ["loratadine", "antihistaminicum loratadine", "loratadine zo nodig"],
        ),
        (
            ("epinephrine",),
            [
                "adrenaline auto-injector",
                "noodpen adrenaline",
                "epinefrine auto-injector",
            ],
        ),
        (
            ("buprenorphine", "naloxone"),
            [
                "buprenorfine/naloxon",
                "substitutie buprenorfine-naloxon",
                "buprenorfine-naloxon sublinguaal",
            ],
        ),
        (
            ("dienogest", "estradiol"),
            [
                "estradiol/dienogest schema",
                "cyclisch estradiol-dienogest",
                "hormonaal combinatiepreparaat",
            ],
        ),
    ]
    for required_terms, variants in variant_rules:
        if all(term in lowered for term in required_terms):
            return _cycle_pick(variants, index + offset, stride=7)

    cleaned = text
    cleaned = re.sub(r"\{[^{}]{0,220}\}", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:MG|MCG|ML|UNT|MEQ|ACTUAT)(?:/[A-Z]+)?\b",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\b(?:Oral|Tablet|Tablets|Capsule|Capsules|Solution|Gel|Pack|Topical|Injection|Injectable|"
        r"Chewable|Sublingual|Extended Release|Delayed Release)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = " ".join(cleaned.replace("/", " / ").split())
    return cleaned or text


def _medication_concept_key(raw_value: str) -> str:
    lowered = str(raw_value).lower()
    concept_terms = [
        ("acetaminophen_hydrocodone", ("acetaminophen", "hydrocodone")),
        ("acetaminophen_codeine", ("acetaminophen", "codeine")),
        ("sodium_fluoride", ("sodium fluoride",)),
        ("amoxicillin_clavulanate", ("amoxicillin", "clavulanate")),
        ("amoxicillin", ("amoxicillin",)),
        ("acetaminophen", ("acetaminophen",)),
        ("ibuprofen", ("ibuprofen",)),
        ("sodium_chloride", ("sodium chloride",)),
        ("oxycodone", ("oxycodone",)),
        ("cefuroxime", ("cefuroxime",)),
        ("cholecalciferol", ("cholecalciferol",)),
        ("penicillin", ("penicillin",)),
        ("lisinopril", ("lisinopril",)),
        ("hydrochlorothiazide", ("hydrochlorothiazide",)),
        ("fluoxetine", ("fluoxetine",)),
        ("loratadine", ("loratadine",)),
        ("epinephrine", ("epinephrine",)),
        ("buprenorphine_naloxone", ("buprenorphine", "naloxone")),
        ("dienogest_estradiol", ("dienogest", "estradiol")),
    ]
    for key, required_terms in concept_terms:
        if all(term in lowered for term in required_terms):
            return key
    return " ".join(lowered.split())


def _normalized_medications(values: list, *, index: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for offset, value in enumerate(values):
        concept_key = _medication_concept_key(str(value))
        if concept_key in seen:
            continue
        display = _medication_display_name(str(value), index=index, offset=offset)
        seen.add(concept_key)
        normalized.append(display)
    return normalized


def _normalized_observation_name(raw_name: str) -> str:
    lowered = raw_name.lower()
    rules = [
        (("pain severity", "0-10"), "pijnscore"),
        (("body height",), "lengte"),
        (("body weight",), "gewicht"),
        (("body mass index",), "BMI"),
        (("diastolic blood pressure",), "diastolische bloeddruk"),
        (("systolic blood pressure",), "systolische bloeddruk"),
        (("heart rate",), "hartfrequentie"),
        (("respiratory rate",), "ademhalingsfrequentie"),
        (("body temperature",), "temperatuur"),
        (("oxygen saturation",), "zuurstofsaturatie"),
        (
            ("head occipital-frontal circumference percentile",),
            "hoofdomtrek percentiel",
        ),
        (("head occipital-frontal circumference",), "hoofdomtrek"),
        (("weight-for-length",), "gewicht-lengte percentiel"),
    ]
    for required_terms, display in rules:
        if all(term in lowered for term in required_terms):
            return display
    cleaned = re.sub(r"\s*\[[^\]]+\]\s*", " ", raw_name)
    cleaned = re.sub(
        r"\s*-\s*(?:reported|measured|calculated)\b", " ", cleaned, flags=re.I
    )
    return " ".join(cleaned.split())


def _normalized_observation_unit(raw_unit: str) -> str:
    unit = str(raw_unit)
    replacements = {
        "{score}": "/10",
        "Cel": "°C",
        "mm[Hg]": "mmHg",
        "kg/m2": "kg/m2",
    }
    return replacements.get(unit, unit)


def _normalized_observations(values: list, *, index: int) -> list[tuple[str, str, str]]:
    normalized: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for offset, item in enumerate(values):
        if isinstance(item, dict):
            name = item.get("name") or item.get("description")
            value = item.get("value", "")
            unit = item.get("unit", "")
        elif isinstance(item, (list, tuple)) and item:
            name = item[0]
            value = item[1] if len(item) > 1 else ""
            unit = item[2] if len(item) > 2 else ""
        else:
            continue
        display_name = _normalized_observation_name(str(name))
        key = display_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        display_value = _varied_numeric_value(
            str(value),
            index=index + offset,
            key=f"synthea|{display_name}|{unit}",
            unit=str(unit),
        )
        if display_name in {
            "diastolische bloeddruk",
            "systolische bloeddruk",
        } and _SCALAR_NUMBER_RE.fullmatch(display_value):
            display_value = str(int(round(float(display_value.replace(",", ".")))))
        normalized.append(
            (
                display_name,
                display_value,
                _normalized_observation_unit(str(unit)),
            )
        )
    return normalized


def _medication_order_variant(order: str, *, index: int, offset: int) -> str:
    variants = {
        "2 dd 1": ["2 dd 1", "tweemaal daags 1", "1 tablet ochtend en avond"],
        "zo nodig max 3/d": [
            "zo nodig max 3/d",
            "zo nodig, maximaal driemaal per dag",
            "PRN tot 3 innames/dag",
        ],
        "stop NSAID": ["stop NSAID", "NSAID tijdelijk staken", "geen NSAID verder"],
        "herstart na controlelabo": [
            "herstart na controlelabo",
            "hernemen na labo-opvolging",
            "opnieuw starten na labocontrole",
        ],
        "opbouwen per week": [
            "opbouwen per week",
            "wekelijks titreren",
            "stapsgewijs ophogen",
        ],
        "halve dosis 5 dagen": [
            "halve dosis 5 dagen",
            "5 dagen halve dosis",
            "tijdelijk halve dosis",
        ],
        "verderzetten huidig schema": [
            "verderzetten huidig schema",
            "huidig schema behouden",
            "ongewijzigd verderzetten",
        ],
        "controle therapietrouw": [
            "controle therapietrouw",
            "innamepatroon controleren",
            "therapietrouw navragen",
        ],
        "profylaxe volgens schema": [
            "profylaxe volgens schema",
            "profylactisch schema volgen",
            "preventief volgens voorschrift",
        ],
        "pijnstilling trap 1": [
            "pijnstilling trap 1",
            "basisanalgesie",
            "WHO-trap 1 analgesie",
        ],
        "antico hernemen morgen": [
            "antico hernemen morgen",
            "anticoagulatie morgen hervatten",
            "morgen herstart antico",
        ],
        "verband droog houden": [
            "verband droog houden",
            "verband droog laten",
            "wondverband droog houden",
        ],
    }
    options = variants.get(order)
    if not options:
        return order
    return _cycle_pick(options, index + offset, stride=5)


def _style_profile(index: int, document_type: str) -> dict:
    profile = deepcopy(
        _cycle_pick(STYLE_PROFILES, index + _stable_offset(document_type), stride=11)
    )
    profile.setdefault("name", f"profile-{index % max(1, len(STYLE_PROFILES))}")
    profile.setdefault("document_type_hint", document_type)
    profile.setdefault("common_abbreviations", [])
    profile.setdefault("belgian_terms", [])
    profile.setdefault("numeric_density", "medium")
    profile.setdefault("abbreviation_rate", 0.2)
    return profile


def _selected(items: list, index: int, count: int, stride: int = 1) -> list:
    if not items or count <= 0:
        return []
    selected = []
    seen = set()
    for offset in range(len(items)):
        item = deepcopy(_cycle_pick(items, index + offset, stride=stride))
        marker = (
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            if isinstance(item, (dict, list))
            else str(item)
        )
        if marker in seen:
            continue
        selected.append(item)
        seen.add(marker)
        if len(selected) == count:
            break
    return selected


def _selected_randomized(items: list, seed: int, count: int) -> list:
    if not items or count <= 0:
        return []
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    selected = []
    seen = set()
    for item_index in indices:
        item = deepcopy(items[item_index])
        marker = (
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            if isinstance(item, (dict, list))
            else str(item)
        )
        if marker in seen:
            continue
        selected.append(item)
        seen.add(marker)
        if len(selected) == count:
            break
    return selected


def _variable_count(seed: int, minimum: int, maximum: int, available: int) -> int:
    if available <= 0 or maximum <= 0:
        return 0
    lower = min(max(0, minimum), available)
    upper = min(max(lower, maximum), available)
    return lower + seed % (upper - lower + 1)


def _nonempty_pool_values(pool: dict, key: str, fallback: list) -> list:
    values = pool.get(key)
    return values if isinstance(values, list) and values else fallback


def _numeric_result_count(style_profile: dict) -> int:
    density = str(style_profile.get("numeric_density", "medium")).lower()
    if density == "high":
        return 5
    if density == "low":
        return 2
    return 3


def _procedure_snippet_count(index: int, style_profile: dict, pool: dict) -> int:
    snippets = pool.get("procedure_snippets", [])
    if not isinstance(snippets, list) or not snippets:
        return 0
    document_type = str(style_profile.get("document_type_hint", "")).lower()
    sentence_style = str(style_profile.get("sentence_style", "")).lower()
    if (
        document_type
        in {"device_implant_note", "radiology_summary", "pathology_report"}
        or sentence_style == "radiology"
    ):
        pattern = [1, 2, 1, 3, 0, 2, 1, 2]
    elif sentence_style in {"fragmented", "semi_structured", "handover"}:
        pattern = [0, 1, 2, 2, 0, 1, 3, 1]
    else:
        pattern = [0, 1, 2, 0, 3, 1, 1, 2]
    offset = _stable_offset(str(pool.get("name", "")), document_type, sentence_style)
    return min(pattern[(index + offset) % len(pattern)], len(snippets))


def _clinical_numeric_rows(
    condition: dict, style_profile: dict
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lab in condition.get("labs", [])[: _numeric_result_count(style_profile)]:
        if isinstance(lab, dict):
            name = lab.get("name")
            value = lab.get("value")
            unit = lab.get("unit", "")
        elif isinstance(lab, (list, tuple)) and len(lab) >= 2:
            name = lab[0]
            value = lab[1]
            unit = lab[2] if len(lab) > 2 else ""
        else:
            continue
        if name is None or value is None:
            continue
        rows.append(
            {
                "name": str(name),
                "value": str(value),
                "unit": str(unit),
                "source": "condition",
            }
        )
    return rows


def _medical_detail_phase(
    index: int, condition: dict, style_profile: dict, pool: dict, key: str
) -> int:
    return _stable_int(
        "medical_details",
        index,
        condition.get("name", ""),
        style_profile.get("name", ""),
        style_profile.get("document_type_hint", ""),
        pool.get("name", ""),
        key,
    )


def _medical_details(
    index: int,
    condition: dict,
    style_profile: dict,
) -> dict:
    pool_seed = _stable_int(
        "medical_detail_pool",
        index,
        condition.get("name", ""),
        style_profile.get("name", ""),
        style_profile.get("document_type_hint", ""),
    )
    pool = deepcopy(MEDICAL_DETAIL_POOLS[pool_seed % len(MEDICAL_DETAIL_POOLS)])
    abbreviation_index = _medical_detail_phase(
        index, condition, style_profile, pool, "abbreviations"
    )
    style_abbreviation_index = _medical_detail_phase(
        index, condition, style_profile, pool, "style_profile_abbreviations"
    )
    numeric_index = _medical_detail_phase(
        index, condition, style_profile, pool, "numeric_findings"
    )
    procedure_index = index // max(
        1, len(MEDICAL_DETAIL_POOLS)
    ) + _medical_detail_phase(
        index, condition, style_profile, pool, "procedure_snippets"
    )
    timeframe_index = _medical_detail_phase(
        index, condition, style_profile, pool, "timeframe"
    )
    severity_index = _medical_detail_phase(
        index, condition, style_profile, pool, "severity"
    )
    comorbidity_index = _medical_detail_phase(
        index, condition, style_profile, pool, "comorbidities"
    )
    phrase_index = _medical_detail_phase(
        index, condition, style_profile, pool, "clinical_phrases"
    )
    vitals_index = _medical_detail_phase(
        index, condition, style_profile, pool, "vitals"
    )
    catalog_index = _medical_detail_phase(
        index, condition, style_profile, pool, "pixelpharma_medicines"
    )
    eponym_index = _medical_detail_phase(
        index, condition, style_profile, pool, "ensie_medical_eponyms"
    )

    abbreviation_pool = _selected_randomized(
        pool.get("abbreviations", []), abbreviation_index, 3
    )
    abbreviation_pool.extend(
        {"abbreviation": str(item), "source": "style_profile"}
        for item in _selected_randomized(
            style_profile.get("common_abbreviations", []), style_abbreviation_index, 2
        )
    )
    abbreviation_count = max(
        2,
        min(
            len(abbreviation_pool),
            round(3 + float(style_profile.get("abbreviation_rate", 0.2)) * 10),
        ),
    )
    medications = [str(item) for item in condition.get("medications", [])[:3]]
    medication_order_pool = _nonempty_pool_values(
        pool, "medication_orders", ["volgens schema"]
    )
    medication_orders = []
    for medication_offset, medication in enumerate(medications):
        order_index = _medical_detail_phase(
            index,
            condition,
            style_profile,
            pool,
            f"medication_order_{medication_offset}",
        )
        order = _medication_order_variant(
            _cycle_pick(medication_order_pool, order_index),
            index=index,
            offset=medication_offset,
        )
        medication_orders.append(f"{medication}: {order}")
    numeric_results = _clinical_numeric_rows(condition, style_profile)
    numeric_finding_count = _variable_count(
        numeric_index, 1, 3, len(pool.get("numeric_findings", []))
    )
    pool_numeric_results = [
        _varied_numeric_row(item, index=index, offset=numeric_index + item_index)
        for item_index, item in enumerate(
            _selected_randomized(
                pool.get("numeric_findings", []), numeric_index, numeric_finding_count
            )
        )
    ]
    numeric_results.extend(pool_numeric_results)
    procedure_count = _procedure_snippet_count(procedure_index, style_profile, pool)
    comorbidity_count = _variable_count(
        comorbidity_index, 1, 3, len(pool.get("comorbidities", []))
    )
    phrase_count = _variable_count(
        phrase_index, 1, 3, len(pool.get("clinical_phrases", []))
    )
    vitals_base_count = _numeric_result_count(style_profile)
    vitals_count = _variable_count(
        vitals_index,
        max(1, vitals_base_count - 1),
        vitals_base_count + 1,
        len(pool.get("vitals", [])),
    )
    catalog_count = _variable_count(catalog_index, 2, 4, len(PIXELPHARMA_MEDICINES))
    eponym_count = _variable_count(eponym_index, 3, 5, len(ENSIE_MEDICAL_EPONYMS))
    return {
        "detail_pool": pool.get("name", "default"),
        "timeframe": _cycle_pick(
            _nonempty_pool_values(pool, "timeframes", ["recent"]),
            timeframe_index,
            stride=5,
        ),
        "severity": _cycle_pick(
            _nonempty_pool_values(pool, "severity", ["matig"]), severity_index, stride=3
        ),
        "comorbidities": _selected_randomized(
            pool.get("comorbidities", []), comorbidity_index, comorbidity_count
        ),
        "procedure_snippets": [
            _varied_procedure_item(
                item, index=index, offset=procedure_index + item_index
            )
            for item_index, item in enumerate(
                _selected_randomized(
                    pool.get("procedure_snippets", []), procedure_index, procedure_count
                )
            )
        ],
        "clinical_phrases": _selected_randomized(
            pool.get("clinical_phrases", []), phrase_index, phrase_count
        ),
        "abbreviations": abbreviation_pool[:abbreviation_count],
        "vitals": [
            _varied_numeric_row(item, index=index, offset=vitals_index + item_index)
            for item_index, item in enumerate(
                _selected_randomized(pool.get("vitals", []), vitals_index, vitals_count)
            )
        ],
        "numeric_findings": deepcopy(numeric_results),
        "numeric_results": numeric_results,
        "medication_orders": medication_orders,
        "catalog_medications": _selected_randomized(
            PIXELPHARMA_MEDICINES, catalog_index, catalog_count
        ),
        "medical_eponyms": _selected_randomized(
            ENSIE_MEDICAL_EPONYMS, eponym_index, eponym_count
        ),
        "result_format": style_profile.get("result_format", "mixed"),
    }


def _department_for_condition(condition: dict, index: int) -> str:
    departments = condition.get("departments") or DEPARTMENTS
    return departments[index % len(departments)]


def _identifiers(rng: random.Random, birth_date: date | None = None) -> dict:
    patient_ids = patient_identifier_bundle(rng)
    return {
        "patient_number": patient_number(rng),
        "his_patient_id": his_patient_id(rng),
        "national_register": national_register(rng, birth_date=birth_date),
        "lab_accession": patient_ids[0],
        "pathology_accession": patient_ids[1],
        "imaging_key": patient_ids[2],
        "operating_room_case": patient_ids[3],
        "cfdna_reference": patient_ids[4],
        "study_name": patient_ids[5],
        "study_reference": patient_ids[6],
        "study_protocol_id": study_protocol_identifier(rng),
        "study_protocol_name": study_protocol_name(rng),
        "crisis_card": patient_ids[7],
        "device_serial": patient_ids[8],
        "patient_file": patient_ids[9],
        "caregiver_registry": caregiver_id(rng),
        "material_lot": material_lot_number(rng),
    }


def _condition_for_synthea_seed(
    base_condition: dict, synthea_seed: dict | None, index: int
) -> dict:
    condition = dict(base_condition)
    if not synthea_seed:
        return condition

    age_group = synthea_seed.get("synthea_age_group")
    if synthea_seed.get("condition"):
        condition["name"] = synthea_seed["condition"]
    elif age_group in PEDIATRIC_FALLBACK_CONDITIONS:
        condition = dict(PEDIATRIC_FALLBACK_CONDITIONS[age_group])

    condition["departments"] = synthea_seed.get("departments") or condition.get(
        "departments"
    )
    if synthea_seed.get("medications"):
        condition["medications"] = _normalized_medications(
            synthea_seed["medications"], index=index
        )
    if synthea_seed.get("observations"):
        condition["labs"] = _normalized_observations(
            synthea_seed["observations"], index=index
        )
    return condition


def _is_pediatric_seed(synthea_seed: dict | None) -> bool:
    return (synthea_seed or {}).get("synthea_age_group") in PEDIATRIC_AGE_GROUPS


def _pediatric_synthea_pools(seeds: list[dict]) -> dict[str, list[dict]]:
    pediatric = [seed for seed in seeds if _is_pediatric_seed(seed)]
    pools: dict[str, list[dict]] = {}
    for group in PEDIATRIC_GROUP_ORDER:
        group_seeds = [
            seed for seed in pediatric if seed.get("synthea_age_group") == group
        ]
        with_conditions = [seed for seed in group_seeds if seed.get("condition")]
        without_conditions = [seed for seed in group_seeds if not seed.get("condition")]
        pools[group] = with_conditions + without_conditions
    return pools


def _profession_for_age_group(index: int, synthea_seed: dict | None) -> str:
    age_group = (synthea_seed or {}).get("synthea_age_group")
    activities = PEDIATRIC_ACTIVITY_BY_GROUP.get(age_group)
    if activities:
        return _cycle_pick(activities, index, stride=3)
    return _cycle_pick(PROFESSIONS, index, stride=29)


def _administrative_gender(index: int, synthea_seed: dict | None) -> str:
    raw_gender = (synthea_seed or {}).get("gender") or (synthea_seed or {}).get("sex")
    if isinstance(raw_gender, str):
        normalized = raw_gender.strip().lower()
        if normalized in {"m", "male", "man", "masculin"}:
            source_options = ["M", "m", "man", "mannelijk"]
        elif normalized in {"f", "female", "vrouw", "v", "feminin"}:
            source_options = ["V", "v", "vrouw", "vrouwelijk"]
        else:
            source_options = [raw_gender.strip()]
        return _cycle_pick(source_options, index, stride=3)

    options = [
        "M",
        "V",
        "m",
        "v",
        "man",
        "vrouw",
        "X",
        "non-binair",
        "onbekend",
        "niet vermeld",
    ]
    return _cycle_pick(options, index, stride=17)


def _other_org_for_age_group(index: int, synthea_seed: dict | None) -> str:
    age_group = (synthea_seed or {}).get("synthea_age_group")
    organizations = PEDIATRIC_ORGS_BY_GROUP.get(age_group)
    if organizations:
        return _cycle_pick(organizations, index, stride=5)
    return _cycle_pick(OTHER_ORGS, index, stride=43)


def _coverage_targets(index: int, document_type: str) -> list[str]:
    """Select a small, realistic PII subset to improve label balance at scale."""

    targets = ["patient.name", "encounter_date"]

    if index % 2 == 0:
        targets.append("patient_address.text")
    if index % 3 == 0:
        targets.append("profession")
    if index % 4 == 0:
        targets.extend(["hospital", "caregiver.name"])
    if index % 5 == 0:
        targets.extend(["caregiver_locality", "identifiers.caregiver_registry"])
    if index % 5 == 1:
        targets.append("healthcare_institution")
    if index % 6 == 0:
        targets.append("other_location")
    if index % 7 == 0:
        targets.append("other_org")
    if index % 3 == 1:
        targets.append("contact.patient_phone")
    if index % 4 == 2:
        targets.append("contact.caregiver_internal_phone")
    if index % 4 == 3:
        targets.append("relative.name")
    if index % 6 == 2:
        targets.append("date_overview.history_start_year")
    if index % 6 == 4:
        targets.append("date_overview.todo_date")
    if _is_date_focus_case(index):
        targets.extend(
            [
                "date_focus.aanmelding_numeric_slash_long",
                "date_focus.staal_numeric_dot_long",
                "date_focus.consult_textual_full",
                "date_focus.mdo_weekday_numeric",
                "date_focus.antecedenten_month_year_1",
                "date_focus.voorgeschiedenis_year_1",
                "date_focus.dagplan_day_month_numeric",
                "date_focus.lange_termijn_year_only",
            ]
        )

    document_type_targets = {
        "lab_report": ["identifiers.lab_accession", "healthcare_institution"],
        "genetics_report": ["identifiers.cfdna_reference", "relative.name"],
        "radiology_summary": ["identifiers.imaging_key", "hospital"],
        "pathology_report": ["identifiers.pathology_accession", "caregiver.name"],
        "device_implant_note": [
            "identifiers.device_serial",
            "identifiers.material_lot",
            "hospital",
        ],
        "consult_letter": [
            "caregiver.name",
            "contact.caregiver_internal_phone",
            "healthcare_institution",
            "profession",
        ],
        "discharge_summary": ["caregiver.name", "caregiver_locality", "hospital"],
        "referral_letter": [
            "caregiver.name",
            "contact.caregiver_internal_phone",
            "identifiers.caregiver_registry",
            "healthcare_institution",
        ],
        "home_care_report": [
            "patient_address.text",
            "contact.relative_email",
            "healthcare_institution",
        ],
        "rehab_progress": ["profession", "other_org", "healthcare_institution"],
        "nursing_note": [
            "contact.relative_phone",
            "contact.caregiver_internal_phone",
            "hospital",
        ],
        "medication_reconciliation": ["contact.relative_email", "relative.name"],
        "ed_note": [
            "contact.patient_phone",
            "identifiers.national_register",
            "relative.name",
        ],
        "oncology_mdo": [
            "identifiers.study_name",
            "identifiers.study_protocol_id",
            "identifiers.study_protocol_name",
            "hospital",
            "caregiver.name",
        ],
        "ai_scribe_note": ["profession", "contact.patient_email"],
    }
    targets.extend(document_type_targets.get(document_type, []))

    deduped = []
    for target in targets:
        if target not in deduped:
            deduped.append(target)
    max_targets = 10 if _is_date_focus_case(index) else 8
    if len(deduped) > max_targets:
        original_order = {target: position for position, target in enumerate(deduped)}
        priority_order = [
            "patient.name",
            "encounter_date",
            "date_focus.aanmelding_numeric_slash_long",
            "date_focus.staal_numeric_dot_long",
            "date_focus.consult_textual_full",
            "date_focus.mdo_weekday_numeric",
            "date_focus.antecedenten_month_year_1",
            "date_focus.voorgeschiedenis_year_1",
            "date_focus.dagplan_day_month_numeric",
            "date_focus.lange_termijn_year_only",
            "date_overview.history_start_year",
            "date_overview.todo_date",
            "patient_address.text",
            "caregiver_locality",
            "other_location",
            "identifiers.caregiver_registry",
            "identifiers.material_lot",
            "profession",
            "other_org",
            "hospital",
            "healthcare_institution",
            "contact.patient_phone",
            "contact.patient_email",
            "contact.relative_phone",
            "contact.relative_email",
            "contact.caregiver_internal_phone",
            "relative.name",
            "caregiver.name",
        ]
        priority = {target: position for position, target in enumerate(priority_order)}
        keep = sorted(
            deduped,
            key=lambda target: (
                priority.get(target, len(priority) + original_order[target]),
                original_order[target],
            ),
        )[:max_targets]
        deduped = sorted(keep, key=original_order.get)
    return deduped


def generate_case(
    sampler: LookupSampler,
    rng: random.Random,
    index: int,
    synthea_seed: dict | None = None,
) -> ClinicalCase:
    condition = _condition_for_synthea_seed(
        _cycle_pick(CONDITIONS, index, stride=37),
        synthea_seed,
        index,
    )
    patient = _person(sampler)
    relative = _person(sampler)
    encounter = _encounter_date(rng)
    followup = _followup_date(rng, encounter)
    age_text, birth_date = _age_profile(rng, index, encounter, synthea_seed)
    age_text = _age_text_variant(age_text, index, synthea_seed)
    age_context = _age_context_variant(age_text, index, synthea_seed)
    document_type = DOCUMENT_TYPES[index % len(DOCUMENT_TYPES)]
    style_profile = _style_profile(index, document_type)
    medical_details = _medical_details(index, condition, style_profile)
    birthdate_index = index + 2

    return ClinicalCase(
        document_type=document_type,
        language=LANGUAGES[index % len(LANGUAGES)],
        department=_department_for_condition(condition, index),
        condition=condition,
        patient=patient,
        caregiver=_person(sampler),
        secondary_caregiver=_person(sampler),
        relative=relative,
        patient_address=_address(sampler, rng),
        hospital=sampler.hospital(),
        healthcare_institution=sampler.healthcare_institution(),
        caregiver_locality=sampler.locality(),
        other_location=sampler.locality(),
        profession=_profession_for_age_group(index, synthea_seed),
        other_org=_other_org_for_age_group(index, synthea_seed),
        age_text=age_text,
        age_context=age_context,
        birthdate=_format_date_variant(birth_date, birthdate_index, "birthdate"),
        birthdate_prefix=_date_context_prefix(birthdate_index, "birthdate"),
        encounter_date=_format_date_variant(encounter, index, "date"),
        followup_date=_format_date_variant(followup, index + 1, "date"),
        genetic_finding=dict(_cycle_pick(GENETIC_FINDINGS, index, stride=17)),
        identifiers=_identifiers(rng, birth_date=birth_date),
        contact={
            "patient_phone": phone(rng),
            "patient_email": email(patient.name, rng),
            "relative_phone": phone(rng),
            "relative_email": email(relative.name, rng),
            "caregiver_internal_phone": internal_phone(rng),
        },
        hard_negatives=hard_negative_codes(rng),
        note_style=_cycle_pick(NOTE_STYLES, index, stride=13),
        synthea_source=synthea_seed,
        coverage_targets=_coverage_targets(index, document_type),
        style_profile=style_profile,
        medical_details=medical_details,
        date_overview=_date_overview(encounter, followup, index),
        date_times=_date_times(index),
        date_periods=_date_periods(index),
        date_focus=_date_focus(encounter, index),
        date_focus_template=_date_focus_template(index),
        date_focus_style=_date_focus_style(index),
        administrative_gender=_administrative_gender(index, synthea_seed),
    )


def case_to_record(case_id: str, case: ClinicalCase) -> dict:
    record = asdict(case)
    record["case_id"] = case_id
    record["pii_policy"] = {
        "source": "Belgian lookup lists and generated Belgian identifiers",
        "note": "Synthea clinical content is used only for medical story/facts; Synthea PII is not used.",
    }
    record["annotation_policy"] = {
        "true_pii_code_examples": [
            "patient numbers",
            "HIS patient IDs",
            "national register numbers",
            "lab/pathology/imaging accessions",
            "patient-linked report files",
            "study protocol references",
            "study protocol IDs and patient-linked study protocol names",
            "CFDNA references",
            "patient-linked implanted-device serials",
            "caregiver registry IDs",
            "material lot numbers linked to caregiver/institution material",
        ],
        "hard_negative_code_examples": [
            "gene names",
            "biomarker names",
            "HGVS variants",
            "rsIDs",
            "LOINC/SNOMED/ICD terminology codes",
            "generic device model names",
        ],
        "span_boundary_examples": [
            "age approximation words such as ongeveer, bijna and ca. stay inside Age_Birthdate spans when sampled; keep them rare",
            "age field labels such as leeftijd stay outside Age_Birthdate spans",
            "age qualifiers such as oud and postnataal stay outside Age_Birthdate spans",
            "birthdate prefixes such as ° stay outside Age_Birthdate spans",
            "times next to dates stay outside Date spans",
            "date_focus slots are deliberately dense Date examples with mixed formats",
            "caregiver initials, first-name-only, surname-only forms and partial names are Name:Caregiver when written",
            "administrative gender or sex markers such as M, V, X, man and vrouw are not PII",
            "HIS templates may split patient first name and surname into separate Name:Patient spans",
            "medication names, formulations, dosages, routes and schedules are clinical content and stay unannotated",
            "study protocol IDs and patient-linked study protocol names are ID:Patient when they identify the patient enrollment or study dossier",
            "doctor titles, specialties and caregiver functions such as HAIO, ASO, cardioloog, huisarts, spoedarts, gynaecoloog, pediater, verpleegkundig specialist and klinisch apotheker are not Profession",
        ],
    }
    return record


def case_from_record(record: dict) -> ClinicalCase:
    return ClinicalCase(
        document_type=record["document_type"],
        language=record["language"],
        department=record["department"],
        condition=record["condition"],
        patient=PersonProfile(**record["patient"]),
        caregiver=PersonProfile(**record["caregiver"]),
        secondary_caregiver=PersonProfile(**record["secondary_caregiver"]),
        relative=PersonProfile(**record["relative"]),
        patient_address=AddressProfile(**record["patient_address"]),
        hospital=tuple(record["hospital"]),
        healthcare_institution=tuple(record["healthcare_institution"]),
        caregiver_locality=tuple(record["caregiver_locality"]),
        other_location=tuple(record["other_location"]),
        profession=record["profession"],
        other_org=record["other_org"],
        age_text=record["age_text"],
        age_context=record.get("age_context", ""),
        birthdate=record["birthdate"],
        birthdate_prefix=record.get("birthdate_prefix", ""),
        encounter_date=record["encounter_date"],
        followup_date=record["followup_date"],
        genetic_finding=record["genetic_finding"],
        identifiers=record["identifiers"],
        contact=record["contact"],
        hard_negatives=record["hard_negatives"],
        note_style=record.get("note_style", "compacte SOAP-nota"),
        synthea_source=record.get("synthea_source"),
        coverage_targets=record.get("coverage_targets"),
        style_profile=record.get("style_profile"),
        medical_details=record.get("medical_details"),
        date_overview=record.get("date_overview"),
        date_times=record.get("date_times"),
        date_periods=record.get("date_periods"),
        date_focus=record.get("date_focus"),
        date_focus_template=record.get("date_focus_template"),
        date_focus_style=record.get("date_focus_style"),
        administrative_gender=record.get("administrative_gender", ""),
    )


def generate_case_records(
    count: int,
    seed: int = 20260508,
    synthea_seeds: list[dict] | None = None,
    start_index: int = 0,
) -> list[dict]:
    sampler = LookupSampler(seed=seed)
    rng = random.Random(seed)
    seeds = synthea_seeds or []
    pediatric_pools = _pediatric_synthea_pools(seeds)
    has_pediatric_pool = any(pediatric_pools.values())
    pediatric_target = (
        math.ceil(count * PEDIATRIC_TARGET_RATIO) if has_pediatric_pool else 0
    )
    pediatric_count = 0
    pediatric_group_counts = {group: 0 for group in PEDIATRIC_GROUP_ORDER}
    pediatric_cursors = {
        group: start_index % len(pool) if pool else 0
        for group, pool in pediatric_pools.items()
    }
    records = []
    for local_index in range(count):
        index = start_index + local_index
        synthea_seed = seeds[index % len(seeds)] if seeds else None
        desired_pediatric_count = (
            math.floor(((local_index + 1) * pediatric_target) / count) if count else 0
        )
        if (
            has_pediatric_pool
            and not _is_pediatric_seed(synthea_seed)
            and pediatric_count < desired_pediatric_count
        ):
            group = min(
                (group for group in PEDIATRIC_GROUP_ORDER if pediatric_pools[group]),
                key=lambda candidate: (
                    pediatric_group_counts[candidate],
                    PEDIATRIC_GROUP_ORDER.index(candidate),
                ),
            )
            pool = pediatric_pools[group]
            synthea_seed = dict(pool[pediatric_cursors[group] % len(pool)])
            synthea_seed["synthea_balanced_resample"] = True
            pediatric_cursors[group] += 1
        if _is_pediatric_seed(synthea_seed):
            pediatric_count += 1
            pediatric_group_counts[synthea_seed["synthea_age_group"]] += 1
        case = generate_case(sampler, rng, index, synthea_seed=synthea_seed)
        records.append(case_to_record(f"case-{index + 1:05d}", case))
    return records
