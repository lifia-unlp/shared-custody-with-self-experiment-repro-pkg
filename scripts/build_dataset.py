#!/usr/bin/env python3
"""Build the cleaned, English, pseudonymized dataset from the raw data.

Source:  the raw Google Forms export (Spanish). This file contains free-text
         comments and exact timestamps and is NOT distributed with this
         package; it is available from the authors on reasonable request.
Output:  data/shared-custody-sus-experiment.csv

Usage:
  uv run python scripts/build_dataset.py /path/to/resultados-experimento.csv

Transformations applied:
  - rows sorted by submission timestamp; participants pseudonymized P01..P67
  - timestamps reduced to the experiment day (1..3) for privacy
  - headers and categorical values translated to English
  - Likert labels ("1 (Muy en desacuerdo)", "5 (Totalmente de acuerdo)")
    parsed to integers 1..5 WITHOUT any capping
  - SUS score computed as (sum(odd-1) + sum(5-even)) * 2.5   (0..100)
  - security score computed as ((sum(odd)-3) + (15-sum(even))) * 2.5  (0..60).
    The score is computed only for participants who answered all six items;
    otherwise it is left empty. One participant (P53, treatment 2) answered
    five of the six items and therefore has no score, matching the exclusion
    rule stated in the paper; the number of answered items is recorded in
    security_items_answered so analysts can apply a different rule.
  - free-text comment column dropped for privacy (available on request)

Run from anywhere; the output path is resolved relative to this file.
"""
import csv
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "shared-custody-sus-experiment.csv")

MAP_TREAT = {
    "Utilice sólo un dispositivo móvil":
        (1, "Single-signature (1-of-1), mobile device only"),
    "Utilice dos dispositivos - Iniciando en la PC":
        (2, "Multi-signature (2-of-2), initiated on desktop, authorized on mobile"),
    "Utilice dos dispositivos - Iniciando en el móvil":
        (3, "Multi-signature (2-of-2), initiated on mobile, authorized on desktop"),
    "Utilice solo la PC":
        (4, "Single-signature (1-of-1), desktop browser only"),
}
MAP_GENDER = {"Hombre": "Man", "Mujer": "Woman", "": ""}
MAP_EDU = {"Primario": "Primary", "Secundario": "Secondary",
           "Terciario / Universitario": "Tertiary/University",
           "Posgrado": "Postgraduate", "": ""}
MAP_NAT = {"Argentina": "Argentina", "Bolivia": "Bolivia", "Brasil": "Brazil",
           "Colombia": "Colombia", "Italia": "Italy", "Paraguay": "Paraguay",
           "Venezuela": "Venezuela", "": ""}
MAP_OCC = {"Empleado": "Employed", "Estudiante": "Student",
           "Estudiante, Desempleado": "Student, Unemployed",
           "Estudiante, Empelador": "Student, Employer",
           "Estudiante, Empleado": "Student, Employed",
           "Estudiante, Jubilado": "Student, Retired",
           "Estudiante, Trabajador independiente": "Student, Self-employed",
           "": ""}
MAP_EXP = {"": "", "Menos de 1 año": "Less than 1 year", "1 a 3 años": "1-3 years",
           "5 a 6 años": "5-6 years", "7 a 9 años": "7-9 years",
           "10 o mas años": "10+ years"}
MAP_FREQ = {"": "", "A diario": "Daily",
            "Algunas veces a la semana": "A few times a week",
            "Algunas veces al mes": "A few times a month",
            "Casi nunca": "Almost never", "Nunca": "Never"}

ROLES = ["software_developer", "qa", "architect", "analyst",
         "support", "teacher", "student", "researcher"]


def likert(v):
    """Parse '1 (Muy en desacuerdo)' / '5 (Totalmente de acuerdo)' / '3' -> int."""
    v = v.strip()
    return int(v[0]) if v else ""


def norm_devices(v):
    v = v.strip()
    if v == "":
        return ""
    low = v.lower()
    if low in ("2", "dos", "dos dispositivos") or "dos dispositivos" in low:
        return "two"
    if low in ("uno", "1", "un dispositivo"):
        return "one"
    return "unclear"


def sus_score(items):
    return round((sum(x - 1 for x in items[0::2])
                  + sum(5 - x for x in items[1::2])) * 2.5, 1)


def security_score(items):
    """Score 0..60, only defined for a complete six-item response."""
    return round(((sum(items[0::2]) - 3) + (15 - sum(items[1::2]))) * 2.5, 1)


def build(raw):
    with open(raw, newline="", encoding="utf-8-sig") as f:
        data = [[c.strip() for c in r] for r in csv.reader(f)][1:]
    data.sort(key=lambda r: datetime.datetime.strptime(r[0], "%d/%m/%Y %H:%M:%S"))

    header = (["participant_id", "day", "treatment", "treatment_description",
               "nationality", "age", "gender", "education_level", "occupation"]
              + [f"it_experience_{r}" for r in ROLES]
              + ["fintech_wallet_use_frequency", "crypto_app_use_frequency",
                 "crypto_self_perceived_knowledge"]
              + [f"sus_q{i}" for i in range(1, 11)] + ["sus_score"]
              + [f"security_q{i}" for i in range(1, 7)]
              + ["security_items_answered", "security_score",
                 "feels_safer_with_devices"])

    out = []
    for n, r in enumerate(data, 1):
        day = {6: 1, 7: 2, 8: 3}[
            datetime.datetime.strptime(r[0], "%d/%m/%Y %H:%M:%S").day]
        tnum, tdesc = MAP_TREAT[r[1]]
        sus = [likert(r[i]) for i in range(19, 29)]
        sec = [likert(r[i]) for i in range(29, 35)]
        out.append(
            [f"P{n:02d}", day, tnum, tdesc, MAP_NAT[r[3]], r[4],
             MAP_GENDER[r[5]], MAP_EDU[r[6]], MAP_OCC[r[7]]]
            + [MAP_EXP[r[8 + i]] for i in range(8)]
            + [MAP_FREQ[r[16]], MAP_FREQ[r[17]], r[18]]
            + sus + [sus_score(sus) if all(v != "" for v in sus) else ""]
            + sec + [sum(v != "" for v in sec),
                     security_score(sec) if all(v != "" for v in sec) else "",
                     norm_devices(r[35])])

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    print(f"wrote {len(out)} rows to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: build_dataset.py /path/to/raw-google-forms-export.csv")
    build(sys.argv[1])
