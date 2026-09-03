"""Data shim for the co-authors' notebooks.

The notebooks in this directory were written against the sheet "Datos cuanti"
of a private spreadsheet (Wallet-pattern-.xlsx) that also contains
identifying information. This module rebuilds that sheet, with the same
Spanish column names, from the public pseudonymized CSV, so the notebooks run
unchanged apart from the loading line.

It also changes the working directory to output/notebooks/ so that the files
the notebooks write (.tex, .xlsx, .png, .pdf) land there instead of next to
the notebooks.

Differences with respect to the original sheet (see data/CODEBOOK.md):
  - rows are in participant order P01..P67 (the sheet kept the form's export
    order, which swaps two rows; no statistic depends on row order);
  - "seguridad sus" is derived from security_score and is therefore NaN for
    every participant with an incomplete security block: the seven who
    answered no item (the sheet stored 0 for them) and P53, who answered
    five of six (the sheet stored -9, obtained by summing the blank as 0).
    The notebooks exclude all eight rows, matching the paper.
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "shared-custody-sus-experiment.csv"
OUTPUT = ROOT / "output" / "notebooks"

TREATMENT_LABEL = {
    1: "Utilice sólo un dispositivo móvil",
    2: "Utilice dos dispositivos - Iniciando en la PC",
    3: "Utilice dos dispositivos - Iniciando en el móvil",
    4: "Utilice solo la PC",
}
SUS_COLS = [
    "Creo que me gustaría utilizar este sistema con frecuencia",
    "Encontré el sistema innecesariamente complejo",
    "Pienso que el sistema fue fácil de usar",
    "Creo que necesitaría el apoyo de una persona técnica para poder usar este sistema",
    "Descubrí que las diversas funciones de este sistema estaban bien integradas",
    "Pienso que había demasiada inconsistencia en este sistema",
    "Me imagino que la mayoría de la gente aprendería a usar este sistema muy rápidamente",
    "Encontré el sistema muy engorroso de usar",
    "Me sentí muy seguro de usar el sistema",
    "Necesité aprender muchas cosas antes de poder comenzar con este sistema",
]
SEC_COLS = [
    "Siento que este método de envío de criptoactivos es lo suficientemente seguro para valores que representen hasta 1 (un) ingreso mensual",
    "Siento que este método de envío de criptoactivos es no lo suficientemente seguro incluso para valores que representen hasta 2 (dos) ingresos mensuales",
    "Siento que este método de envío de criptoactivos es lo suficientemente seguro para la totalidad del monto de todos mis criptoactivos con este método.",
    "Siento que este método de envío de criptoactivos no es lo suficientemente seguro para todos mis criptoactivos con este método.",
    "Siento que este método de envío de criptoactivos es lo suficientemente seguro independiente del monto que representan.",
    "Siento que este método de envío de criptoactivos no es lo suficientemente seguro independiente del monto que representan.",
]


def read_datos_cuanti(*_args, **_kwargs):
    """Drop-in replacement for pd.read_excel(ARCHIVO, sheet_name="Datos cuanti")."""
    df = pd.read_csv(DATA)
    out = pd.DataFrame({
        "Grupo": df["treatment"].astype(int),
        "¿En qué experiencia participaste?": df["treatment"].map(TREATMENT_LABEL),
    })
    for i, c in enumerate(SUS_COLS, 1):
        out[c] = df[f"sus_q{i}"].astype(int)
    out["SUS UX"] = df["sus_score"].astype(float)
    for i, c in enumerate(SEC_COLS, 1):
        out[c] = df[f"security_q{i}"]
    out["seguridad sus"] = df["security_score"] / 2.5 - 12
    return out


OUTPUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT)
