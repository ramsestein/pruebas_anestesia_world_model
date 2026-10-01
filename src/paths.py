"""paths.py — fuente ÚNICA de rutas y cohortes del proyecto anestesia_world.

Este módulo concentra TODAS las rutas de lectura de datos y las listas de
cohortes. Cualquier código que lea datos debe hacerlo a través de las
constantes de aquí; NO debe haber strings de ruta sueltos.

Reglas:
  - Sin lógica de cómputo: solo constantes y (como mucho) una verificación de
    existencia con mensaje claro.
  - DATA_ROOT se puede sobrescribir con la variable de entorno
    ANESTESIA_DATA_ROOT (para apuntar, p. ej., a la copia de D:).
  - COHORTS contiene SOLO las cohortes VIGENTES (windows_v4 / v7).
  - LOST_COHORTS es un REGISTRO histórico: cohortes y directorios de ventanas
    que ya no existen en disco. Sirve para que el código que deba nombrarlas
    lo haga por una constante con nombre explícito, no por un string suelto.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raíz del repositorio (padre de src/).
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_root() -> Path:
    env = os.environ.get("ANESTESIA_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "data"


# Directorio raíz de datos (sobrescribible con ANESTESIA_DATA_ROOT).
DATA_ROOT = _resolve_data_root()

# ---------------------------------------------------------------------------
# Directorios de artefactos vigentes (bajo DATA_ROOT)
# ---------------------------------------------------------------------------
WINDOWS_DIR = DATA_ROOT / "windows_v4"   # ventanas vigentes (cohortes v7)
TOKENS_DIR = DATA_ROOT / "tokens_v1"
PK_DIR = DATA_ROOT / "pk_v1"
CONTEXT_DIR = DATA_ROOT / "context_v1"
# AE vigente del proyecto. En el paso 2 (fase D) se repunta a ae_v2; antes
# apunta a ae_v1 (histórico). Véanse AE_V1_DIR y AE_V2_DIR más abajo.
AE_DIR = DATA_ROOT / "ae_v1"
DIAGNOSTICS_DIR = DATA_ROOT / "diagnostics"
AUDIT_DIR = DATA_ROOT / "audit"

# Artefactos de autoencoder con nombre explícito (independientes del repunte
# de AE_DIR): ae_v1 es el histórico (ae_bal/ae_real congelados, snapshots),
# ae_v2 es el nuevo AE entrenado sobre real + v7 en el paso 2.
AE_V1_DIR = DATA_ROOT / "ae_v1"
AE_V2_DIR = DATA_ROOT / "ae_v2"

# Directorios del repo (no de datos).
MANIFESTS_DIR = REPO_ROOT / "manifests"
REPORTS_DIR = REPO_ROOT / "reports"

# ---------------------------------------------------------------------------
# Directorios de ventanas históricos (perdidos) — registro con nombre explícito
# ---------------------------------------------------------------------------
WINDOWS_V2_DIR = DATA_ROOT / "windows_v2"
WINDOWS_V3_DIR = DATA_ROOT / "windows_v3"

# ---------------------------------------------------------------------------
# Cohortes VIGENTES: nombre canónico -> directorio de casos.
# Nota: el directorio en disco de la cohorte vaso es "synthetic_vaso_reinf_v7"
# aunque el nombre canónico de source es "vaso_reinf_v7".
# ---------------------------------------------------------------------------
COHORTS: dict[str, Path] = {
    "real": DATA_ROOT / "real",
    "synthetic_v7": DATA_ROOT / "synthetic_v7",
    "vaso_reinf_v7": DATA_ROOT / "synthetic_vaso_reinf_v7",
    "cf_v7": DATA_ROOT / "cf_v7",
}

# Lista de las tres cohortes sintéticas vigentes.
SYNTH_COHORTS: list[str] = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]

# Todas las cohortes vigentes (real + sintéticas).
ALL_COHORTS: list[str] = ["real"] + SYNTH_COHORTS

# ---------------------------------------------------------------------------
# Cohortes y directorios de ventanas PERDIDOS (solo registro histórico).
# ---------------------------------------------------------------------------
LOST_SYNTH_V5: list[str] = ["synthetic_v5", "vaso_reinf_v5", "cf_v5"]
LOST_SYNTH_V6: list[str] = ["synthetic_v6", "vaso_reinf_v6", "cf_v6"]

LOST_COHORTS: list[str] = (
    LOST_SYNTH_V5 + LOST_SYNTH_V6 + ["windows_v2", "windows_v3"]
)

# Directorios históricos de casos de las cohortes perdidas (registro).
LOST_COHORT_DIRS: dict[str, Path] = {
    "synthetic_v5": DATA_ROOT / "synthetic_v5",
    "vaso_reinf_v5": DATA_ROOT / "synthetic_vaso_reinf_v5",
    "cf_v5": DATA_ROOT / "cf_v5",
    "synthetic_v6": DATA_ROOT / "synthetic_v6",
    "vaso_reinf_v6": DATA_ROOT / "synthetic_vaso_reinf_v6",
    "cf_v6": DATA_ROOT / "cf_v6",
}


# ---------------------------------------------------------------------------
# Verificación de existencia (sin lógica de cómputo)
# ---------------------------------------------------------------------------

def require_vigent_dirs() -> None:
    """Verifica que los directorios de datos vigentes existen; falla con un
    mensaje claro si falta alguno. No se llama en el import: solo bajo demanda.
    """
    missing: list[Path] = []
    for d in (WINDOWS_DIR, TOKENS_DIR, PK_DIR, CONTEXT_DIR, AE_DIR,
              DIAGNOSTICS_DIR):
        if not d.is_dir():
            missing.append(d)
    for name, d in COHORTS.items():
        if not d.is_dir():
            missing.append(d)
    if missing:
        raise RuntimeError(
            "Faltan directorios de datos vigentes (¿falta ANESTESIA_DATA_ROOT "
            "o el dataset no está montado?):\n  " +
            "\n  ".join(str(p) for p in missing))
