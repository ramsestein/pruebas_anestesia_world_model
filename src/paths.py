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


def _env_dir(name: str, default: Path) -> Path:
    """Directorio de artefacto sobrescribible con ``ANESTESIA_<NAME>``.

    Se usa SOLO en las regeneraciones del paso 3d (cf_v7_1 -> windows_v4_1,
    pk_v2_1, tokens_v2_1) mientras los directorios vigentes siguen apuntando a
    los artefactos adoptados. Va por variable de entorno y NO por monkeypatch
    porque el entorno SI se hereda entre procesos: los workers de
    ``autoencoder.window`` son procesos nuevos (spawn en Windows) que
    re-importan los módulos y no verían un parche hecho en el padre.
    """
    env = os.environ.get(f"ANESTESIA_{name.upper()}")
    return Path(env).expanduser() if env else default

# ---------------------------------------------------------------------------
# Directorios de artefactos vigentes (bajo DATA_ROOT)
# ---------------------------------------------------------------------------
WINDOWS_DIR = _env_dir("WINDOWS_DIR", DATA_ROOT / "windows_v4")   # ventanas vigentes (cohortes v7)

# Artefactos de tokens con nombre explícito por versión (paso 3). *_DIR
# apunta a la versión vigente; las constantes *_V1_DIR / *_V2_DIR son
# explícitas e independientes del repunte.
PK_V1_DIR = DATA_ROOT / "pk_v1"
PK_V2_DIR = DATA_ROOT / "pk_v2"
CONTEXT_V1_DIR = DATA_ROOT / "context_v1"
CONTEXT_V2_DIR = DATA_ROOT / "context_v2"
TOKENS_V1_DIR = DATA_ROOT / "tokens_v1"
TOKENS_V2_DIR = DATA_ROOT / "tokens_v2"

# Vigentes (paso 3b): pk_v2, context_v2 y tokens_v2 adoptados.
# Los tres son sobrescribibles por entorno durante las regeneraciones del paso
# 3d (ANESTESIA_PK_DIR, ANESTESIA_TOKENS_DIR y ANESTESIA_CONTEXT_DIR).
TOKENS_DIR = _env_dir("TOKENS_DIR", TOKENS_V2_DIR)
PK_DIR = _env_dir("PK_DIR", PK_V2_DIR)
CONTEXT_DIR = _env_dir("CONTEXT_DIR", CONTEXT_V2_DIR)

# AE vigente del proyecto. En el paso 2 (fase D) se repunta a ae_v2; antes
# apunta a ae_v1 (histórico). Véanse AE_V1_DIR y AE_V2_DIR más abajo.
AE_DIR = DATA_ROOT / "ae_v2"
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
    "cf_v7_1": DATA_ROOT / "cf_v7_1",
}

# ---------------------------------------------------------------------------
# Cohorte de CF ACTIVA (paso 3d)
# ---------------------------------------------------------------------------
# cf_v7 es la cohorte ADOPTADA: los artefactos vigentes (windows_v4, pk_v2,
# tokens_v2 y sus anotaciones) salen de ella. cf_v7_1 es la regenerada con el
# muestreo v7.1 en la rejilla del escalón.
#
# La cohorte y sus artefactos derivados tienen que cambiar JUNTOS. Si la cohorte
# activa fuese cf_v7_1 mientras las ventanas, el pk y los tokens siguen siendo
# los de cf_v7, cualquier rutina que lea casos CRUDOS a través de ``cf_pairs``
# (la anotación de pares) leería una cohorte y contrastaría contra anotaciones y
# tokens de otra, en silencio. Por eso el valor por defecto es ``cf_v7`` hasta
# que la Fase 5 adopte en bloque cf_v7_1 con sus artefactos, y las regeneraciones
# del paso 3d fijan la cohorte nueva SÓLO durante su ejecución con
# ANESTESIA_CF_COHORT.
CF_V7_DIR = COHORTS["cf_v7"]
CF_V7_1_DIR = COHORTS["cf_v7_1"]
CF_COHORT_DEFAULT = "cf_v7"
CF_COHORT_ENV_VAR = "ANESTESIA_CF_COHORT"
CF_COHORT_ACTIVE = os.environ.get(CF_COHORT_ENV_VAR, CF_COHORT_DEFAULT)
if CF_COHORT_ACTIVE not in COHORTS:
    raise RuntimeError(
        f"{CF_COHORT_ENV_VAR}={CF_COHORT_ACTIVE!r} no es una cohorte conocida: "
        f"{sorted(COHORTS)}")

# Etiqueta LÓGICA de partición -> cohorte que la alimenta. Se conserva la
# etiqueta ``cf_v7`` en las particiones (``source=cf_v7``) para que los
# artefactos regenerados sean comparables partición a partición con los
# adoptados; el directorio FÍSICO lo registran los manifiestos vía
# ``cohort_label_map()``.
COHORT_LABELS: dict[str, str] = {"cf_v7": CF_COHORT_ACTIVE}


def cohort_label_map() -> dict[str, dict[str, str]]:
    """Mapeo explícito etiqueta lógica -> cohorte y directorio físico."""
    return {label: {"cohort": cohort, "dir": str(COHORTS[cohort])}
            for label, cohort in COHORT_LABELS.items()}


def dataset_sources() -> dict[str, Path]:
    """Fuentes de los datasets (ventanas, pk y tokens).

    real, synthetic_v7, vaso_reinf_v7 y el CF ACTIVO bajo la etiqueta lógica
    ``cf_v7``. Es lo que deben usar los constructores en lugar de recorrer
    ``COHORTS`` (que ahora incluye también la cohorte histórica y la de v7.1).
    """
    return {
        "real": COHORTS["real"],
        "synthetic_v7": COHORTS["synthetic_v7"],
        "vaso_reinf_v7": COHORTS["vaso_reinf_v7"],
        "cf_v7": COHORTS[CF_COHORT_ACTIVE],
    }


def cohort_dir(name: str) -> Path:
    """Directorio de una cohorte por nombre, con la etiqueta lógica resuelta a
    la cohorte activa. Lectura DIFERIDA: nunca se captura en una constante de
    módulo (era el bug de ``cf_pairs.CF_CASES_DIR``)."""
    return COHORTS[COHORT_LABELS.get(name, name)]


# ---------------------------------------------------------------------------
# Guarda de coherencia cohorte <-> artefactos derivados
# ---------------------------------------------------------------------------
# La cohorte de CF y las ventanas, el pk y los tokens tienen que ser de la MISMA
# generación. Si la cohorte activa fuese cf_v7_1 mientras los artefactos siguen
# siendo los de cf_v7, cualquier rutina que lea casos CRUDOS (de ahí salen los
# setpoints ``vent_*``) resolvería a una cohorte y contrastaría contra tokens de
# otra, sin error ni aviso. El contexto se comparte entre generaciones (se
# reutiliza context_v2, verificado por G3c).
ARTIFACTS_BY_COHORT: dict[str, dict[str, str]] = {
    "cf_v7": {"windows": "windows_v4", "pk": "pk_v2",
              "tokens": "tokens_v2", "context": "context_v2"},
    "cf_v7_1": {"windows": "windows_v4_1", "pk": "pk_v2_1",
                "tokens": "tokens_v2_1", "context": "context_v2"},
}


def generation_mismatches() -> list[str]:
    """Artefactos cuyo directorio no es el de la generación de la cohorte
    activa. Lista vacía si todo cuadra (o si la cohorte no está tabulada)."""
    expected = ARTIFACTS_BY_COHORT.get(CF_COHORT_ACTIVE)
    if not expected:
        return []
    actual = {"windows": WINDOWS_DIR.name, "pk": PK_DIR.name,
              "tokens": TOKENS_DIR.name, "context": CONTEXT_DIR.name}
    return [f"{k}: la cohorte {CF_COHORT_ACTIVE} espera '{v}' y hay "
            f"'{actual[k]}'"
            for k, v in expected.items() if actual.get(k) != v]


def require_same_generation() -> None:
    """Falla con mensaje claro si la cohorte activa y los directorios de
    artefactos no son de la misma generación. No se llama en el import: sólo
    bajo demanda (y desde ``require_vigent_dirs``)."""
    bad = generation_mismatches()
    if bad:
        raise RuntimeError(
            f"{CF_COHORT_ENV_VAR}={CF_COHORT_ACTIVE} no cuadra con los "
            "directorios de artefactos: la cohorte y sus artefactos derivados "
            "cambian JUNTOS.\n  " + "\n  ".join(bad))

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

    Incluye la guarda de coherencia entre la cohorte de CF activa y la
    generación de los artefactos de ventanas, pk y tokens.
    """
    require_same_generation()
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
