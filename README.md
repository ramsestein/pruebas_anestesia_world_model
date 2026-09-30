# World model de anestesia — `anestesia_world`

Repositorio de un **world model de anestesia** en **fase de mantenimiento**.
Contiene el generador procedural de casos de anestesia (`anessim`), la
tokenización de la fisiología intraoperatoria, el autoencoder de fisiología y
el conjunto de diagnósticos que auditan y validan cada etapa. El modelo de
transición (el "mundo" propiamente dicho) todavía **no se ha empezado**.

> **Aviso:** herramienta de investigación. No es un dispositivo médico, no
> prescribe dosis y no afirma efectos causales en pacientes reales.

## Estado actual

| Pieza | Estado |
|-------|--------|
| Generador `anessim` | **cerrado en v7** (`src/anessim/`, `configs/synthetic_v7.yaml`) |
| Ventanas | `data/windows_v4` (cohortes real + v7) |
| Tokens v1 | generados **sobre `windows_v2` (PERDIDO)**; pendientes de regenerar sobre `windows_v4` |
| AE de fisiología | **entrenado y congelado: variante `ae_bal`** (`data/ae_v1/ae_bal/`) |
| Transferencia sintético→real | **confirmada** (`REPORT_v9.txt`) |
| Modelo de transición | no empezado |

El detalle del estado, las decisiones cerradas y los hechos verificados están
en [`MEMORIA.md`](MEMORIA.md). Las brechas conocidas del generador, en
[`LIMITACIONES_GENERADOR_v7.md`](LIMITACIONES_GENERADOR_v7.md).

## Estructura del repositorio

```
anestesia_world/
├── src/anessim/            # Generador procedural de casos (simulador)
│   ├── configs/            # YAML: default.yaml, synthetic_v5/v6/v7.yaml
│   ├── scripts/            # Generación de cohortes y pares contrafactuales
│   ├── pk/  pd/            # Farmacocinética / farmacodinamia
│   ├── simulate.py         # Orquestador de un caso
│   └── cli.py              # Punto de entrada anessim
├── src/autoencoder/        # window.py: construcción de ventanas de 5 s
├── src/ae/                 # physio_ae.py: autoencoder de fisiología (latente 32d)
├── src/tokens/             # tokenize.py, pk_tokens.py, context_vocab.py
├── src/diagnostics/        # cohort_gap, gap_addendum, v6_validate,
│                           #   v7_validate, v7_attribution, v7_transfer_probe
├── scripts/                # generate_v7.py, build_windows_v4.py y v9/v10
├── tests/                  # pytest (unitarios + integración sobre data/)
├── manifests/              # Manifiestos de procedencia (versionados)
├── reports/                # Salidas pytest (_pytest_*) e informes REPORT_*.txt
├── data/                   # Datos. NO versionado (ver .gitignore)
├── MEMORIA.md              # Documento de conocimiento
├── LIMITACIONES_GENERADOR_v7.md
├── contrato_ae_v1.md       # Contrato del autoencoder de fisiología
├── contrato_tokens_v1.md   # Contrato del tokenizador
├── pyproject.toml
└── requirements.txt
```

- **`src/anessim/`** — simulador procedural de casos intraoperatorios en
  formato VitalDB: paciente, línea temporal, fármacos (PK/PD), respiración,
  sensores y render. Módulos: `simulate.py` (orquestador), `clinical.py`,
  `patients.py`, `timeline.py`, `actions.py`, `respiratory.py`,
  `nociception.py`, `sensors.py`, `track_presence.py`, `render.py`,
  `fidelity.py`, `randomization.py`, `labs.py`, `config.py`, `cli.py`, más
  `pk/` y `pd/`.
- **`src/autoencoder/window.py`** — convierte los parquets de caso en ventanas
  de 12 celdas de 5 s (`windows_v4`), con split por hash de subject_id y
  máscaras de plausibilidad.
- **`src/ae/physio_ae.py`** — autoencoder de fisiología: 28 canales de entrada
  (14 valores + 14 máscaras) → latente ordenado de 32 → 14 valores. Entrenado
  con las variantes `ae_real` y `ae_bal`; **congelado `ae_bal`**.
- **`src/tokens/`** — tokenizador v1 en tres módulos: `pk_tokens.py`
  (farmacocinética poblacional), `context_vocab.py` (contexto estático por
  caso) y `tokenize.py` (fármaco + ventilación + contexto + tiempo).
- **`src/diagnostics/`** — seis módulos de diagnóstico:
  `cohort_gap.py` (brecha real/sintético), `gap_addendum.py` (mediciones P1),
  `v6_validate.py` (validación V1–V7), `v7_validate.py` (misma validación
  sobre cohortes v7), `v7_attribution.py` (atribución de la sonda no lineal) y
  `v7_transfer_probe.py` (sonda de transferencia sintético→real).
- **`manifests/`** — copias versionadas de los manifiestos y resultados de
  gates/sondas que viven en `data/` (que está fuera de git). Cadena de
  procedencia del proyecto.
- **`reports/`** — salidas de pytest (`_pytest_*_rojo.txt` / `_pytest_*_verde.txt`)
  e informes de resultados (`REPORT_*.txt`, `*.json`).

## Cohortes

**Existentes** en `data/` (y windowed en `data/windows_v4/`):

| Cohorte | Descripción |
|---------|-------------|
| `real` | 6388 casos reales VitalDB + INSPIRE |
| `synthetic_v7` | 10 000 casos sintéticos base (offset 150001) |
| `vaso_reinf_v7` | 500 casos de refuerzo vasoactivo (offset 160001) |
| `cf_v7` | 6345 pares contrafactuales (pharma 870, vent 600, learning 4875) |

**Perdidas** (no están en disco): `windows_v2` (ventanas v5), `windows_v3`
(ventanas v6), y las cohortes `synthetic_v5`, `vaso_reinf_v5`, `cf_v5`,
`synthetic_v6`, `vaso_reinf_v6`, `cf_v6`.

## Ejecución

Los módulos se ejecutan como paquetes (`python -m ...`) con el `src/` en el
path. Con el entorno activado y el paquete instalado en modo editable
(`pip install -e .`), los imports de `anessim`, `ae`, `tokens`, `diagnostics`
y `autoencoder` resuelven contra `src/`.

Orden del pipeline (de datos brutos a representación):

```bash
# 1. Regenerar cohortes v7 (NO sobrescribe v5/v6)
python scripts/generate_v7.py --all        # o --base | --vaso | --cf ; --smoke N

# 2. Construir las ventanas (rejilla 5 s) -> data/windows_v4
python scripts/build_windows_v4.py --workers 14

# 3. Tokenizar
python -m tokens.pk_tokens run             # PK poblacional (o: verify, manifest)
python -m tokens.context_vocab run         # contexto estático (o: report)
python -m tokens.tokenize run              # fármaco+vent+ctx+tiempo (o: verify, manifest, report)

# 4. Autoencoder de fisiología
python -m ae.physio_ae run                 # entrena ambas variantes
python -m ae.physio_ae run --variant ae_bal
python -m ae.physio_ae evaluate            # re-evalúa sin reentrenar
python -m ae.physio_ae report              # escribe REPORT_ae_v4.txt

# 5. Diagnósticos
python -m diagnostics.cohort_gap run       # brecha real/sintético (o: report)
python -m diagnostics.gap_addendum run     # mediciones P1 (o: report)
python -m diagnostics.v6_validate run      # validación V1-V7 (o: report)
python -m diagnostics.v7_validate run      # validación V1-V7 sobre v7
python -m diagnostics.v7_attribution run   # atribución sonda no lineal
python -m diagnostics.v7_transfer_probe run --cohort v7   # sonda de transferencia
```

Scripts de verificación v9/v10 (integridad y equivalencia; se ejecutan
directamente con `python scripts/<nombre>.py`):

- `v9_verify_real_equivalence.py` — equivalencia de la cohorte real entre
  `windows_v2` y `windows_v4`.
- `v9_check_reconstruction.py` — valida numéricamente la reconstrucción de
  `v6_validate.py`.
- `v9_inventory_backup.py` — inventario sha256 de artefactos + manifiesto global.
- `v10_gate_recheck.py` — reejecuta gates 2/3 del `ae_bal` congelado sobre
  `windows_v4` real val.
- `v10_marginales_val.py` — marginales del real split=val vs tabla D4.
- `v10_real_null_audit.py` — auditoría de nulls/placeholders en casos reales.
- `v10_real_schema_variety.py` — inventario de columnas por caso (esquemas).
- `v10_backup.py` — copia de seguridad de lo irreemplazable (fuera del repo).
- `diag_source_probe_physio.py` — sondeo de fuente (sin aserción).

## Tests

```bash
python -m pytest
```

Los tests que dependen de datos ausentes (`windows_v2`) siguen fallando: es
esperado y está declarado. Ningún test debe fallar por imports rotos ni por
rutas movidas.

## Reproducibilidad

- Semilla global por run (`--seed`) + semilla por caso (`seed + caseid`).
- Un par contrafactual usa la misma semilla en ambas ramas (prefijo idéntico
  bit a bit hasta `split_t`).
- No se usa `hash()` de Python.

## Referencias

- [`MEMORIA.md`](MEMORIA.md) — conocimiento del proyecto y decisiones cerradas.
- [`LIMITACIONES_GENERADOR_v7.md`](LIMITACIONES_GENERADOR_v7.md) — brechas
  medidas del generador v7 contra la cohorte real.
- [`contrato_ae_v1.md`](contrato_ae_v1.md), [`contrato_tokens_v1.md`](contrato_tokens_v1.md)
  — contratos de especificación (ver aviso de desactualización en su cabecera).
