# Anesthesia Synthetic Case Generator (`anessim`)

Este repositorio contiene exclusivamente el **generador procedural de casos de anestesia** alineado con el formato VitalDB, junto con los datasets generados y los datos reales de referencia. No incluye código de modelo, entrenamiento, evaluación ni harness de ventanas.

> **Aviso:** herramienta de investigación. No es un dispositivo médico, no prescribe dosis y no afirma efectos causales en pacientes reales.

---

## Estructura del repositorio

```
anestesia_world/
├── src/anessim/              # Paquete generador
│   ├── configs/              # Configuraciones YAML del simulador
│   ├── scripts/              # Scripts de generación de casos/contrafactuales
│   ├── pk/                   # Farmacocinética poblacional
│   ├── pd/                   # Farmacodinamia (BIS, MAP, HR)
│   ├── simulate.py           # Orquestador principal de un caso
│   ├── cli.py                # Punto de entrada por consola
│   └── ...
├── src/autoencoder/          # window.py + auditor sintético
├── data/
│   ├── real/                 # Casos reales VitalDB + INSPIRE
│   ├── synthetic_v5/         # 10 000 casos sintéticos base (v5)
│   ├── synthetic_vaso_reinf_v5/ # 500 casos de refuerzo vasoactivo (v5)
│   └── cf_v5/                # Pares contrafactuales de evaluación (v5)
├── pyproject.toml            # Empaquetado del paquete anessim
└── requirements.txt          # Dependencias mínimas
```

> **Nota sobre v3:** las cohortes `synthetic_v3/`, `synthetic_vaso_reinf/` y
> `counterfactual_eval/` fueron **retiradas** por defectos de canal que
> afectaban al 90-100% de los casos (documentados en
> `src/autoencoder/REPORT_audit_synthetic.txt`): presencia de tracks ~50%,
> EtCO2 clavado en ~15 mmHg, SpO2/BT degenerados, FIO2 en fracción, palancas CF
> sin efecto y prefijo CF no idéntico. El generador fue corregido
> (`src/anessim/CHANGELOG_v5.md`, `src/anessim/REPORT_regen_v5.txt`) y la
> cohorte regenerada como v5.

---

## `src/anessim/` — Generador procedural

El paquete `anessim` simula casos intraoperatorios completos: paciente, línea temporal de la cirugía, administración de fármacos, farmacocinética/farmacodinamia, respiración, sensores y salida en formato compatible con VitalDB.

### Módulos principales

| Módulo | Función |
|--------|---------|
| `simulate.py` | Orquesta un caso completo. Genera el paciente, la línea temporal, las acciones, integra PK/PD, respiración y sensores. Soporta bifurcación contrafactual. |
| `config.py` | Configuración global `SimulatorConfig`, paths por defecto, tracks canónicos y constantes. |
| `cli.py` | Punto de entrada `anessim` para generar cohortes en paralelo. |
| `patients.py` | Muestreo de pacientes virtuales desde `clinical_data_enriched.parquet`. |
| `actions.py` | Tipos de acción: bolos, infusiones, cambios ventilatorios, estímulos, fluidos, vasopresores. |
| `timeline.py` | Fases clínicas del caso (preop, inducción, intubación, mantenimiento, emergencia, extubación). |
| `randomization.py` | Perturbaciones de política y variabilidad de infusiones/bolos. |
| `pk/base.py` | Maquinaria PK base con integración ODE. |
| `pk/propofol.py` | Modelos de propofol: Marsh, Schnider, Eleveld. |
| `pk/remifentanil.py` | Modelo de remifentanilo Minto. |
| `pd/hemodynamics.py` | MAP/HR desde Ce propofol/remi/sevo, estímulo nociceptivo y vasopresores. |
| `pd/bis_surface.py` | Superficie BIS combinada de propofol, remifentanilo y sevoflurano. |
| `respiratory.py` | FiO2/PEEP → PaO2 → SpO2; MV → PaCO2 → EtCO2; compliance → PIP. |
| `nociception.py` | Estímulo quirúrgico y atenuación por remifentanilo. |
| `sensors.py` | Ruido de sensor, discretización, gaps y jitter de cadencia. |
| `track_presence.py` | Prevalencia de tracks imitando casos reales. |
| `render.py` | Escritura de `cases/`, `truth/`, `clinical/`, `clinical_notes/` y `metadata/`. |
| `clinical.py` | Generación de notas clínicas en texto. |
| `labs.py` | Generación de tabla sintética de laboratorios. |
| `fidelity.py` | Comparación distribucional sintético vs real. |

### `src/anessim/configs/`

Configuraciones YAML del simulador:

- **`default.yaml`**: configuración por defecto del generador.
- **`synthetic_v5.yaml`**: configuración para generar la cohorte base v5 (`n_cases: 10000`, `case_offset: 40001`).

### `src/anessim/scripts/`

Scripts de generación de datasets. Se ejecutan como módulos, por ejemplo:

```bash
python -m anessim.scripts.generate_f2_cf pharma --n-pairs 250 --start-caseid 70001 --output-dir D:/data/anestesia_world/cf_pharma_v5 --workers 6
```

| Script | Propósito |
|--------|-----------|
| `generate_f2_cf.py` | Generador principal de pares contrafactuales v5: `pharma` (4 palancas), `vent` (4 palancas) y `learning` (13 palancas). |
| `generate_counterfactuals.py` | Generador legacy de pares CF (propofol vs null, noradrenalina vs nada, remi vs no remi). |
| `generate_cf_ephedrine.py` | Pares CF específicos de efedrina (bolo vs nada). |
| `generate_vaso_reinforcement.py` | Dataset de refuerzo vasoactivo (70% efedrina, 20% fenilefrina, 10% noradrenalina). |
| `rerender_learning_truth.py` | Re-simula la colección CF learning conservando lineage para añadir canales al truth. |
| `_rebuild_vaso_clinical.py` | Reconstruye determinísticamente filas clínicas del conjunto vaso_reinf. |
| `smoke_v31.py` | Smoke test multi-seed del simulador v3.1. |
| `_smoke_cf_fix.py` | Smoke de 1 par por palanca afectada tras correcciones de CF. |

---

## `data/` — Datasets

Todos los datasets están en disco. El generador los produce o los consume como referencia.

### `data/real/` — Datos reales VitalDB + INSPIRE

Contiene los casos reales descargados de VitalDB y enriquecidos con la tabla INSPIRE.

```
data/real/
├── cases/                          # ~6.388 parquets, uno por caso
│   ├── 0001.parquet
│   ├── 0002.parquet
│   └── ...
├── clinical_data.parquet           # Tabla clínica base
├── clinical_data_enriched.parquet  # Tabla clínica enriquecida con INSPIRE (82 columnas)
├── lab_data.parquet                # Laboratorios agregados
└── download_progress.json          # Progreso de descarga
```

#### `cases/{caseid}.parquet`

Cada archivo es una serie temporal con índice `time` (segundos desde `casestart`, `float64`). Contiene los tracks monitorizados del caso en formato VitalDB.

- **Filas:** entre ~15 000 y ~30 000 según duración del caso.
- **Columnas:** ~68 tracks (varía ligeramente por caso). Ejemplo de columnas:
  - `BIS/BIS`, `BIS/EMG`, `BIS/SEF`, `BIS/SQI`, `BIS/SR`, `BIS/TOTPOW`
  - `Primus/ETCO2`, `Primus/FIO2`, `Primus/MAC`, `Primus/PEEP_MBAR`, `Primus/PIP_MBAR`, `Primus/MV`, `Primus/TV`, `Primus/RR_CO2`
  - `Solar8000/HR`, `Solar8000/ART_MBP`, `Solar8000/ART_SBP`, `Solar8000/ART_DBP`, `Solar8000/PLETH_SPO2`, `Solar8000/ETCO2`, `Solar8000/BT`
  - `Orchestra/PPF20_RATE`, `Orchestra/RFTN20_RATE` (presentes en una minoría de casos reales)
- **Valores:** numéricos `float32`; `null` indica missing.
- **Cadencia:** no uniforme. Cada track tiene su propia cadencia (BIS ~1 s, hemodinámica ~2 s, ventilación ~6 s).

#### `clinical_data_enriched.parquet`

Tabla de una fila por caso con variables clínicas estáticas. 82 columnas que incluyen:

- Identificadores: `caseid`, `subjectid`, `casestart`, `caseend`, `anestart`, `aneend`, `opstart`, `opend`.
- Demografía: `age`, `sex`, `height`, `weight`, `bmi`.
- Estado basal: `asa`, `emop`, `department`, `optype`, `dx`, `opname`, `approach`, `position`, `ane_type`.
- Preoperatorio: `preop_htn`, `preop_dm`, `preop_ecg`, `preop_pft`, `preop_hb`, `preop_plt`, `preop_pt`, `preop_aptt`, `preop_na`, `preop_k`, `preop_gluc`, `preop_alb`, `preop_ast`, `preop_alt`, `preop_bun`, `preop_cr`, `preop_ph`, `preop_hco3`, `preop_be`, `preop_pao2`, `preop_paco2`, `preop_sao2`.
- Vía aérea: `cormack`, `airway`, `tubesize`, `dltubesize`, `lmasize`.
- Accesos: `iv1`, `iv2`, `aline1`, `aline2`, `cline1`, `cline2`.
- Intraoperatorio: `intraop_ebl`, `intraop_uo`, `intraop_rbc`, `intraop_ffp`, `intraop_crystalloid`, `intraop_colloid`, `intraop_ppf`, `intraop_mdz`, `intraop_ftn`, `intraop_rocu`, `intraop_vecu`, `intraop_eph`, `intraop_phe`, `intraop_epi`, `intraop_ca`.
- Enriquecimiento INSPIRE: `diagnosis_count`, `top_diagnosis_code`, `top_diagnosis_chapter`, `lab_count`, `lab_unique_items`, `medication_count`, `medication_unique`, `common_route`.

#### `lab_data.parquet`

Tabla de laboratorios con una fila por `(caseid, item, result_time)`. Columnas típicas: `caseid`, `subjectid`, `result_time`, `item`, `value`, `flag`.

---

### `data/synthetic_v5/` — Casos sintéticos base

Cohorte de 10 000 casos sintéticos generados con el simulador `anessim`
corregido (v5, caseids 40001–50000).

```
data/synthetic_v5/
├── cases/              # Series temporales observadas (formato VitalDB)
├── truth/              # Ground truth fisiológico limpio
├── clinical/           # Fila clínica estática por caso
├── clinical_notes/     # Nota clínica en texto
├── metadata/           # Metadatos oráculo por caso
├── config.yaml         # Config usada para generar la cohorte
├── clinical_data.parquet
└── lab_data.parquet
```

#### `cases/{caseid}.parquet`

Series temporal observada. Similar a los casos reales, pero con algunas diferencias:

- Incluye tracks de bomba `Orchestra/*` que en los reales a menudo faltan.
- Incluye columnas de bolos explícitos: `ppf_bolus_mg`, `remi_bolus_ug`, `roc_bolus_mg`, `phen_bolus_mcg`, `eph_bolus_mg`.
- `Orchestra/PPF20_RATE` y `RFTN20_RATE` están en **mL/h**.
- `Primus/FIO2` y `Primus/SET_FIO2` están en **%** (21–100).
- El índice es `time` en segundos.

#### `truth/{caseid}_truth.parquet`

Ground truth fisiológico completo sin ruido de sensor. Incluye ~40 columnas:

| Columna | Descripción |
|---------|-------------|
| `time` | Tiempo en segundos |
| `phase` | Fase clínica (preop, induction, etc.) |
| `ce_propofol`, `cp_propofol` | Concentración efectiva y plasmática de propofol |
| `ce_remifentanil`, `cp_remifentanil` | Concentración efectiva y plasmática de remifentanilo |
| `bis` | BIS real |
| `map`, `map_baseline` | MAP real y baseline |
| `hr` | Frecuencia cardíaca real |
| `propofol_rate`, `remifentanil_rate` | Infusiones en mg/min o µg/min |
| `noradrenaline_rate` | Infusión de nora en µg/kg/min |
| `ephedrine_dose`, `phenylephrine_dose` | Bolos acumulados |
| `sevoflurane_mac` | MAC de sevoflurano |
| `surgical_stimulus` | Estímulo quirúrgico |
| `nociceptive_map`, `nociceptive_hr`, `nociceptive_bis` | Componentes nociceptivos |
| `remi_attenuation_map`, `remi_attenuation_hr` | Atenuación por remifentanilo |
| `a_null_mask` | Máscara de acción nula |
| `propofol_perturbation`, `remi_perturbation` | Perturbaciones aleatorias |
| `pao2_true`, `shunt_fraction` | Oxigenación arterial y shunt |
| `vco2`, `fio2_applied`, `peep_applied`, `rr_applied`, `tv_applied` | Variables respiratorias |
| `pip_true`, `compliance` | PIP y compliance real |

#### `clinical/{caseid}_clinical.parquet`

Fila clínica estática del caso (82 columnas, mismo schema que `clinical_data_enriched.parquet`).

#### `clinical_notes/{caseid}.txt`

Nota clínica en texto libre que resume el paciente, la cirugía y los eventos relevantes.

#### `metadata/{caseid}_meta.json`

Metadatos oráculo del caso. Claves principales:

- `pk_model`: modelo PK usado (Marsh/Schnider/Eleveld para propofol, Minto para remi).
- `prop_sensitivity`: parámetros de sensibilidad al propofol.
- `vasopressor_response`: parámetros de respuesta a vasopresores.
- `nociception`: perfil de estímulo quirúrgico.
- `is_counterfactual`: `false` para casos base.
- `counterfactual_split_t`: null para casos base.

---

### `data/cf_v5/` — Pares contrafactuales

Contiene pares de casos contrafactuales: dos ramas del mismo paciente con la misma historia hasta un instante `split_t`, y a partir de ahí una sola palanca distinta.

```
data/cf_v5/
├── cases/              # Series temporales observadas de ambas ramas
├── clinical/           # Fila clínica compartida por par
├── clinical_notes/     # Nota clínica compartida por par
├── metadata/           # Metadatos de cada par (cf_pair_*.json)
└── truth/              # Ground truth de ambas ramas
```

- **Caseids:** pares consecutivos impar/par (ej. `70001.parquet` y `70002.parquet`).
- El caseid impar es la rama de intervención; el par es la rama control (regla uniforme).
- El prefijo antes de `split_t` es idéntico bit a bit en ambas ramas (mismo seed y mismo paciente).
- Los metadatos incluyen la palanca aplicada (`lever`), el seed y el punto de split.
- Palancas: `pharma` (propofol_bolus, noradrenaline, remi_up, ephedrine),
  `vent` (peep_up, peep_down, fio2_down, sevo_up) y `learning` (13 palancas).

---

### `data/synthetic_vaso_reinf_v5/` — Refuerzo vasoactivo

Cohorte de 500 casos (caseids 61001–61500) diseñada para reforzar la señal de vasopresores.

```
data/synthetic_vaso_reinf_v5/
├── cases/              # Series temporales observadas
├── truth/              # Ground truth (con columna phase)
├── clinical/           # Filas clínicas
├── clinical_notes/     # Notas clínicas
└── metadata/           # Metadatos oráculo
```

- Distribución de drogas: 70% efedrina, 20% fenilefrina, 10% noradrenalina.
- Las filas clínicas se generan con el simulador corregido (ya no requieren reconstrucción externa).

---

## Instalación

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Esto instala el paquete `anessim` con los entry points:

- `anessim` → generador de casos base.
- `anessim-fidelity` → informe de fidelidad sintético vs real.

---

## Uso básico

### Generar casos base

```bash
python -m anessim.cli \
  --config src/anessim/configs/synthetic_v5.yaml \
  --n-cases 10000 --case-offset 40001 --workers 16
```

O, con el paquete instalado:

```bash
anessim --config src/anessim/configs/synthetic_v5.yaml --n-cases 10000 --case-offset 40001 --workers 16
```

### Generar pares contrafactuales

```bash
python -m anessim.scripts.generate_f2_cf pharma \
  --n-pairs 250 --start-caseid 70001 \
  --output-dir D:/data/anestesia_world/cf_pharma_v5 --workers 6

python -m anessim.scripts.generate_f2_cf vent \
  --n-pairs 150 --start-caseid 80001 \
  --output-dir D:/data/anestesia_world/cf_vent_v5 --workers 6

python -m anessim.scripts.generate_f2_cf learning \
  --n-pairs 375 --start-caseid 90001 \
  --output-dir D:/data/anestesia_world/cf_learning_v5 --workers 6
```

### Informe de fidelidad

```bash
anessim-fidelity --n-cases 500 --out reports/fidelity_v5.json
```

---

## Reproducibilidad

- Semilla global por run (`--seed`) + semilla por caso (`seed + caseid`).
- Paciente, línea temporal, perturbaciones y ruido de sensor son deterministas dada la semilla.
- No se usa `hash()` de Python.
- Un par contrafactual usa la **misma semilla** en ambas ramas.

---

## Unidades (contrato v5)

| Magnitud | Unidad | Nota |
|----------|--------|------|
| `Orchestra/PPF20_RATE` | mL/h | Concentración 20 mg/mL |
| `Orchestra/RFTN20_RATE` | mL/h | Concentración 20 µg/mL |
| `Orchestra/PHEN_RATE` | µg/min | — |
| `Orchestra/NEPI_RATE` | µg/kg/min | — |
| `Primus/FIO2`, `Primus/SET_FIO2` | % | 21–100 |
| Bolus de propofol | mg | — |
| Bolus de remifentanilo | µg | — |
| Bolus de rocuronio | mg | — |
| Bolus de fenilefrina | µg | — |
| Bolus de efedrina | mg | — |

---

## Fisiología modelada (flechas causales)

- **Propofol/remifentanilo/sevoflurano** → depresión de BIS, MAP y HR.
- **Estímulo quirúrgico** → elevación de MAP/HR; atenuado por remifentanilo.
- **FiO2** → PaO2 → SpO2 (curva de Hill + shunt).
- **MV** → PaCO2 → EtCO2.
- **PEEP** → reclutamiento (↑SpO2) y ↓precarga (↓MAP).
- **Noradrenalina/fenilefrina/efedrina** → respuesta hemodinámica.
