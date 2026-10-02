# LIMITACIONES_GENERADOR_v7.md

Limitaciones conocidas del generador v7 (documentadas, NO corregidas en esta
iteración). Los números son residuos medidos contra la cohorte real
(`data/windows_v4`, equivalencia verificada en `REPORT_v9_datos.txt`). Se
aceptan con la siguiente lógica: la decisión de transferencia (sonda v9) no
depende de cerrar estas brechas; se revisan solo si el modelo de transición
del proyecto rinde por debajo de lo esperado.

---

## 1. Patrón de mantenimiento descuadrado (cadencia de muestreo/medida)

`frac_nochg` real vs sintético v7 (fracción de pasos consecutivos de 5 s sin
cambio del valor observado). El sintético cambia de más o de menos según la
variable:

| Variable            | real    | sintético | problema                    |
|---------------------|---------|-----------|-----------------------------|
| Primus/MV           | 0.8208  | 0.4497    | cambia demasiado            |
| Primus/RR_CO2       | 0.9779  | 0.7050    | cambia demasiado            |
| Solar8000/PLETH_SPO2| 0.9350  | 0.6121    | cambia demasiado            |
| Primus/PIP_MBAR     | 0.7499  | 0.5575    | cambia demasiado            |
| Primus/ETCO2        | 0.7358  | 0.6165    | cambia demasiado            |
| BIS/EMG             | 0.0854  | 0.0200    | cambia de menos             |
| BIS/BIS             | 0.1191  | 0.0504    | cambia de menos             |
| Primus/PEEP_MBAR    | 0.9176  | 0.9873    | casi no cambia (pasado)     |
| Solar8000/HR        | 0.5260  | 0.6389    | cambia de menos (pasado)    |

Causa: cadencia efectiva de muestreo y mantenimiento entre muestras en la capa
de medida. Aceptado porque el gate de transferencia (D3 vs D2) no se apoya en
el paso único, y el patrón es una discrepancia de forma, no de nivel.

## 2. Identidad de las presiones arteriales violada

Discrepancia de información mutua (pares) sintético − real (A6):

| Par                  | ΔMI     |
|----------------------|---------|
| ART_SBP \| ART_DBP   | +0.7676 |
| ART_MBP \| ART_SBP   | +0.4970 |
| ART_MBP \| ART_DBP   | +0.3437 |

En el real las tres están ligadas casi determinísticamente
(MAP ~ DBP + (SBP−DBP)/3). El generador las emite como `map_vals + 35.0`,
`map_vals`, `map_vals − 20.0`, cada una contaminada con ruido de medida
INDEPENDIENTE (`_obs_art` genera su propio offset y ruido por canal). Un árbol
lo detecta en una celda. Aceptado por ser capa de observación (derivación y
contaminación de SBP/MBP/DBP a partir del MAP fisiológico), no ganancia
hemodinámica: la relación fármaco→MAP no está rota.

## 3. Soporte discreto estrecho

`n_distinct` real vs sintético (valores distintos que el monitor visita):

| Variable      | real | sintético |
|---------------|------|-----------|
| ART_MBP       | 297  | 166       |
| ART_SBP       | 308  | 171       |
| ART_DBP       | 277  | 170       |
| SpO2          | 42   | 28        |
| HR            | 181  | 142       |
| BIS           | 869  | 660       |

El sintético no alcanza la mitad de los valores que el real visita: regla de
decisión gratis para un clasificador. Es variabilidad interindividual e
intracaso insuficiente, no colas a inventar.

## 4. Varianza de incrementos ~3x menor (brecha de dinámica)

std de los deltas (x_{t+1} − x_t) medidos en la sonda:

| Variable | std delta sintético | std delta real | ratio sint/real |
|----------|--------------------:|---------------:|----------------:|
| HR       | 1.540               | 4.119          | 0.37            |
| BIS      | 0.895               | 2.750          | 0.33            |

Los incrementos sintéticos tienen entre un tercio y un 40 % de la variabilidad
de los reales. Es una brecha de dinámica real y medida (hallazgo 2.3 de la
sonda v9). No se corrige ahora; se revisa solo si el modelo de transición
rinde por debajo de lo esperado.

## 5. Acoplamiento PIP-TV

Discrepancia de correlación −0.3939 y de MI +0.1915 entre PIP y TV. El flujo
inspiratorio no se deriva de TV y del tiempo inspiratorio (función de RR y de
la relación I:E): es independiente. La W1 de PIP empeoró (0.0527 → 0.0903) al
introducir el término resistivo. Revisar también PIP|RR_CO2 (+0.3194) y
PEEP|TV (+0.2631), que apuntan al mismo sitio. La corrección es de derivación,
no de parámetro.

## 6. BT: cohorte v7 en disco con C3 sin revertir

- Código (`src/anessim/simulate.py`): C3 REVERTIDO — `_bt_start = uniform(36.5, 37.4)`.
- Datos en disco (`synthetic_v7`/`windows_v4`): W1 BT 0.9293, media 34.92 vs
  real 35.77 — generados con la versión ensanchada `normal(36.0, 0.9)` que ya
  no existe en el código.
- Decisión (v8, 3.1): documentado, no regenerado. Se corrige en la próxima
  regeneración con el código revertido.

## 7. Valores V1/V2/V3 registrados (cache no reproducible)

- V1 lineal: AUC 0.7654 (objetivo ≤ 0.7036) — FALLA.
- V2 no lineal HGB: AUC 0.9767 (objetivo < 0.85) — FALLA.
- V3 solapamiento 20-NN: 0.1465 (objetivo > 0.25) — FALLA.

Estos valores salen de `v7_validate_results.json`, cache generada con el
`v6_validate.py` original (perdido). En v9 se VALIDÓ la reconstrucción de
`v6_validate.py` reproduciendo V1/V2/V3/V4 numéricamente sobre la cohorte real
de windows_v4 (equivalente) y las cohortes v7: los valores coinciden
exactamente (ver `REPORT_v9_datos.txt` 1.3). Quedan registrados como son, con
la nota de que la cache no es reproducible por sí sola sin la reconstrucción.

## 8. Pares CF: retraso de la acción y divergencia no resuelta (paso 3b)

El paso 3b anota los 6345 pares CF con `t_action`, `t_divergence_raw`,
`effect_lag_windows` y `lever_effective` (ver `REPORT_paso3b_pares_cf.txt` y
`data/tokens_v2/pairs_annotated.parquet`). El generador aplica las
intervenciones en `t_action = split_t + 10 s` (las Actions) o `t_action =
split_t` (overrides de ventilación); el criterio corregido es
`post_action = t1 > t_action` refinado a la rejilla de tokens.

### 8.1 Cuantización de la capa de tokens (5 s)

`pk_v2` reporta en el inicio de la celda `[5k, 5k+5)` el efecto de las acciones
que caen dentro de ella: en el par 171501 la acción está en 6781.7 s y
`ce_efedrina` ya salta en el punto de rejilla 6780. Por eso la identidad de la
intervención puede aparecer en las features hasta 5 s ANTES de `t_action` y el
prefijo idéntico (C4) debe medirse como `t1 < floor(t_action/5)·5`. Sin este
refinamiento, 120 pares aparecían con divergencia en el prefijo.

### 8.2 Palancas de consigna persistente (overrides de ventilación)

`simulate.py` aplica los overrides de ventilación como un delta acumulativo
sobre la señal (`peep_arr[post] = clip(peep_arr[post] + delta, ...)`, línea
346), NO como un evento puntual. Consecuencia: la consigna observada
(`Primus/SET_INTER_PEEP`, `SET_RR_IPPV`, `SET_TV_L`, `SET_FIO2`) no cambia en
`t_action` sino cuando el plan base cambia esa consigna. Ejemplo: el par
180319 (`peep_down`) tiene PEEP base 0 en `t_action` (recorte a 0, sin cambio) y
la divergencia observada aparece 45 s después, cuando el plan base sube la PEEP
a 4 y la rama de intervención la deja en 0.

Por eso los criterios C1 (`|t_divergence_raw − t_action| <= 30 s`) y C2
(efecto en la ventana de `t_action` o la siguiente) se evalúan sobre las
ACCIONES PUNTUALES (pharma/learning/sevo) y las consignas persistentes se
reportan aparte. Medido: C1 pasa en el 100 % de las acciones puntuales (0/4380
fallos) y falla en 78/1797 consignas persistentes; C2 da 99.93 % en acciones
puntuales (>= 99 %) y 94.94 % en consignas persistentes. Para estas últimas el
efecto existe pero llega con la cadencia del plan.

### 8.3 Pares sin efecto observable

Los pares con `lever_effective = False` (sin efecto en las features del
`lever_group` tras `t_action`) se clasifican con evidencia en
`manifests/tokens_v2_cf_pairs_annotation.json`. Causas:

- `clip_bound` — el cambio pedido queda anulado por un recorte del simulador.
  Afecta a PEEP: `peep_down` (delta −5) y `set_peep` (delta uniforme(−5,5))
  con PEEP base 0 se recortan al mínimo del clip `[0, 25]`, de modo que la
  PEEP aplicada no cambia (≈ 46 % de los casos tras la recalibración v6 parten
  de PEEP 0).
- `below_resolution` — la intervención SÍ cambia la VERDAD, pero la diferencia
  no se resuelve en la variable observada que alimenta las features:
  (a) tasas `rftn20_rate`, `ppf20_rate`, `remi_up`: la acción de mantenimiento
  de la base sobrescribe la infusión a ~1.5 s de `t_action` (la verdad, p. ej.
  `remifentanil_rate`, difiere durante un tramo de ~1.5 s) y la cadencia
  irregular del track observado `Orchestra/*_RATE` no captura ese tramo, por lo
  que la Ce recalculada por `pk_tokens.py` es idéntica; (b) `set_rr` con
  `|rr_delta|` por debajo de la resolución del setpoint RR entero; (c)
  `set_peep` con `|peep_delta|` por debajo del escalón del setpoint (step 1).
- `no_change_requested` — el valor pedido coincide con el valor base en
  `t_action` (tolerancia 5 % relativa): no se pide cambio alguno.
- `case_ends` — el par termina antes de que exista una ventana con
  `t1 > t_action`.

Estas limitaciones son del GENERADOR (capa de observación, cadencia del plan
base y cuantización de las consignas), no del tokenizador. Se documentan, NO se
corrigen en esta iteración: el conjunto CF de entrenamiento se define como los
pares con `lever_effective = True` de `pairs_annotated.parquet`.

