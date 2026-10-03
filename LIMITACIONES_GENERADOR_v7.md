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

## 8. Pares CF: retraso de la acción y divergencia no resuelta (pasos 3b y 3c)

El paso 3b anota los 6345 pares CF con `t_action`, `t_divergence_raw`,
`effect_lag_windows` y `lever_effective` (ver `REPORT_paso3b_pares_cf.txt` y
`data/tokens_v2/pairs_annotated.parquet`). El generador aplica las
intervenciones en `t_action = split_t + 10 s` (las Actions) o `t_action =
split_t` (overrides de ventilación); el criterio corregido es
`post_action = t1 > t_action` refinado a la rejilla de tokens.

El paso 3c corrige esa anotación (`REPORT_paso3c_pares_cf.txt`,
`data/tokens_v2/pairs_annotated_v2.parquet`,
`manifests/tokens_v2_cf_pairs_annotation_v2.json`): sustituye el margen fijo de
2 celdas por una frontera medida (`t_boundary`), localiza `t_effective` y decide
si la anotación corregida es adoptable. **No lo es** (§8.4): el prefijo
fisiológico de la imagen no es limpio en el 14.198 % de los pares efectivos, de
modo que `pairs_annotated.parquet` sigue siendo la referencia del pipeline y
`cf_pairs.load_pairs_annotated_v2()` es sólo una herramienta de diagnóstico.

### 8.1 Cuantización de la capa de tokens (5 s)

La rejilla es `t = a0 + 5k` y **un punto de rejilla `g` es el CIERRE de la celda
semiabierta `(g-5, g]`**; el instante `g` pertenece a esa celda. Los tracks
continuos toman en `g` la última muestra cruda con `time <= g` y los bolos se
estampan en el primer punto de rejilla `>= t_evento` (`window.py:785`,
`searchsorted(grid, tb, side="left")`). Además `pk_tokens.py:161-164` y
`:194-197` aplican el bolo de la celda *i* al inicio de la celda *i−1*
(`bolus_at[:-1] += bol[1:]`), de modo que la Ce diverge una celda entera antes.

Consecuencia medible (fase C del paso 3c, 6190 pares con divergencia de rejilla):
el adelanto `t_effective − t_div_grid` es de **una celda como máximo** (máximo
5.2 s, p95 4.02 s). Sólo cuatro palancas superan los 5 s
(`phen_rate`/`phen_bolus`/`eph_bolus`/`ephedrine`, máx. 5.2 s) y 77 pares caen en
la clase 5–10 s; 0 pares por encima de 10 s. Los dos casos extremos medidos:
acción en 3125.2 s → Ce en el punto 3120 (+5.2 s) y acción en 4685.0 s → Ce en
4680 (+5.0 s). El bolo se estampa a la baja en el eje de 0.5 s
(`t_evento = floor_0.5(t_action)`), lo que aporta ≤ 0.5 s.; el resto (≈ 4.5 s) es
la rejilla de 5 s más el desplazamiento de `pk_tokens.py`.

Por eso la identidad de la intervención puede aparecer en las features hasta una
celda ANTES de la frontera y el prefijo idéntico (C5) debe medirse con
`pre_action_bounded(t1, t_boundary) <=> t1 < t_boundary`, donde
`t_boundary = min(t_div_grid, floor(t_effective/5)·5)`. Medido: **C5 pasa en los
6345 pares** (`c5_ok = true`, 0 fallos): el prefijo de tokens es idéntico entre
ramas en las 65 features y las 12 máscaras.

### 8.2 Palancas de consigna persistente (overrides de ventilación)

`simulate.py` aplica los overrides de ventilación como un delta **aditivo**
sobre la consigna del plan base (`peep_arr[post] = clip(peep_arr[post] + delta,
...)` con `post = t > split_t`), NO como un delta acumulativo en el tiempo ni
como un evento puntual. Consecuencia: la consigna observada
(`Primus/SET_INTER_PEEP`, `SET_RR_IPPV`, `SET_TV_L`, `SET_FIO2`) no cambia en
`t_action = split_t` sino cuando el plan base vuelve a escribir esa consigna.
Ejemplo: el par 180319 (`peep_down`) tiene PEEP base 0 en `t_action` (recorte a 0,
sin cambio) y la divergencia observada aparece 45.11 s después, cuando el plan
base sube la PEEP a 4 y la rama de intervención la deja en 0.

Medido en 3c sobre las 1797 consignas persistentes efectivas: **1719 inmediatas**
(`|t_divergence_raw − t_action| <= 30 s`) y **78 retrasadas** (retraso mediano
484.245 s, máximo 12531.81 s), todas por recorte contra un límite físico
(PEEP/FIO2 negativos, TV/RR recortados). El valor único de referencia es el de
3c: el 94.88 %/94.94 % que 3b daba para C2 en esta clase queda sustituido por la
medida de la fase D (efecto en la ventana de la frontera o la siguiente, con
`t_boundary` en lugar de `t_action`).

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
  no se resuelve en la variable observada que alimenta las features. En 3c los
  13 pares de tasa (`rftn20_rate` 8, `remi_up` 3, `ppf20_rate` 2) se
  reclasifican como **`overridden_by_base_plan`** con evidencia en el manifiesto
  v2: el cambio pedido llega al track observado `Orchestra/*_RATE` entre 0.0 y
  7.1 s después de `t_action` y el plan base lo repone (a) en 7 pares la serie
  observada es IDÉNTICA en ambas ramas desde `t_action` (el plan escribe su valor
  antes de que la observación muestree la diferencia) y (b) en 6 pares la
  diferencia se mantiene sólo 2.0–4.0 s (< 1 celda de 5 s) y se pierde en la
  agregación de `pk_v2`. El resto de causas `below_resolution` se mantienen:
  (b) `set_rr`/`set_tv` con `|delta|` por debajo del escalón del setpoint y
  (c) `set_peep` con `|peep_delta|` por debajo del escalón 1.
- `no_change_requested` — el valor pedido coincide con el valor base en
  `t_action` (tolerancia 5 % relativa): no se pide cambio alguno. **Unidades
  (re-verificado en 3c)**: la petición de `intervention_a` va en mg/min o
  mcg/min y el track `Orchestra/*_RATE` en mL/h; con PPF20 = 20 mg/mL y
  RFTN20 = 20 mcg/mL la conversión es `mL/h = (mg/min)·3`, de modo que los dos
  pares de este tipo están dentro de la tolerancia: 190231 pide 7.43 mg/min →
  22.29 mL/h frente a 22.181519 observados (**0.49 %**) y 190567 pide
  4.31 mg/min → 12.93 mL/h frente a 13.190063 (**1.97 %**).
- `case_ends` — el par termina antes de que exista una ventana con
  `t1 > t_action`.

Estas limitaciones son del GENERADOR (capa de observación, cadencia del plan
base y cuantización de las consignas), no del tokenizador. Se documentan, NO se
corrigen en esta iteración: el conjunto CF de entrenamiento se define como los
pares con `lever_effective = True` de `pairs_annotated.parquet`.

### 8.4 Prefijo fisiológico de la imagen: BLOQUEO (paso 3c)

**El prefijo fisiológico no es limpio y por eso la anotación corregida NO se
adopta: `pairs_annotated.parquet` (3b) sigue siendo la referencia y NINGUNA de
las dos anotaciones debe usarse para entrenar hasta que el generador corrija el
desfase descrito aquí.**

Medición (B1, bloqueante): exigiendo que las 14 variables de la imagen de
`windows_v4` (valores y máscaras `m_`/`m_raw_`) sean idénticas entre ramas en
toda celda que cierre en o antes de `t_effective`, **fallan 877 de los 6177
pares efectivos (14.198 %)**. Desglose por palanca: `set_fio2` 247, `set_rr` 145,
`set_tv` 131, `set_peep` 126, `fio2_down` 96, `peep_up` 88, `peep_down` 35,
`sevo_mac` 5, `eph_bolus` 3, `phen_bolus` 1. Sobre los tracks crudos (B2,
informativo) fallan 1773.

El fallo **no es una fuga hacia el futuro**: en los 877 pares la primera celda
divergente de la imagen cierra en o DESPUÉS de `t_action` (mínimo
`imagen − t_action` = 0.0 s, 0 pares con la imagen por delante del acto). Lo que
falla es que `t_effective` — definido para las consignas persistentes como
`t_divergence_raw`, medido sobre los `SET_*` crudos — es un **proxy tardío** del
instante en que la intervención actúa:

- 337 pares: la primera celda divergente de la imagen cierra entre 0 y 5 s
  ANTES de `t_effective` (una celda): consigna e imagen responden al mismo acto,
  pero el registro disperso `SET_*` se sella en otro punto de rejilla;
- 138 pares: entre 5 y 10 s antes de `t_effective` (dos celdas), el mismo
  mecanismo en las palancas con rango de cambio amplio (`set_fio2`, `set_tv`);
- 73 pares: más de 10 s, todos ellos consignas persistentes retrasadas (§8.2)
  donde la imagen responde al acto (o a la consigna) mucho después;
- 329 pares: la primera celda divergente de la imagen CIERRA exactamente en
  `t_effective`, es decir la celda `(t_effective−5, t_effective]` que B1 incluye
  por su "en o antes" ya contiene el instante de la divergencia.

Los `SET_*` del caso crudo son **dispersos** (NaN salvo cuando el plan escribe:
p. ej. `Primus/SET_RR_IPPV` sólo tiene muestras cada ~35 s en el par 196761), así
que en la rejilla se retienen con *hold* y el token `vent_*` cae al **proxy
medido** (`Primus/RR_CO2`, etc.) cuando el setpoint no está presente: por eso la
misma frontera explica también que el efecto del grupo aparezca más tarde en las
features (fase D). El efecto se localiza en la ventana de la frontera o la
siguiente en el 100 % de los pares para 18 de las 21 palancas; sólo falla el
criterio del 99 % en `set_rr` (93.94 %), `set_tv` (97.87 %) y `set_peep`
(94.50 %), con 49 pares de lag ≥ 2 y máximos de 157 ventanas (≈ 2.6 h) en
`set_peep`.

Qué habría que cambiar en el generador (propuesta, no ejecutada en esta
iteración): que la consigna registrada (`Primus/SET_*`) refleje el valor
APLICADO desde `split_t` en lugar de esperar a que el plan base reescriba la
consigna, y que la capa de observación escriba las consignas con la misma
cadencia que los medidos. Mientras eso no ocurra, `t_effective` no es
identificable de forma fiable para las 1797 consignas persistentes.

