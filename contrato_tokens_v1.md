Contrato de tokens v1 — anestesia_world
18 sept 2026 · @Ramses
Alcance
Este contrato fija qué tokens produce el tokenizador v1 a partir de data/windows_v2/ (rejilla 5 s, imagen de 14 variables, 61 columnas por ventana) y las 82 columnas clínicas comunes a las cuatro cohortes (real, synthetic_v5, vaso_reinf_v5, cf_v5). Es la fuente de verdad del tokenizador, junto al manifest.json de windows_v2; cualquier cambio en tokens, features o unidades pasa por aquí antes que por el código.
Cubre cuatro tipos de token: contexto (estático por caso), fármaco, ventilación y tiempo, más los tokens de estado que salen del autoencoder de fisiología con nested dropout. La v1 se restringe a la fase de mantenimiento (estabilidad): ventanas con t0 y t1 dentro de [opstart, opend). Inducción y educción quedan fuera. No cubre la arquitectura del modelo de transición ni el entrenamiento del autoencoder.
Bloque A: convenciones globales
El paso de transición es de 60 s (12 celdas de la rejilla de 5 s); t0 es el cierre de la primera celda y t1 = t0 + 60 el de la última. En entrenamiento el stride es de 60 s sin solape; para evaluación densa se permite stride de 5 s.
Convención
Valor
Paso de transición
60 s = 12 celdas; t0 cierre de la primera, t1 cierre de la última
Stride
60 s en train; 5 s disponible para evaluación
Estructura de token
(type_id, item_id, features[], mask)
Tipos de token
contexto, fármaco, ventilación, tiempo, estado
Máscara (3 estados)
0 = observado; 1 = ausente en cohorte (la variable no existe en la fuente); 2 = ausente en caso (existe en la fuente, falta aquí)
Normalización
por item_id, media y desviación calculadas solo sobre split=train; guardadas en el manifest del tokenizador
Unidades canónicas
mg (bolo), µg (bolo remi/fenilefrina), µg/mL (Ce propofol), ng/mL (Ce remi), µg/kg/min (nora), mbar, mL, %, MAC
Token de fuente
no existe; la cohorte solo puede inferirse por ausencias documentadas
Padding
secuencias de longitud variable por tipo, con máscara de padding aparte de la máscara de observación
Split
heredado bit a bit de windows_v2/split.parquet (sha 40935d1c…); no se recalcula
Fase
solo mantenimiento: t0 ≥ opstart y t1 < opend; las ventanas de inducción y educción no se tokenizan
Forward-fill de tasas de bomba
para integrar Ce, la última tasa se mantiene sin tope de edad (una bomba no se para porque falte un registro); el max_age de 30 s se aplica solo a la máscara m_ce
Metadato source
la salida del tokenizador conserva source y split como metadatos por ventana, nunca como token ni feature; sirven para el entrenamiento por etapas y para los sondeos de fuente
Metadatos de par CF
cada ventana de cf_v5 lleva pair_id, cf_role (base | intervencion), split_t y lever; permiten emparejar ramas y aplicar la pérdida contrafactual en el preentrenamiento. post_split = t1 > split_t: la ventana que contiene la intervención (t0 < split_t < t1) cuenta como la primera post-split, porque es donde empieza el efecto
Transformación de features sesgadas
Ce, Ce_max, bolo_en_ventana y dosis_acumulada se transforman con log1p antes del z-score (estadísticos de train); ventilación y contexto solo z-score
Alineación a la rejilla
toda serie leída de los parquets de caso (setpoints incluidos) se lleva a la rejilla de 5 s de windows_v2 por forward-fill antes de indexarse; nunca se indexa un array en el eje temporal crudo con posiciones de la rejilla
Val denso
las métricas de evaluación se calculan solo sobre ventanas dense=False (stride 60); las dense=True (stride 5, ~91 % de val) sirven para inspección de trayectorias, nunca para promediar métricas
Casos sin marcas de fase
se excluye todo caso real cuyas celdas en windows_v2 sean 100 % maintenance: son casos cuya fila clínica stub dejó a window.py sin opstart ni opend, así que no se puede acotar el mantenimiento y entrarían inducciones y educciones. El criterio es la fase ya escrita en windows_v2, no la nulidad de opend bajo una regla de deduplicación cualquiera: la fase la fijó window.py con su regla (no nulos sobre las 82 columnas), mientras que context_vocab deduplica sobre las 37 de contexto para sus tokens; son dos reglas para dos propósitos y no se unifican
La distinción entre ausente-en-cohorte y ausente-en-caso es la que evita que "no hay columna de bolo" en real se confunda con "bolo = 0". El embedding de máscara se aprende por estado; el valor de la feature en ambos casos de ausencia es 0 tras normalizar.
Bloque B: inventario v1
Entran siete fármacos, cinco setpoints de ventilación, el contexto preoperatorio, un token de tiempo y cuatro tokens de estado. Los 18 drug_rate raros (AMD, DEX2, DEX4, DOBU, DOPA, DTZ, EPI, FUT, MRN, NPS, NTG, OXY, PGE1, VASO, VEC, RFTN50 y afines) quedan fuera de v1 con nota.
Fármacos
item_id
Fuente de dosis
Concentración efectiva
Notas
propofol
PPF20_RATE, ppf_bolus
PPF20_CE (µg/mL), ya en windows_v2
modelo PK aleatorio por caso en sintéticos
remifentanilo
RFTN20_RATE, remi_bolus
RFTN20_CE (ng/mL), ya en windows_v2
RFTN50 fuera de v1
sevoflurano
SET_MAC
Primus/MAC (MAC espirado)
hipnótico principal en mantenimiento
fenilefrina
PHEN_RATE, phen_bolus
calculada (sección PK)
nunca desde PHEN_VOL
noradrenalina
NEPI_RATE
calculada (ganancia directa)

efedrina
eph_bolus (solo bolo: EPH_RATE es un track spike-hold, no una infusión)
calculada (sección PK)
token frágil: 50 % presencia en sintéticos; EPH_VOL no existe en real, así que la efedrina es inobservable en real (máscara 1 en todo el caso) y el token no se emite cuando la máscara es 1
rocuronio
roc_bolus en sintéticos (ROC_RATE es spike-hold); en real solo ROC_RATE (los bolos por bomba aparecen como picos de tasa)
calculada (Wierda, supuesto)
ROC_VOL no se usa en ninguna cohorte: en sintéticos es acumulado en mg·s y en real contiene cambios de jeringa
Ventilación (acciones)
item_id
Setpoint
Proxy medido si falta el setpoint
fio2
SET_FIO2
Primus/FIO2
tv
SET_TV_L
Primus/TV
rr
SET_RR_IPPV
Primus/RR_CO2
pip
SET_PIP
Primus/PIP_MBAR
peep
SET_INTER_PEEP
Primus/PEEP_MBAR
Los medidos de ventilación ya están en la imagen de 14 y por tanto en el estado; aquí solo entran como proxy con flag.
Contexto, tiempo y estado
• Contexto: antecedentes binarios por eje fisiopatológico, categóricos de procedimiento y riesgo, y continuos de demografía y laboratorio preoperatorio (tabla en la sección de mapeo).
• Tiempo: un token con t_since_opstart y fase (induction, maintenance, emergence) desde phase_from_clinical.
• Estado: 4 tokens × 32 dimensiones, cortes consecutivos del latente de 128 en orden nested; el primer token contiene las dimensiones que el decoder congelado reconstruye.
Bloque C: features por tipo de token
Cada tipo lleva un vector de features de longitud fija; los tokens del mismo tipo comparten proyección de entrada y se distinguen por el embedding de item_id. Los fármacos llevan dos concentraciones (inicio y fin de ventana) y no un delta, para que el modelo vea nivel y cambio a la vez.
Tipo
Features
Comentario
Fármaco
Ce(t0), Ce(t1), Ce_max en (t0, t1], bolo_en_ventana (dosis, 0 si no hay), dosis_acumulada desde anestart, bolo_observable (0/1)
dosis_acumulada es necesaria para la taquifilaxia de efedrina; bolo_observable = 0 en toda la cohorte real
Ventilación
valor(t0), valor(t1), es_proxy (0/1)
es_proxy = 1 cuando el valor viene del medido y no del setpoint; tras alinear los setpoints a la rejilla vale 0 en el 96-100 % de las ventanas y solo es no nulo en real, así que viaja como metadato y no como feature
Contexto continuo
valor normalizado
ausente = token no emitido
Contexto binario
ninguna (solo presencia)
el token existe si el antecedente es positivo
Contexto categórico
ninguna (un item_id por categoría)
p. ej. optype:colorectal, asa:3; categorías con < 50 casos en train se agrupan en otros
Tiempo
t_since_opstart (min, normalizado), sin fase (constante en mantenimiento)
t_since_opstart NaN → máscara 2
Estado
32 valores del latente
sin normalización adicional; el AE ya sale estandarizado
El valor de una feature enmascarada (estados 1 y 2) es 0 tras normalizar, y el embedding de estado de máscara se suma al token para que el modelo pueda distinguir "cero real" de "no observado".
Mapeo de contexto a ejes y columnas excluidas
De las 82 columnas clínicas entran 37 como contexto y se excluyen 45; el resto son identificadores, tiempos de rejilla o información posterior a anestart. Regla de admisión: un token de contexto solo puede construirse con información disponible antes de anestart.
Inventario de contexto v1 (decisión tras el gate 6 de context_vocab, 2026-09-18). Solo entran en v1 los tokens de demografía y riesgo global: age, sex:F, height, weight, bmi (continuos/binario) y asa, emop (categóricos). Son las únicas variables que el generador consume en su PK/PD y las únicas cuya distribución y presencia coinciden entre real y sintético (KS/TVD < 0.13, presencia ~100 % en las cuatro cohortes). El resto de la tabla de ejes (labs, gasometría, vía aérea, diagnóstico, procedimiento, comorbilidad) queda fuera del alcance de v1: se conserva en vocab.json con la marca scope = v2 como referencia, pero el tokenizador no los emite. Motivo: la capa clínica sintética no está alineada con INSPIRE (sondeo de fuente 0.9956, y 0.9935 solo con el patrón de presencia), y un contexto que identifica la cohorte impide que el sintético transfiera al régimen real.
Ejes
Eje
Columnas
Tipo de token
Demografía
age, height, weight, bmi
continuo
Demografía
sex
binario (sex:F)
Riesgo global
asa, emop
categórico
Procedimiento
optype, approach, position, ane_type
categórico
Diagnóstico
top_diagnosis_chapter
categórico (capítulo CIE)
Cardiovascular
preop_htn, preop_ecg (normal/anormal)
binario
Metabólico
preop_dm
binario
Metabólico
preop_gluc
continuo
Respiratorio
preop_pft (normal/anormal)
binario
Respiratorio
preop_pao2, preop_paco2, preop_sao2
continuo
Renal / electrolitos
preop_cr, preop_bun, preop_na, preop_k
continuo
Hepático / coagulación
preop_alb, preop_ast, preop_alt, preop_pt, preop_aptt, preop_plt, preop_hb
continuo
Ácido-base
preop_ph, preop_hco3, preop_be
continuo
Vía aérea
airway, cormack
categórico
Vía aérea
tubesize
continuo
El eje se guarda como atributo del item_id y se usa para inicializar el embedding de cada antecedente desde un embedding compartido del eje, de modo que los antecedentes infrecuentes hereden de sus vecinos.
Excluidas
Columnas
Motivo
caseid, subjectid, casestart, caseend, anestart, aneend, opstart, opend
identificadores y marcas de rejilla; ya usados por window.py
adm, dis, icu_days, death_inhosp
posteriores a la anestesia
intraop_ebl, intraop_uo, intraop_rbc, intraop_ffp, intraop_crystalloid, intraop_colloid
totales del caso completo; fuga del futuro
intraop_ppf, intraop_mdz, intraop_ftn, intraop_rocu, intraop_vecu, intraop_eph, intraop_phe, intraop_epi, intraop_ca
totales de fármaco del caso completo; fuga del futuro
diagnosis_count, top_diagnosis_code, lab_count, lab_unique_items, medication_count, medication_unique, common_route
derivadas del ingreso completo; se desconoce si están acotadas a preop
dx, opname
texto libre; fuera de v1
dltubesize, lmasize, iv1, iv2, aline1, aline2, cline1, cline2
accesos y dispositivos; bajo valor y presencia heterogénea
department
colapsado con optype (decisión)
Las derivadas pueden reentrar si se verifica que se calculan solo con datos previos a anestart.
Modelos PK por fármaco
La Ce de cada fármaco se calcula con las mismas ecuaciones que usa el generador en pk/, aplicadas también a la cohorte real. El criterio es coherencia entre cohortes, no fidelidad farmacológica: una Ce calculada distinto en real y en sintético sería una señal de cohorte gratuita.
Fármaco
Modelo
Parámetros
Estado
Propofol
3 compartimentos + Ce (Marsh / Schnider / Eleveld)
ke0 0.26 / 0.456 / ajustado por edad
ya en windows_v2 (PPF20_CE); en real se recalcula con Schnider salvo decisión contraria
Remifentanilo
Minto, 3 compartimentos + Ce
ke0 = 0.6 min⁻¹
ya en windows_v2 (RFTN20_CE)
Sevoflurano
sin PK; MAC espirado como concentración efectiva
—
Primus/MAC directo
Fenilefrina
bolo con decaimiento exponencial + infusión en estado estacionario
ke0 = ln2/300 s ≈ 0.1386 min⁻¹
igual que el generador
Efedrina
bolo con decaimiento exponencial; la taquifilaxia queda en el token vía dosis_acumulada
ke0 = ln2/600 s ≈ 0.0693 min⁻¹
igual que el generador
Noradrenalina
ganancia directa (estado estacionario inmediato)
t½ ≈ 2.5 min
Ce = tasa en µg/kg/min
Rocuronio
2 compartimentos + Ce (Wierda)
V1 = 0.07 L/kg, Cl = 3.7 mL/kg/min, ke0 = 0.105 min⁻¹; V2 = 0.13 L/kg, k12 = 0.093 min⁻¹, k21 = k12·V1/V2 (Wierda publicado); tercer compartimento desacoplado; en real ROC_RATE en mL/h se convierte con 10 mg/mL (concentración estándar, supuesto)
supuesto documentado; no existe en el generador. Módulo pk_tokens cerrado en iteración 3 (sha 1e3f695b…, gate 3 = 0.085 % / 0.090 %)
El código PK del tokenizador se versiona y su sha entra en el manifest del tokenizador, junto a la tabla de parámetros anterior.
Bolos en la cohorte real
Real no tiene columnas de bolo y no se infiere ninguno: los bolos administrados por bomba aparecen en RATE como picos breves (p. ej. 1200 mL/h) y la integración desde la tasa los captura. Las columnas *_VOL no se usan (contienen cambios de jeringa y artefactos de transición). En real bolo_observable = 0 para todos los fármacos; con alcance de mantenimiento, los bolos de inducción quedan fuera en cualquier caso.
El generador aplica variabilidad interindividual (IIV) lognormal (σ 0.20 en k, 0.15 en V y ke0) a propofol y remifentanilo y no la guarda. El token lleva la Ce poblacional en todas las cohortes; la desviación resultante (~20–25 % frente a la Ce individual del generador) se acepta como suelo de ruido realista, equivalente al de la PK poblacional en pacientes reales.
Bloque D: gates del tokenizador
El tokenizador solo se da por válido si pasa los diez criterios siguientes sobre el dataset completo; el resultado se registra en un REPORT_tokens.txt con el mismo formato que los informes de windows.
#
Criterio
Umbral
1
Cobertura por tipo de token y cohorte, con desglose por estado de máscara (0/1/2)
tabla completa; sin umbral, pero toda ausencia-en-cohorte documentada en este contrato
2
Prefijo CF idéntico a nivel de token hasta split_t y divergencia exactamente en el token de la palanca
100 % de los pares; 0 divergencias fuera de la palanca
3
Equivalencia del integrador: Ce del módulo frente al integrador de anessim ejecutado con los mismos parámetros poblacionales y la misma serie de tasa y bolos, para propofol y remifentanilo. Diagnóstico sin aserción: error frente al PPF20_CE / RFTN20_CE guardado (esperado ~20 % por IIV) en celdas de mantenimiento con tasa constante los 3 min previos y sin bolo
error relativo medio < 1 % frente al integrador de anessim; el diagnóstico frente al Ce guardado se reporta con mediana y p90
4
Ninguna columna excluida por fuga aparece en ningún token
0 columnas
5
Split idéntico a windows_v2
sha de split.parquet igual
6
Sondeo de fuente: regresión logística sobre tokens de contexto (sin máscaras) para predecir cohorte
accuracy ≤ 0.60 en val; si supera, identificar qué token la explica
7
Distribución real vs sintético de cada lab preoperatorio y de cada Ce en mantenimiento
KS por variable; se reporta, sin umbral, para decidir pesos de muestreo
8
Categorías con < 50 casos en train agrupadas en otros
0 categorías por debajo
9
Estadísticos de normalización calculados solo sobre train
verificable en el manifest
10
Tokens de estado: reconstrucción del decoder congelado desde el primer token de 32 igual a la del AE original
error idéntico ± 1e-4
El criterio 2 es el que protege el objetivo contrafactual: si un token diverge antes de la palanca, el modelo dispone de una pista sobre qué rama es la intervención.
Riesgos abiertos y decisiones pendientes
El riesgo principal no lo resuelve el contrato: el generador tiene PD lineales (MAP −5·Ce de propofol, nora +150 mmHg por µg/kg/min) y aporta 3.5 veces más ventanas que la cohorte real, así que la transición aprenderá primero la PD del simulador. La dosis-respuesta estratificada del modelo se evalúa contra real, nunca contra sintético, y el sintético entra con peso de muestreo reducido (valor a fijar en el contrato de entrenamiento).
Riesgo
Efecto
Mitigación en v1
Bolos no observables en real
Ce de fenilefrina infraestimada en real; señal de cohorte
bolo_observable; inferencia desde VOL donde hay doble conteo
Efedrina 50 % de presencia y bolo no observable en real
token frágil, embedding poco entrenado
se mantiene por decisión; se reporta aparte en el gate 1
Labs sintéticos posiblemente no acoplados a la fisiología
el modelo aprende que los labs no importan
gate 7; si KS es grande, ponderar o enmascarar labs en sintético
RR_CO2 sintético casi constante (11.8–16.4)
nada que aprender sobre frecuencia respiratoria desde sintéticos
documentado; sin acción en v1
12 % de synthetic_v5 sin ninguna presión arterial
tokens de estado con máscara 0 en ART_*
ya manejado por el AE; se reporta
Rocuronio sin PK en el generador
Ce inventada para una variable que el generador no modela
supuesto documentado; sin PD asociada, el token informa solo de exposición
Alcance solo mantenimiento
el 99 % de los bolos de propofol (inducción) quedan fuera; el problema de bolos no observables en real se reduce a fenilefrina y efedrina
ninguna; es una consecuencia favorable del alcance
Presencia parcial de bombas en real: PPF20 3512/6388 casos, RFTN20 4772, PHEN 127, NEPI 88, ROC 281, EPH 0
la dosis-respuesta de vasopresores contra real se evaluará con ~100 casos; la efedrina no se puede evaluar en real
reportar n por fármaco en cada evaluación; máscara estado 2 en los casos sin bomba
Efedrina inobservable en real
un token que solo existe en sintéticos identifica la cohorte
no emitir el token cuando la máscara es 1; el gate 6 lo verifica
IIV del generador no registrada
la Ce del token se desvía ~20–25 % de la que gobierna la PD sintética
aceptado como ruido realista; el modelo debe absorberlo, igual que en real
Capa clínica sintética no alineada con INSPIRE (gate 6 = 0.9956; solo presencia 0.9935; sin top-3 0.78)
el contexto identifica la cohorte por valores y por ausencias; el sintético dejaría de transferir al régimen real
decidido: contexto v1 mínimo (age, sex, height, weight, bmi, asa, emop); el resto fuera de alcance. Pendiente: repetir gate 6 con el mínimo y sondear la fisiología
Fuga de cohorte en la fisiología (sondeo sobre estadísticos de ventana: 0.99 con valores, 0.59 solo con ausencias; sin BIS/EMG, RR_CO2 y BIS sigue en 0.96; synthetic_v5 vs cf_v5 solo 0.65)
cualquier modelo entrenado en mezcla identifica la cohorte por la textura de las señales; el sintético no transfiere al régimen real
propuesta: entrenamiento por etapas (preentrenar en sintético con pares CF y pérdida contrafactual; ajuste fino en real con replay de pares CF a peso bajo). Adversario de dominio como experimento comparativo. Gate 6 se extiende a los tokens de estado cuando exista el AE
Decisiones pendientes
[x] PK de propofol en real: Schnider.
[x] department se colapsa con optype; umbral de agrupación de categorías: 50 casos en train.
[x] Alcance v1: solo fase de mantenimiento. t_since_opstart no necesita tratamiento especial: en sintéticos opstart = 0 y todas sus ventanas son mantenimiento.
[x] Forward-fill de tasas sin tope para la integración PK; max_age 30 s solo para la máscara. En real no se infieren bolos desde VOL: la integración desde RATE ya captura los bolos por bomba (iteración 2).
[x] Rocuronio y efedrina son solo-bolo en el generador; Wierda con V2 y k12 fijados.
[x] Gate 3 reformulado como equivalencia del integrador frente a anessim; el IIV del generador se acepta como suelo de ruido (iteración 2).
[ ] Efedrina en real: no se emite el token cuando la máscara es 1 (decidido).
[ ] Estrategia de entrenamiento: decidido, por etapas (preentrenar en sintético con pares CF; ajuste fino en real con replay de pares CF a peso bajo). El peso de muestreo deja de aplicar. Detalle en el contrato de entrenamiento.
[x] Contexto v1 mínimo: age, sex, height, weight, bmi, asa, emop. Labs, gasometría, vía aérea, diagnóstico, procedimiento y comorbilidad fuera de alcance (scope v2).
[ ] Sondeo de fuente sobre la fisiología: hecho (0.99 con valores). La fuga es del dataset completo, no del contexto.
[x] Gate 6 sobre el contexto v1: PASA (0.566; sin emop:0 sigue en 0.566). Módulo context_vocab cerrado en iteración 2 (sha 527024b5…).
[x] emop: presente al 11,7–13,7 % en real y 0,000 % en las tres cohortes sintéticas (emop:1 no existe en sintético). La ablación lo confirma como única fuente de señal: sin emop:0 y emop:1 el sondeo cae a 0.5000 exacto, y sin el grupo asa se queda en 0.5662. Decidido: se mantiene, y el embedding de emop:1 se inicializa desde el de emop:0 para que no llegue sin entrenar al ajuste fino. asa se mantiene con sus cinco categorías.