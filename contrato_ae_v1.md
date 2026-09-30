> ⚠️ AVISO DE DESACTUALIZACIÓN (2026-09-30): ESTE CONTRATO YA NO REFLEJA DEL
> TODO EL ESTADO ACTUAL. EL GATE 3 ANTIGUO (INDEPENDENCIA DE MÁSCARA CON
> DIFERENCIA MEDIANA < 1 lpm) FUE SUSTITUIDO EN LA ITER 2 POR UN GATE 3 NUEVO
> (ERROR SIN ART < 2 lpm). EL GATE 6 SOLO-POR-CASO AHORA SE EVALÚA POR CASO Y
> POR CELDA. LA VARIANTE QUEDÓ DECIDIDA Y CERRADA: B (ae_bal), ENTRENADA Y
> CONGELADA. VÉASE MEMORIA.md.

Contrato del autoencoder de fisiología v1 — anestesia_world
20 sept 2026 · @Ramses
Alcance
El autoencoder comprime la fisiología de una celda de la rejilla de 5 s de data/windows_v2/ en un latente ordenado de 32 dimensiones, y su decoder queda congelado como lengua franca del resto del sistema: todo lo que el modelo de transición prediga se lee a través de él.
Este contrato fija la entrada, la pérdida, la arquitectura, el formato del estado y los gates del AE. No cubre el modelo de transición ni el entrenamiento por etapas; sí fija el formato de estado que ambos consumen, y cierra el gate 10 del contrato de tokens v1 (contrato_tokens_v1.md), que queda pendiente hasta que existan estos pesos.
Alcance temporal: solo celdas de fase maintenance, coherente con la v1 del tokenizador. Split heredado bit a bit de windows_v2/split.parquet (sha 40935d1c…); no se recalcula. Los 35 casos reales sin marcas de fase y el caso 4476 sin celdas quedan fuera, igual que en el tokenizador.
Entradas, normalización y pérdida
La entrada son 28 canales: los 14 valores de la imagen z-scoreados más sus 14 máscaras. Sin las máscaras, un valor ausente normalizado a 0 sería indistinguible de un valor medio real.
Elemento
Valor
Variables
las 14 de la imagen de windows_v2: BIS/BIS, BIS/EMG, Solar8000/HR, PLETH_SPO2, Primus/ETCO2, PEEP_MBAR, PIP_MBAR, MV, TV, RR_CO2, Solar8000/BT, ART_MBP, ART_SBP, ART_DBP
Fuera
Primus/FIO2: su setpoint ya es un token de acción y el medido duplicaría esa señal en el estado
Canales de entrada
14 valores z-scoreados + 14 máscaras (0/1) = 28
Normalización
media y desviación por variable, calculadas solo sobre celdas de split=train con máscara 1 (ddof = 0; std = 0 → 1.0)
Valor enmascarado
0 tras normalizar, con su canal de máscara a 0
Salida del decoder
14 valores; las máscaras no se reconstruyen
La pérdida es MSE calculada solo sobre celdas con máscara 1, promediada primero por variable y después entre las 14. Promediar por variable evita que las tres presiones arteriales pesen el triple que el BIS por el mero hecho de ser tres columnas, y que las variables de rango ancho dominen a las de rango estrecho.
Las métricas de evaluación se promedian primero por caso y luego entre casos, para que un caso de seis horas no pese diez veces más que uno de cuarenta minutos.
Arquitectura y nested dropout
Un MLP basta: para una sola celda no hay estructura espacial ni temporal que explotar.
Elemento
Valor
Encoder
28 → 256 → 256 → 32, GELU
Decoder
32 → 256 → 256 → 14, GELU
Normalización de capa
en las ocultas sí; nunca sobre el latente, porque rompería el orden
Nested dropout
por muestra: se sortea la longitud del prefijo k ~ Uniforme{1..32} y se ponen a cero las dimensiones k+1..32 antes del decoder
Inferencia
siempre con las 32 dimensiones (sin dropout)
El latente de 32 es expansivo frente a las 14 variables de entrada, y eso es deliberado: buscamos orden y redundancia, no compresión. Con un AE sobrecompleto el error de reconstrucción global será bajo casi por construcción, así que no es la métrica interesante; lo es el perfil de orden (gate 1). El nested dropout es justamente lo que impide que un AE sobrecompleto aprenda la identidad.
Se elige uniforme en lugar de la geométrica del artículo original de Rippel porque con solo 32 dimensiones la geométrica concentra demasiada masa en prefijos muy cortos. Si el perfil de orden sale plano, se prueba geométrica con p ≈ 0,08 y se reporta la comparación de ambos perfiles.
Entrenamiento y las dos variantes de cohorte
Se entrenan dos variantes con el mismo código, la misma semilla y los mismos hiperparámetros, y se comparan antes de elegir. No es una duda que se pueda zanjar razonando: es un experimento pequeño.
Variante
Datos de entrenamiento
Argumento a favor
A — ae_real
solo celdas de source=real, split=train
el latente es el idioma del paciente real y el sintético se traduce a él; el decoder congelado queda calibrado sobre lo que al final queremos predecir
B — ae_bal
las cuatro cohortes, split=train, con muestreo equilibrado al 50 % real / 50 % sintético
el preentrenamiento sintético decodifica por una representación que sí vio; evita que la mayoría sintética (3,5×) decida qué dimensiones van primero
La comparación se decide con tres números, reportados para ambas variantes: error de reconstrucción por variable y por cohorte en val, perfil de orden (gate 1), y sondeo de fuente sobre el latente (gate 6). La preocupación concreta con A es que las rarezas del sintético (12 % de synthetic_v5 sin presión arterial, RR_CO2 casi constante, BIS/EMG sin ruido realista) reconstruyan mal y hagan ruidoso el preentrenamiento; con B, que el orden de las dimensiones lo fije el sintético.
Hiperparámetro
Valor
Optimizador
AdamW, lr 1e-3, weight decay 1e-4
Batch
4096 celdas
Schedule
cosine con 5 % de warm-up
Parada
por pérdida de val con paciencia 5, sobre una submuestra de val FIJA (las mismas celdas en todas las épocas) y con el MISMO criterio para las dos variantes (val real + val sintético); el tope de épocas se fija lo bastante alto para que pare la paciencia y no el tope
Semilla
42, fijada y registrada; dos ejecuciones con la misma semilla dan pesos idénticos
Muestreo (B)
por caso y luego por celda, para que un caso largo no domine
Formato del estado, warm-up y artefactos
El estado que consume el modelo de transición es un vector de 128 dimensiones troceado en los 4 tokens de 32 del contrato de tokens:
z(t0) = [ enc(celda t0)  |  warmup(10 min previos) ]
           32 dims           96 dims
         token 1           tokens 2, 3 y 4
El token 1 es exactamente el latente del AE, y es el único que el decoder congelado lee. Los tokens 2 a 4 son historia oculta: no los produce el AE, los inicializa el warm-up y los evoluciona la transición.
Warm-up. Un encoder pequeño (GRU o atención, a decidir en el contrato de transición) consume los latentes del AE de los 10 anclajes de ventana previos (t0 − 600, t0 − 540, …, t0 − 60; es decir 10 latentes espaciados 60 s) y produce las 96 dimensiones. Resolución de 60 s sobre 10 minutos es suficiente para separar "venía bajando" de "bajó por el fármaco", y es doce veces más barato que consumir las 120 celdas de 5 s. Una ventana con menos de 10 minutos de mantenimiento por delante se rellena con lo disponible y lleva un campo warmup_len con cuántos anclajes reales entraron. La implementación vive en el módulo de transición; aquí solo se fija el formato.
Artefactos. data/ae_v1/<variante>/ con encoder.pt, decoder.pt, norm_stats.json (media y desviación por variable, solo train) y manifest_ae.json (sha de los pesos, del código y de este contrato, semilla, hiperparámetros, perfil de orden y métricas por cohorte). Variantes: ae_real y ae_bal.
Los latentes no se materializan. El encoder es un MLP diminuto y el DataLoader lo aplica al vuelo; precalcular los latentes de t0 y t1 de las 14,5 M ventanas serían unos 3,7 GB de parquet para ahorrar un producto de matrices.
Gates
Cada variante pasa los nueve criterios por separado; el resultado se registra en REPORT_ae.txt con el mismo formato que los informes anteriores.
#
Criterio
Umbral
1
Perfil de orden: error de reconstrucción en función de k = 1..32 dimensiones retenidas, global y por variable, en val real y val sintético
monótono no creciente en k, y err(k=8) − err(k=32) ≤ 0,2 × (err(k=1) − err(k=32)): los 8 primeros capturan el 80 % de la reducción alcanzable
2
Reconstrucción por variable y cohorte en unidades físicas, solo máscara 1, promediada por caso
ART_MBP error mediano < 2 mmHg y HR < 2 lpm; las otras 12 se reportan sin umbral
3
Independencia de máscara: reconstrucción de HR con y sin ART observado en los mismos casos
diferencia mediana < 1 lpm
4
Congelación y determinismo: dos pasadas del decoder sobre el mismo latente, y dos entrenamientos con la misma semilla
salida idéntica bit a bit; sha de los pesos en el manifest
5
Gate 10 del contrato de tokens: reconstrucción desde el token de estado 1 igual a la del AE original
± 1e-4
6
Sondeo de fuente sobre el latente (32 dims, y solo las 8 primeras), real vs sintético en val, 5-fold por caso
informativo, sin umbral; se espera alto, como la fisiología cruda (0,99)
7
Sin fuga temporal: el encoder solo ve la celda t0
mecánico, sobre las columnas leídas
8
Estadísticos de normalización calculados solo sobre train; split heredado de windows_v2
verificable en el manifest
9
Comparación A vs B: los tres números de la sección anterior, lado a lado
sin umbral; es la evidencia con la que se elige variante
El gate 1 es el que decide si el nested dropout hizo su trabajo: si el perfil es plano, el latente no está ordenado y el troceado en 4 tokens pierde su sentido. El gate 6 mide cuánta cohorte se cuela en el latente; se documenta, no se combate, porque el entrenamiento por etapas es lo que se encarga de eso.
Riesgos abiertos y decisiones pendientes
Riesgo
Efecto
Mitigación en v1
La cohorte se cuela en el latente
los tokens de estado identifican la fuente, como la fisiología cruda (0,99)
aceptado; lo resuelve el entrenamiento por etapas, y el gate 6 lo mide
AE sobrecompleto (32 > 14)
reconstrucción casi perfecta que no dice nada de la calidad del latente
la métrica de calidad es el perfil de orden (gate 1), no el error global
Uniforme{1..32} puede castigar demasiado los prefijos largos
error de reconstrucción peor del necesario con las 32 dimensiones
si el gate 2 falla, probar geométrica p ≈ 0,08 y reportar ambos perfiles
12 % de synthetic_v5 sin presión arterial
el AE ve tres canales enmascarados a la vez en muchos casos sintéticos
canales de máscara en la entrada y gate 3
RR_CO2 sintético casi constante y BIS/EMG sin ruido
esas dimensiones del latente se entrenan con variabilidad irreal en la variante B
comparación A vs B (gate 9)
El decoder congelado se calibra sobre la mezcla que se elija
si luego cambiamos de variante, hay que reentrenar todo lo que dependa de él
la elección se cierra antes de entrenar la transición y su sha entra en el manifest de la transición
Decisiones pendientes
[ ] Elegir variante (A ae_real o B ae_bal) con la evidencia del gate 9. DECIDIDO: B (ae_bal), por regularidad geométrica del latente entre etapas y no por precisión de reconstrucción: ambas variantes quedan un orden de magnitud por debajo del ruido del monitor (0,32 frente a 0,42 mmHg de MAP), pero el perfil de orden de A sobre sintético es grumoso (escalones en k=7 y k=12), señal de que las celdas sintéticas viven en una región del latente que el modelo entrenado solo con real no organizó, y ahí es donde trabaja el preentrenamiento. El decoder que se congela debe salir de una ejecución convergida.
[ ] Arquitectura exacta del warm-up (GRU o atención) y si sus 96 dimensiones se entrenan junto con la transición o por separado: va en el contrato de transición.
[ ] Si el gate 2 no se cumple con 32 dimensiones, decidir entre subir el latente o relajar el umbral; no se toca el troceado 4×32 sin revisar el contrato de tokens.