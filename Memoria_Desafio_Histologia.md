# Memoria Técnica: Desafío de Clasificación de Histologías Colorrectales (Deep Learning con PyTorch)

**Autor:** Pablo de la Cruz  
**Titulación:** Grado en Ingeniería Informática — Universidad de Alicante  
**Asignatura:** Desafíos de Programación  
**Convocatoria:** C2 (Noviembre 2026)  

---

## 1. Resumen ejecutivo y mapa de trazabilidad de la rúbrica

La presente memoria documenta el diseño, implementación, experimentación y análisis riguroso de modelos de Deep Learning para la detección de adenocarcinoma colorrectal primario a partir de imágenes histológicas. El proyecto se ha desarrollado íntegramente en **PyTorch**, priorizando el rigor metodológico, la justificación biológica y matemática de cada decisión y la ausencia total de fuga de datos (*data leakage*).

| Criterio de la Rúbrica | Nivel | Requisito del Enunciado | Sección en esta Memoria | Estado |
|---|---|---|---|---|
| Clasificación binaria | Básico (2 pts) | TUMOR vs. NO TUMOR (desbalance 1:7) | Sección 2.1 y 2.4 | Implementado y documentado |
| Métricas | Básico (2 pts) | ROC-AUC como métrica principal, Accuracy y F1 | Sección 3.3 y 4.3 | Evaluado en 10-CV continua |
| Validación cruzada | Básico (2 pts) | Stratified 10-fold CV idéntica en todos los modelos | Sección 3.1 | Congelado con SEED=42 |
| Baseline LeNet-5 | Básico (2 pts) | Réplica y mejora de la CNN base del profesor | Sección 4 | Medido (AUC 0.8218 ± 0.0291) |
| Arquitectura | Básico (2 pts) | Variación de profundidad, filtros, Dropout, BatchNorm | Sección 5 | En curso |
| Optimizadores | Básico (2 pts) | Comparación sistemática (SGD, Adam, AdamW) | Sección 5 | En curso |
| Comparación estadística | Básico (2 pts) | Wilcoxon pareado sobre los 10 pliegues | Sección 3.3 y 5.5 | Arnés implementado |
| Data Augmentation | Básico (2 pts) | Aumento de datos con justificación biológica | Sección 5.4 | Planificado |
| Clasificación multiclase | Medio (1 pto) | Clasificación en las 8 categorías del tejido (0 a 7) | Sección 6 | Planificado (Fase 5) |
| Preprocesado avanzado | Medio (1 pto) | Ecualización de histograma / filtros | Sección 6.2 | Planificado (Fase 5) |
| Análisis de errores | Medio (1 pto) | Estudio de ≥ 3 instancias fallidas con hipótesis | Sección 7 | Planificado (Fase 5) |
| Explicabilidad (XAI) | Medio (1 pto) | Mapas de calor de atención con Grad-CAM | Sección 8 | Planificado (Fase 5) |
| Ética e IA | Transversal | Declaración honesta y detallada del uso de IA | Sección 10 | Registrado en AI_LOG.md |

---

# PARTE I: DESARROLLO CRONOLÓGICO Y MODELO BASE (NIVEL BÁSICO)

---

## 2. Definición del problema, dataset y decisiones de ingeniería

### 2.1 Formulación del problema
El objetivo médico es asistir al patólogo en la identificación automatizada de tejido tumoral a partir de biopsias digitalizadas de colon. Matemáticamente, se formula como un problema de **clasificación binaria supervisada**:
Dado un tensor de imagen de entrada X de dimensiones 3 × H × W, el modelo debe estimar la probabilidad posterior P(Y = 1 | X), donde:
* Y = 1: Tejido patológico (**TUMOR**, clase positiva de interés clínico).
* Y = 0: Tejido no tumoral (**NO TUMOR**, clase negativa compuesta por 7 tipos de tejido estromal y epitelial sano).

### 2.2 Framework y confirmación de uso de PyTorch
El enunciado oficial sugiere TensorFlow/Keras pero autoriza explícitamente PyTorch si se cuenta con experiencia o preferencia técnica. Asimismo, en clase presencial el profesor reiteró verbalmente dicha libertad. Se adopta **PyTorch (v2.14.0)** debido a:
1. Control imperativo total del bucle de entrenamiento, permitiendo calcular métricas y aislar el estado sin efectos colaterales de grafos estáticos.
2. Soporte nativo para aceleración por hardware en Apple Silicon mediante el backend **MPS** (*Metal Performance Shaders*).
3. Transparencia didáctica en la defensa: cada operación tensorial y gradiente está explícitamente codificado.

### 2.3 Auditoría de datos y gestión de memoria (Hardware de 8 GB RAM)
El conjunto de datos procede del repositorio oficial de **Zenodo (record 53169, Kather et al., 2016)**, compuesto por 5.000 teselas RGB de 150×150 píxeles obtenidas a 20x de aumento óptico.

* **El desafío del hardware:** El entorno de desarrollo es un MacBook Pro M2 con 8 GB de memoria RAM unificada (compartida entre CPU, GPU del chip y el sistema operativo).
* **Decisión de empaquetado en uint8:** 
  * Un tensor global en `float32` de 5.000 × 150 × 150 × 3 × 4 bytes ocuparía **1,35 GB** en memoria bruta. Al sumarle los tensores de gradientes, mapas de características activados y buffers del sistema, el consumo superaría los 3–4 GB, provocando paginación a disco (*swap memory*) y ralentizando drásticamente la ejecución.
  * Al empaquetar el dataset en disco y memoria en enteros sin signo de 8 bits (`uint8`, [0, 255]), el archivo `images_150x150.npy` ocupa únicamente **321,87 MB** (exactamente 4 veces menos).
  * La conversión a tensores `float32` normalizados en el rango [0.0, 1.0] se realiza **estrictamente sobre la marcha por lote** en el `DataLoader`. Con un tamaño de lote de 32 imágenes, solo residen en la GPU **~8,6 MB** en cada iteración.

### 2.4 Análisis del desbalance de clases (1:7)
El dataset presenta un marcado desequilibrio de clases derivado de la binarización de 8 categorías equitativas (625 teselas por categoría):
* **TUMOR (clase 1):** 625 muestras (**12,50 %** del total).
* **NO TUMOR (clase 0):** 4.375 muestras (**87,50 %** del total), compuestas por: STROMA (625), COMPLEX (625), LYMPHO (625), DEBRIS (625), MUCOSA (625), ADIPOSE (625) y EMPTY (625).

**Consecuencia directa en la evaluación:**
Un clasificador trivial o "tonto" que prediga ciegamente siempre la clase negativa obtendría una tasa de acierto del **87,50 %** (Exactitud / *Accuracy*), pese a tener una utilidad clínica nula al no detectar ningún paciente con cáncer. Por esta razón, el enunciado exige la métrica **ROC-AUC** como indicador primario de rendimiento, al ser insensible al umbral de decisión rígido y evaluar la capacidad del modelo para ordenar los casos según su riesgo real.

---

## 3. Protocolo experimental y arnés de validación (10-CV)

### 3.1 Validación cruzada estratificada de 10 pliegues (10-CV)
Para garantizar estimaciones insesgadas, se aplica una validación cruzada estratificada de 10 pliegues:
* Se fija de forma reproducible `SEED = 42` y se guardan las particiones en el fichero estructurado `data/splits_10cv.json`.
* **Estratificación multiclase:** Las particiones se han generado sobre las 8 clases originales antes de binarizar. Esto garantiza dos propiedades esenciales:
  1. Cada pliegue de test contiene exactamente 500 imágenes con la proporción binaria exacta: pliegues 0 a 4 (63 TUMOR y 437 NO TUMOR) y pliegues 5 a 9 (62 TUMOR y 438 NO TUMOR).
  2. La clase negativa (NO TUMOR) tiene exactamente la misma representación de los 7 tejidos sanos en todos los pliegues (62–63 muestras de cada uno), impidiendo que un pliegue esté sesgado hacia un tejido particular.
  3. Estas mismas particiones se reutilizan idénticas en el Nivel Medio, garantizando la consistencia estadística entre fases.

### 3.2 Prevención de fuga de datos (Zero Data Leakage)
1. **Estadísticos de clase:** La ponderación de la función de pérdida para compensar el desbalance se calcula en cada pliegue $k$ utilizando única y exclusivamente las etiquetas de entrenamiento (`train_labels`), sin consultar jamás el conjunto de test.
2. **Normalización:** La transformación lineal a $[0.0, 1.0]$ mediante división por 255.0 es una escala fija y determinista sin estado (*stateless*), por lo que no introduce filtración entre conjuntos.
3. **Aislamiento en cada iteración:** Se utiliza el patrón de diseño de fábrica (`model_factory`), instanciando un modelo completamente nuevo con pesos reinicializados en cada pliegue.

### 3.3 Protocolo estadístico y cálculo de ROC-AUC
* **Probabilidad continua:** Se extraen las probabilidades continuas de la clase positiva tras la función softmax (P(Y = 1 | X)). Se prohíbe el uso de `np.round`, que genera empates artificiales.
* **Test de rangos con signo de Wilcoxon:** Para comparar dos modelos A y B, se contrasta la diferencia pareada de sus 10 valores de ROC-AUC pliegue a pliegue (N = 10). El p-valor mínimo medible a dos colas es:
  p_min = 2 / 2^10 = 2 / 1024 ≈ 0,001953
* **Corrección de Holm-Bonferroni:** En comparaciones múltiples, se aplica el procedimiento step-down de Holm para acotar la tasa de error por familia (FWER) al nivel de significación α = 0,05.

---

## 4. Modelo Base: Réplica crítica de LeNet-5

### 4.1 Topología y parámetros de LeNet-5
Como punto de partida estipulado por el profesor en `Colorectal_histology_LeNet5_notebook.ipynb`, se replica la arquitectura clásica convolucional LeNet-5 adaptada a imágenes en color RGB (3 canales):
1. **Entrada:** Imágenes redimensionadas a **32×32** píxeles.
2. **Bloque Convolucional 1:** `Conv2D(3, 6, kernel=5x5)` + activación `Sigmoid` + `MaxPool2d(2x2, stride=2)` $\to$ Mapa de características de salida: $(6, 14, 14)$.
3. **Bloque Convolucional 2:** `Conv2D(6, 16, kernel=5x5)` + activación `Sigmoid` + `MaxPool2d(2x2, stride=2)` $\to$ Mapa de características de salida: $(16, 5, 5)$.
4. **Capa de aplanado (*Flatten*):** Vector unidimensional de $16 \times 5 \times 5 = 400$ características.
5. **Capa Densa 1:** `Linear(400, 120)` + activación `Sigmoid`.
6. **Capa Densa 2:** `Linear(120, 84)` + activación `Sigmoid`.
7. **Capa Densa de Salida:** `Linear(84, 2)` (produce los logits no normalizados para la clasificación binaria).
* **Parámetros totales entrenables:** **61.326 pesos**.

### 4.2 Hiperparámetros del Baseline
* **Optimizador:** Descenso por Gradiente Estocástico (`SGD`) con tasa de aprendizaje fija $lr = 0,01$, sin momentum ni decaimiento de pesos.
* **Épocas:** 25 épocas por pliegue.
* **Tamaño de lote (*batch size*):** 32 muestras.
* **Pérdida:** `nn.CrossEntropyLoss(weight=weight_tensor)` donde los pesos de clase se calculan como w_c = N / (C · N_c), resultando en pesos aproximados de w_0 ≈ 0,57 y w_1 ≈ 4,00 (relación 1:7).

### 4.3 Auditoría crítica del notebook del profesor
Durante la fase de análisis del notebook de referencia en tf.keras, se descubrieron dos anomalías técnicas de gran impacto que fueron corregidas en nuestra implementación en PyTorch:
1. **Defecto de redondeo en el cálculo de ROC-AUC:**
   El notebook ejecutaba:
   `metrics.roc_auc_score(y_test, np.round(y_scores[:, 1], 2))`
   Al redondear a 2 decimales, cuando una red tiene salidas poco dispersas (como ocurre con las sigmoides), todas las probabilidades colapsan en el mismo valor discreto (`0.45`). La función de scikit-learn trata todas las muestras como empates, provocando que el ROC-AUC se desplome artificialmente a **0,5000** (equivalente a tirar una moneda al aire). En PyTorch, se evalúan las probabilidades continuas reales extraídas con softmax de 32 bits.
2. **Definición de etiquetas:**
   El notebook invertía el sentido habitual asignando `y[y > 0] = 1`, tratando la clase mayoritaria (sana) como positiva. En esta memoria y en el código de entrega, la clase positiva 1 es estrictamente **TUMOR**, respetando la convención médica y la interpretación diagnóstica estándar.

### 4.4 Resultados experimentales del Baseline (10-CV)
La ejecución completa de los 10 pliegues del Baseline en Apple Silicon (MPS) completó en **178,71 segundos** (~17,8 s/pliegue):

| Pliegue | Muestras Train | Muestras Test | ROC-AUC | Accuracy | F1-Score (TUMOR) |
|---|---|---|---|---|---|
| Fold 0 | 4.500 | 500 | 0,8379 | 0,8740 | 0,0000 |
| Fold 1 | 4.500 | 500 | 0,8585 | 0,8740 | 0,0000 |
| Fold 2 | 4.500 | 500 | 0,8250 | 0,8740 | 0,0000 |
| Fold 3 | 4.500 | 500 | 0,8269 | 0,8740 | 0,0000 |
| Fold 4 | 4.500 | 500 | 0,8360 | 0,8740 | 0,0000 |
| Fold 5 | 4.500 | 500 | 0,8441 | 0,8760 | 0,0000 |
| Fold 6 | 4.500 | 500 | 0,8232 | 0,8760 | 0,0000 |
| Fold 7 | 4.500 | 500 | 0,8163 | 0,8760 | 0,0000 |
| Fold 8 | 4.500 | 500 | 0,7963 | 0,8760 | 0,0000 |
| Fold 9 | 4.500 | 500 | 0,7540 | 0,8760 | 0,0000 |
| **Media ± Std** | — | — | **0,8218 ± 0,0291** | **0,8750 ± 0,0011** | **0,0000 ± 0,0000** |

* **Matriz de confusión global agregada (5.000 instancias OOF):**
  * NO TUMOR predicho: 5.000 (4.375 verdaderos negativos, 625 falsos negativos).
  * TUMOR predicho: 0 (0 verdaderos positivos, 0 falsos positivos).

### 4.5 Diagnóstico técnico: ¿Por qué F1 es 0 y ROC-AUC es 0,8218?
El análisis cuantitativo de las probabilidades predichas `y_prob` sobre las 5.000 muestras OOF arrojó la siguiente distribución:
* Valor mínimo: 0,4542 | Valor máximo: 0,4548.
* Rango dinámico: apenas **0,00067**.
* Media en TUMOR real: **0,45460** | Media en NO TUMOR real: **0,45454**.

**Explicación matemática del fenómeno:**
La combinación de activaciones Sigmoid y optimizador SGD básico provoca un severo **desvanecimiento del gradiente**. La derivada de la sigmoide σ'(z) ≤ 0,25 reduce exponencialmente el gradiente que retropropaga hacia las primeras capas (≈ 0,25⁴ ≈ 0,0039). Como consecuencia, los pesos no logran escalar los logits de salida lo suficiente como para vencer el umbral estándar de decisión τ = 0,5. Todas las predicciones discretas se asignan a la clase 0, provocando un F1 = 0.

Sin embargo, el gradiente sí fue suficiente para orientar los pesos en la dirección correcta: las muestras tumorales reciben sistemáticamente probabilidades mayores que las sanas. Debido a que el **ROC-AUC mide la concordancia de rangos (estadístico U de Mann-Whitney)** a lo largo de todos los umbrales posibles, la red alcanza un meritorio **0,8218**, demostrando que extrae patrones discriminativos pero carece de calibración en la capa de salida.

---

# PARTE II: EXPERIMENTACIÓN SISTEMÁTICA Y MEJORAS (NIVEL BÁSICO)

## 5. Experimentación de mejoras sobre el modelo base

### 5.1 Experimento 1 (Arquitectura y activaciones): LeNet-5 con ReLU
Para aislar rigurosamente el impacto de la función de activación, se mantiene intacta la topología de LeNet-5 (61.326 parámetros), el optimizador (SGD lr=0.01), el número de épocas (25), la resolución (32×32) y las particiones (10-CV con SEED=42), sustituyendo exclusivamente las 4 funciones Sigmoid por activaciones **ReLU** (f(x) = max(0, x)).

#### Hipótesis arquitectónica
En la red original, la función sigmoide comprime las salidas en el rango (0, 1), teniendo una derivada máxima de apenas 0,25 en z=0 que decae rápidamente hacia 0. Al encadenar 5 capas, el gradiente que alcanza a la primera convolución se reduce en un factor aproximado de 0,25⁴ ≈ 0,0039, provocando una severa inanición de gradiente (*gradient starvation*) con SGD básico. Al emplear ReLU, la derivada es constante e igual a 1,0 para cualquier activación positiva, permitiendo que el gradiente fluya sin atenuación hacia las capas convolucionales tempranas para aprender detectores de bordes y texturas tisulares discriminativas.

#### Resultados medidos en 10-CV (LeNet-5 ReLU)
La evaluación en Apple Silicon (MPS) sobre los 10 pliegues produjo los siguientes resultados:

| Pliegue | Muestras Train | Muestras Test | ROC-AUC | Accuracy | F1-Score (TUMOR) |
|---|---|---|---|---|---|
| Fold 0 | 4.500 | 500 | 0,9450 | 0,8540 | 0,6054 |
| Fold 1 | 4.500 | 500 | 0,9376 | 0,8620 | 0,6188 |
| Fold 2 | 4.500 | 500 | 0,9334 | 0,8280 | 0,5784 |
| Fold 3 | 4.500 | 500 | 0,9470 | 0,8600 | 0,6354 |
| Fold 4 | 4.500 | 500 | 0,9408 | 0,8760 | 0,6517 |
| Fold 5 | 4.500 | 500 | 0,9421 | 0,8460 | 0,6010 |
| Fold 6 | 4.500 | 500 | 0,9270 | 0,8440 | 0,5806 |
| Fold 7 | 4.500 | 500 | 0,9357 | 0,8460 | 0,5926 |
| Fold 8 | 4.500 | 500 | 0,9368 | 0,8100 | 0,5662 |
| Fold 9 | 4.500 | 500 | 0,9035 | 0,8040 | 0,5149 |
| **Media ± Std** | — | — | **0,9349 ± 0,0125** | **0,8430 ± 0,0229** | **0,5945 ± 0,0384** |

#### Comparativa directa frente al Baseline

| Modelo | Activación | ROC-AUC (Media ± Std) | Accuracy (Media ± Std) | F1-Score (Media ± Std) |
|---|---|---|---|---|
| `baseline_lenet5` | Sigmoid | 0,8218 ± 0,0291 | 0,8750 ± 0,0011 | 0,0000 ± 0,0000 |
| `lenet5_relu` | ReLU | **0,9349 ± 0,0125** | 0,8430 ± 0,0229 | **0,5945 ± 0,0384** |

#### Contraste estadístico de hipótesis (Test de Wilcoxon)
Para verificar si la mejora observada en ROC-AUC (+0,1131 de media) es estadísticamente significativa y no producto de la variabilidad entre particiones, se ejecutó la prueba no paramétrica de rangos con signo de Wilcoxon sobre los 10 pares de pliegues emparejados:
* **Pliegues ganados:** `lenet5_relu` supera a `baseline_lenet5` en **10 de los 10 pliegues** (100 % de victorias).
* **Diferencia de medianas:** **+0,1078** en ROC-AUC a favor de ReLU.
* **p-valor:** **0,00195** (el mínimo teórico posible con N = 10 pares: 2 / 2^10 = 2 / 1024 = 0,001953).
* **Conclusión estadística:** Dado que p = 0,00195 < 0,05, la superioridad de LeNet-5 con ReLU sobre el baseline con Sigmoid es **estadísticamente significativa** con un nivel de confianza del 99,8 %.

#### Análisis de la matriz de confusión y la aparente paradoja del Accuracy
Al analizar las 5.000 predicciones Out-Of-Fold (OOF) se resuelve la aparente paradoja de por qué el Accuracy disminuye ligeramente (de 87,50 % a 84,30 %) mientras el ROC-AUC y el F1 experimentan un salto masivo:

* **Matriz de confusión global de LeNet-5 ReLU (5.000 muestras OOF):**
  * Verdaderos Negativos (NO TUMOR detectado): 3.643 de 4.375
  * Falsos Positivos (Falsa alarma en tejido sano): 732 de 4.375
  * Falsos Negativos (Tumor no detectado): 53 de 625
  * Verdaderos Positivos (Tumor detectado correctamente): **572 de 625**

* **Interpretación clínica y matemática:**
  1. **Sensibilidad (Recall) médica:** El baseline no detectaba ni un solo tumor (0 de 625, sensibilidad = 0,0 %). En contraste, LeNet-5 con ReLU detecta con éxito **572 de los 625 casos de cáncer** (sensibilidad del **91,52 %**).
  2. **Origen de la caída de Accuracy:** Para cazar el 91,52 % de los tumores en un entorno desbalanceado (1:7) con pérdida ponderada, el clasificador asume un umbral de sospecha que genera 732 falsos positivos en tejidos benignos dudosos. Estos 732 falsos positivos sumados a los 53 falsos negativos representan 785 fallos sobre 5.000 (Accuracy = 4.215 / 5.000 = 84,30 %).
  3. **Significado del ROC-AUC (0,9349):** La métrica ROC-AUC refleja que si se toma una imagen al azar con TUMOR y otra imagen al azar de tejido SANO, en el **93,49 % de las ocasiones el modelo asigna una probabilidad de tumor más alta a la imagen enferma que a la sana**. Demuestra una capacidad intrínseca sobresaliente de ordenación diagnóstica, inmune al sesgo del umbral fijo de 0,5.

---

### 5.2 Experimento 2 (Optimizadores): Comparación sistemática de SGD vs. Adam / AdamW
*(Próximo paso: evaluación de momentos adaptativos y decaimiento de pesos)*

### 5.3 Experimento 3 (Resolución de entrada y CNN moderna): Escalado a 150×150 con BatchNorm y Dropout
*(Planificado)*

### 5.4 Experimento 4 (Aumento de datos con fundamentación biológica)
*(Planificado)*

### 5.5 Contraste estadístico global consolidado
*(Planificado al cierre del Nivel Básico)*

---

# PARTE III: PROFUNDIZACIÓN, ANÁLISIS DE ERRORES Y EXPLICABILIDAD (NIVEL MEDIO)

## 6. Clasificación multiclase en 8 categorías tisulares
*(Fase 5: Multiclase)*

## 7. Análisis cuantitativo y cualitativo de errores (≥ 3 instancias)
*(Fase 5: Diagnóstico patológico de fallos de la red con imágenes)*

## 8. Explicabilidad mediante mapas de activación Grad-CAM
*(Fase 5: Visualización de las regiones donde la CNN enfoca la decisión)*

---

# PARTE IV: CONCLUSIONES Y ÉTICA

## 9. Conclusiones y limitaciones del estudio
*(A redactar al cierre del proyecto)*

## 10. Ética y uso de IA generativa
*(Declaración íntegra y veraz según los registros de AI_LOG.md)*
