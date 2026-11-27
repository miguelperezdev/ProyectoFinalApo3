# Fase 1 · Comprensión del negocio

## 1.1 Contexto y problema

En las plazas de mercado, centros de acopio y empacadoras, las frutas se clasifican **a mano** según su apariencia: color, manchas, golpes, podredumbre y tamaño. El proceso es:

- **lento**: una persona inspecciona fruta por fruta;
- **subjetivo**: dos clasificadores pueden asignar categorías distintas a la misma fruta, y el criterio cambia con el cansancio y la iluminación;
- **poco estandarizado**: sin una categoría objetiva, el precio de venta se negocia caso por caso.

En Colombia se pierden o desperdician cerca de **9,76 millones de toneladas de alimentos al año, el 34 % de la oferta disponible** [DNP, 2016], y a nivel mundial alrededor del 14 % de los alimentos se pierde entre la cosecha y la venta minorista [FAO, 2019]. Una clasificación temprana y consistente permite destinar cada fruta a su mejor uso (venta en fresco, procesamiento o compostaje) en lugar de descartar lotes completos.

## 1.2 Objetivos

**Objetivo de negocio:** apoyar la clasificación de frutas frescas con un sistema automático, rápido y consistente, que funcione con una cámara de celular o una webcam y un computador sin GPU.

**Objetivo analítico:** a partir de una fotografía de una fruta individual, entregar:

1. la **fruta** (manzana, banano, guayaba, limón, naranja o granada);
2. un **grado de calidad** de tres categorías:

   | Grado | Significado comercial | Análogo en las normas Codex |
   |---|---|---|
   | **Primera** | Fruta sana, venta en fresco a precio pleno | Categoría "Extra" / I |
   | **Segunda** | Dudosa o con defectos leves: revisión manual, venta a menor precio o procesamiento | Categoría II |
   | **Descarte** | Podredumbre, moho o daño severo: no apta para la venta | Fuera de norma |

   Las normas del Codex Alimentarius para frutas frescas (por ejemplo, CXS 299-2010 para manzanas y CXS 245-2004 para naranjas) también definen tres categorías comerciales, lo que respalda el uso de tres grados;
3. un **tamaño relativo** (pequeño, mediano o grande) a partir del diámetro normalizado.

**Criterios de éxito** (definidos antes de modelar):

| Criterio | Meta | Por qué |
|---|---|---|
| F1-macro (12 clases fruta × calidad) en prueba | ≥ 0,85 | Rendimiento equilibrado en todas las clases |
| Precisión del grado Primera | ≥ 95 % | Máximo 5 % de frutas malas vendidas como Primera (riesgo para el consumidor) |
| Precisión del grado Descarte | ≥ 95 % | Máximo 5 % de frutas buenas desperdiciadas (pérdida para el productor) |
| Tiempo de inferencia en CPU | < 50 ms por imagen | Uso en tiempo real con cámara |
| Brecha entre la prueba de Kaggle y las fotos propias | Reportarla | Medir la generalización real |

## 1.3 Formulación matemática

Sea una imagen RGB $x \in \{0,\dots,255\}^{H \times W \times 3}$. Tras el preprocesamiento $\phi$ (recorte cuadrado central y redimensionamiento), $\tilde{x} = \phi(x) \in [0,1]^{128 \times 128 \times 3}$.

**Variable objetivo.** El dataset etiqueta cada imagen con una fruta $f \in \mathcal{F}$ ($|\mathcal{F}| = 6$) y una calidad $q \in \{\text{mala}, \text{buena}\}$. Se modela la distribución conjunta sobre las $K = 12$ clases $y = (f, q)$:

$$
h_\theta(\tilde{x}) = \big(P_\theta(y = k \mid \tilde{x})\big)_{k=1}^{12} \in \Delta^{11}, \qquad
P_\theta(y = k \mid \tilde{x}) = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
$$

donde $z = g_\theta(\tilde{x})$ son los *logits* del modelo y $\Delta^{11}$ es el simplex de probabilidad.

**Aprendizaje.** Se minimiza la entropía cruzada ponderada (riesgo empírico) sobre el conjunto de entrenamiento $\mathcal{D} = \{(\tilde{x}_i, y_i)\}_{i=1}^{N}$:

$$
\hat{\theta} = \arg\min_\theta \; -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \log P_\theta(y_i \mid \tilde{x}_i) + \lambda \lVert \theta \rVert_2^2, \qquad w_k = \frac{N}{K\,N_k}
$$

Los pesos $w_k$ compensan el desbalance de clases y $\lambda$ regulariza.

**Salidas derivadas por marginalización:**

$$
p(\tilde{x}) = P(q = \text{buena} \mid \tilde{x}) = \sum_{f \in \mathcal{F}} P_\theta\big((f, \text{buena}) \mid \tilde{x}\big), \qquad
\hat{f} = \arg\max_f \sum_{q} P_\theta\big((f, q) \mid \tilde{x}\big)
$$

**Grado con opción de rechazo** (Chow, 1970):

$$
\text{grado}(\tilde{x}) =
\begin{cases}
\text{Primera} & p(\tilde{x}) \ge \tau_{\text{alto}} \\
\text{Descarte} & p(\tilde{x}) \le \tau_{\text{bajo}} \\
\text{Segunda} & \text{en otro caso}
\end{cases}
\quad
\begin{aligned}
\tau_{\text{alto}} &= \min\{\tau : P(\text{buena} \mid p \ge \tau) \ge \pi\}\\
\tau_{\text{bajo}} &= \max\{\tau : P(\text{mala} \mid p \le \tau) \ge \pi\}
\end{aligned}
$$

Los umbrales se estiman en el conjunto de validación con $\pi = 0{,}98$, un margen de 3 puntos sobre el criterio de éxito (≥ 95 % en prueba), porque la precisión medida justo en el umbral baja al pasar a datos nuevos (ver fase 4).

**Tamaño.** Con la máscara de segmentación $M$ (fondo por umbral de Otsu sobre la distancia de color en CIELab), el área es $A = \sum_{u,v} M(u,v)$, y el diámetro equivalente normalizado es

$$
d_{\text{norm}} = \frac{2\sqrt{A/\pi}}{\min(H, W)} \in (0, 1],
$$

que se discretiza en pequeño, mediano o grande con umbrales por fruta.

**Pseudocódigo del sistema:**

```
entrada: imagen x
x̃ ← recorte_central_y_redimensionar(x, 128)
P ← modelo(x̃)                          # 12 probabilidades
p_buena ← Σ_f P[f, buena]
fruta ← argmax_f (P[f, buena] + P[f, mala])
grado ← Primera si p_buena ≥ τ_alto; Descarte si p_buena ≤ τ_bajo; si no, Segunda
M ← segmentar(x̃);  d ← 2·sqrt(|M|/π)/128
tamaño ← discretizar(d, fruta)
salida: (fruta, grado, p_buena, tamaño)
```

## 1.4 ¿Por qué es un problema complejo de ingeniería?

1. **Alta dimensionalidad:** cada entrada tiene 128·128·3 = 49.152 variables, mientras que solo hay unas 8.000 imágenes de entrenamiento ($d \gg N$ en el espacio de píxeles), así que el riesgo de sobreajuste es alto.
2. **Variabilidad intra-clase alta y separación inter-clase baja:** una mancha pequeña separa una fruta buena de una mala, mientras que dos frutas buenas pueden verse muy distintas por la iluminación, el ángulo, la madurez o la variedad.
3. **Correlaciones espurias:** en el dataset, el fondo de la foto predice la calidad (ver fase 2). Un modelo puede obtener métricas altas con un atajo que no funcionaría en el despliegue.
4. **Datos con dependencias:** copias aumentadas y ráfagas de la misma fruta violan el supuesto i.i.d. Una partición ingenua sobreestima el desempeño.
5. **Requisitos en conflicto:** precisión frente a cobertura (cuántas frutas reciben veredicto firme), exactitud frente a costo computacional (CPU, tiempo real), y costos asimétricos (vender una fruta podrida no cuesta lo mismo que descartar una buena).
6. **Restricciones éticas y sociales:** el sistema afecta los ingresos de pequeños productores y la seguridad del consumidor (sección 1.5).

## 1.5 Análisis ético (PI1)

Se toman como referencia el **Código de Ética y Conducta Profesional de la ACM** (2018), el **Código de Ética del IEEE** (2020) y el **Código de Ética para el ejercicio de la ingeniería en Colombia** (Ley 842 de 2003, Título IV).

| # | Situación o decisión del proyecto | Dilema o responsabilidad | Principio del código de ética | Medida adoptada |
|---|---|---|---|---|
| E1 | Toma de fotos propias en plazas de mercado | Privacidad de vendedores y compradores: rostros, puestos y ubicación | ACM 1.6 (respetar la privacidad); IEEE I.1; Ley 1581 de 2012 (datos personales) | Protocolo de captura sin personas ni nombres de negocios; `src/data/strip_exif.py` borra el GPS y la fecha de las fotos antes de versionarlas |
| E2 | Fotografiar la mercancía de terceros | **Consentimiento informado** del vendedor | ACM 1.3 (ser honesto y confiable); Ley 842, deberes con la sociedad | Se pide permiso verbal, se explica el fin académico y se registra el lugar como "mercado" sin identificar el puesto |
| E3 | Dataset de Kaggle con sesgo de fondo y frutas de India | **Sesgo**: el modelo puede fallar con variedades y condiciones colombianas | ACM 1.4 (ser justo y no discriminar); ACM 2.5 (evaluar riesgos de forma integral) | Se midió el sesgo (el borde predice la calidad con 85 %), se reporta como limitación, se usa Grad-CAM y se evalúa con fotos locales |
| E4 | Reportar métricas | **Honestidad**: una partición ingenua infla los resultados | ACM 1.3; ACM 2.5; IEEE I.5 (afirmaciones honestas basadas en datos) | Deduplicación de copias, partición agrupada por ráfaga y conjunto de prueba usado una sola vez |
| E5 | Fruta podrida clasificada como Primera | **Seguridad** del consumidor (mohos y micotoxinas) | ACM 1.2 (evitar daño); IEEE I.1 (seguridad y salud pública) | Umbral τ_alto que garantiza una precisión ≥ 95 % en Primera; los casos dudosos van a Segunda |
| E6 | Fruta buena clasificada como Descarte | Pérdida económica injusta para el productor | ACM 1.4; ACM 1.1 (contribuir al bienestar humano) | Umbral τ_bajo con precisión ≥ 95 %; categoría Segunda con revisión humana |
| E7 | Automatizar un trabajo manual | Posible **desplazamiento laboral** de clasificadores | ACM 1.1; ACM 3.7 (sistemas que se integran a la sociedad) | Se diseña como **herramienta de apoyo** con humano en el ciclo (la Segunda se revisa a mano), no como reemplazo |
| E8 | Uso de datos y código de terceros | Atribución y propiedad intelectual | ACM 1.5 (respetar el trabajo ajeno); Ley 842 | Dataset con licencia CC0 citado (FruitNet y Kaggle); librerías referenciadas; licencia GPL del repositorio |
| E9 | Confianza excesiva en la herramienta | Uso fuera del dominio (otras frutas, fotos con varias frutas) | ACM 2.5; ACM 3.1 (el bien público en el centro) | La app advierte cuando la confianza es baja o la segmentación no es confiable, e informa las frutas soportadas |

## 1.6 Análisis de impactos (PI2)

| Dimensión | Impactos positivos | Riesgos | Mitigación |
|---|---|---|---|
| **Social** | Clasificación más consistente y transparente para vendedores y compradores; menos fruta en mal estado llega al consumidor | Percepción de **vigilancia** si se instalan cámaras en un puesto; posible desplazamiento laboral; desconfianza si el sistema se equivoca | La cámara solo apunta a la fruta, no se guardan imágenes de personas, procesamiento local sin enviar fotos a la nube, humano en el ciclo para la Segunda |
| **Económica** | Estandariza precios por grado; reduce pérdidas al dar salida a la Segunda (procesamiento) en vez de descartar lotes; costo bajo (celular o webcam y CPU, sin GPU) | Clasificaciones erradas que cuestan dinero (falsos Descarte) o reputación (falsos Primera); dependencia de una herramienta no certificada | Umbrales con precisión ≥ 95 % en los extremos; cobertura reportada; se usa como apoyo y no como certificación oficial |
| **Ambiental** | Menos **desperdicio de alimentos** (en Colombia, 34 % de la oferta [DNP, 2016]); modelo pequeño (113 mil parámetros) con un consumo energético mínimo, que corre en CPU o en una Raspberry Pi | Energía del entrenamiento y del uso continuo de la cámara; residuos electrónicos si se compra hardware dedicado | CNN pequeña entrenada en CPU en minutos (sin GPU); inferencia en milisegundos; se reutilizan celulares y computadores existentes |
| **Global** | Método replicable en otros países y para otras frutas; datos públicos CC0 | Variedades y estándares de calidad distintos (Codex, normas NTC, normas de la UE); regulación de datos distinta (Ley 1581 en Colombia, GDPR en Europa); el dataset es de India | Reentrenar o recalibrar con datos locales (fotos propias); grados alineados con las categorías Codex; procesamiento local que facilita cumplir las leyes de datos |

## Referencias de esta fase

- Departamento Nacional de Planeación (DNP), *Pérdida y desperdicio de alimentos en Colombia*, 2016.
- FAO, *The State of Food and Agriculture 2019: Moving forward on food loss and waste reduction*, Roma, 2019.
- C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Inf. Theory*, vol. 16, no. 1, pp. 41–46, 1970.
- ACM, *ACM Code of Ethics and Professional Conduct*, 2018.
- IEEE, *IEEE Code of Ethics*, 2020.
- Congreso de Colombia, Ley 842 de 2003; Ley 1581 de 2012.
- Codex Alimentarius, *Standard for Apples* (CXS 299-2010); *Standard for Oranges* (CXS 245-2004).
