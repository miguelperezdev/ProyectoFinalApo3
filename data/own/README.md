# Fotos propias del grupo

El enunciado pide de **30 a 50 imágenes propias** de frutas reales, tomadas en plazas de mercado, supermercados o en casa.

## Cómo tomar las fotos
- Una sola fruta por foto, sobre un **fondo liso y uniforme** (cartulina blanca o gris).
- Luz natural o lo más pareja posible, sin sombras fuertes.
- Varias vistas: frente, arriba, abajo y girada.
- Incluir variedad: maduras y verdes, grandes y pequeñas, con golpes, manchas o podredumbre.
- Preferir las mismas frutas del dataset: manzana, banano, guayaba, limón, naranja y granadilla/granada.
- Para estimar el tamaño, incluir de vez en cuando una moneda o una regla como referencia.
- No deben aparecer rostros, personas ni nombres de negocios (privacidad).

## Organización
Guardar las imágenes en `data/own/images/` con nombres como `manzana_001.jpg`, y anotar cada una en `labels.csv`:

| columna | valores |
|---|---|
| `archivo` | nombre del archivo |
| `fruta` | apple, banana, guava, lime, orange, pomegranate |
| `calidad` | por definir (ver README principal) |
| `tamano` | pequeno, mediano, grande |
| `lugar` | casa, mercado, supermercado |
| `autor` | miguel, danna, mateo |
