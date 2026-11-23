# Informe final

Informe en formato IEEE (máximo 7 páginas; actualmente 5), basado en la plantilla `docs/enunciado/plantilla_informe_IEEE.docx`.

| Archivo | Contenido |
|---|---|
| `informe.md` | Texto fuente del informe: se edita aquí |
| `build_informe.py` | Convierte `informe.md` en el `.docx` usando los estilos de la plantilla, y el `.docx` en PDF |
| `informe_final.docx`, `informe_final.pdf` | Documentos generados (entregables) |
| `figures/` | Figuras en SVG y PNG |

Para regenerarlo (necesita LibreOffice):

```bash
python reports/build_informe.py
```

Las figuras con versión `.svg` se incrustan en formato vectorial, con el PNG como respaldo. Las figuras anchas son marcos flotantes en la parte superior de la página: si dos caen en la misma página se tapan, así que después de cambiar el texto hay que revisar el PDF y, si hace falta, mover la figura en `informe.md`.
