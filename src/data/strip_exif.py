"""Elimina los metadatos EXIF de las fotos propias antes de versionarlas.

Las fotos de celular guardan en EXIF la ubicación GPS, la fecha y hora, y el
modelo del teléfono. Publicarlas en un repositorio revelaría dónde y cuándo
estuvo cada integrante, o la ubicación de los puestos de venta fotografiados.
Este script reescribe cada imagen de las carpetas de fotos propias sin
metadatos, aplicando antes la rotación EXIF para que la foto no quede girada:

- ``data/own/images/``        manzanas anotadas en ``labels.csv``
- ``data/own/otras_frutas/``  frutas que el modelo no conoce (``otras_frutas.csv``)

Uso:
    python -m src.data.strip_exif
"""

from __future__ import annotations

from PIL import Image, ImageOps

from src.utils.paths import OWN

EXTS = {".jpg", ".jpeg", ".png"}
FOLDERS = ("images", "otras_frutas")


def strip(path) -> bool:
    img = Image.open(path)
    had_exif = bool(img.getexif())
    if not had_exif:
        return False  # ya está limpia: no se recomprime
    clean = ImageOps.exif_transpose(img).convert("RGB")
    clean.save(path, quality=95)  # se guarda sin el parámetro exif → sin metadatos
    return True


def main() -> None:
    files = [p for d in FOLDERS if (OWN / d).is_dir()
             for p in sorted((OWN / d).iterdir()) if p.suffix.lower() in EXTS]
    n = sum(strip(p) for p in files)
    print(f"{len(files)} imágenes revisadas; {n} tenían metadatos EXIF y fueron limpiadas.")


if __name__ == "__main__":
    main()
