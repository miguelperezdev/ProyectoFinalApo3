r"""Genera el informe IEEE (.docx y .pdf) a partir de ``reports/informe.md``.

El Markdown es la fuente única del informe; este script lo vuelca en la
plantilla oficial del curso (``docs/enunciado/plantilla_informe_IEEE.docx``)
usando sus estilos: *paper title*, *Author*, *Abstract*, *Keywords*,
*Heading 1/2*, *Body Text*, *bullet list*, *figure caption*, *table head*,
*equation* y *references*. Así el .md (legible en GitHub) y el .docx (entrega)
nunca se desincronizan.

Subconjunto de Markdown admitido:
    # Título                       → paper title
    Resumen— texto                 → Abstract
    Palabras clave— texto          → Keywords
    ## Sección / ### Subsección    → Heading 1 / Heading 2 (numeración de la plantilla)
    párrafo con **negrita**, *cursiva*, `código` y $fórmula$ (LaTeX)
    - viñeta                       → bullet list
    ![pie de figura](figures/x.png){ancho=3.3}
    Tabla: título  + tabla con barras verticales
    $$ ecuación en LaTeX \qquad (n) $$ → equation (puede ocupar varias líneas)

Las fórmulas se convierten en ecuaciones nativas de Word con las versiones OMML de
``reports/ecuaciones.py``; si se cambia una fórmula del .md hay que actualizarla allí.
    [n] referencia (tras «## Referencias»)

Requiere LibreOffice (``soffice``) para normalizar la plantilla y exportar el PDF.

Uso:
    python reports/build_informe.py
"""

from __future__ import annotations

import re
from copy import deepcopy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecuaciones import display_omml, inline_omml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "enunciado" / "plantilla_informe_IEEE.docx"
SOURCE = ROOT / "reports" / "informe.md"
OUT_DOCX = ROOT / "reports" / "informe_final.docx"

AUTHORS = [
    ("Miguel Pérez Ojeda", "Código A00407054"),
    ("Danna Isabel Zambrano Sánchez", "Código A00404131"),
    ("Mateo Berrío Villa", "Código A00399867"),
    ("Sebastián Romero León", "Código A00404670"),
    ("Milton Sarria Paja", "Profesor"),
]
AUTHORS_PER_ROW = 3
AFFILIATION = ("Facultad de Ingeniería, Diseño y Ciencias Aplicadas", "Universidad Icesi", "Cali, Colombia")

COLUMN_WIDTH = 3.45  # pulgadas: ancho de una columna del cuerpo a dos columnas
INLINE = re.compile(r"(\$[^$]+\$|\*\*.+?\*\*|\*.+?\*|`.+?`)")


def normalized_template(tmp: Path) -> Path:
    """La plantilla original no abre en python-docx; LibreOffice la reescribe en OOXML estándar."""
    subprocess.run(["soffice", "--headless", "--convert-to", "docx:MS Word 2007 XML", "--outdir", str(tmp),
                    str(TEMPLATE)], check=True, capture_output=True)
    return tmp / TEMPLATE.name


def add_runs(par, text: str, base_italic: bool = False) -> None:
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("$"):  # fórmula en línea → ecuación de Word
            par._p.append(parse_xml(inline_omml(tok[1:-1])))
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Courier New"; r.font.size = Pt(8.5)
        elif tok.startswith("*"):
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if base_italic:
            r.italic = True


class Builder:
    def __init__(self, doc: docx.Document, anchor):
        self.doc = doc
        self.anchor = anchor  # el nuevo contenido se inserta antes de este elemento

    def _new_par(self, style: str):
        p = self.doc.add_paragraph(style=style)
        self.anchor.addprevious(p._p)
        return p

    def heading(self, text: str, level: int):
        p = self._new_par(f"Heading {level}")
        if level in (1, 2):
            # Numeración manual («III.», «A.»): con la automática, LibreOffice dibuja el número
            # encima de una figura ancha cuando el título queda justo debajo de ella.
            num_pr = p._p.get_or_add_pPr().get_or_add_numPr()
            num_pr.get_or_add_ilvl().val = 0
            num_pr.get_or_add_numId().val = 0
            if level == 1:
                self.n_sec = getattr(self, "n_sec", 0) + 1
                self.n_sub = 0
                roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][self.n_sec - 1]
                p.add_run(f"{roman}. ")
            else:
                self.n_sub = getattr(self, "n_sub", 0) + 1
                p.add_run(f"{chr(64 + self.n_sub)}. ")
        add_runs(p, text)

    def body(self, text: str):
        add_runs(self._new_par("Body Text"), text)

    def bullet(self, text: str):
        add_runs(self._new_par("bullet list"), text)

    def equation(self, latex: str, number: int):
        """Ecuación de Word centrada, con su número (n) alineado a la derecha."""
        p = self._new_par("equation")
        p._p.append(parse_xml(display_omml(latex, number)))

    def reference(self, text: str):
        add_runs(self._new_par("references"), text)

    def _attach_svg(self, shape, svg: Path):
        """Incrusta la versión vectorial (SVG) de la figura; el PNG queda como respaldo.

        Es el mecanismo de Word 2016+ (extensión ``asvg:svgBlip``), que LibreOffice también
        lee: así el informe y su PDF conservan los gráficos en calidad vectorial.
        """
        self.n_svg = getattr(self, "n_svg", 0) + 1
        part = Part(PackURI(f"/word/media/figura{self.n_svg}.svg"), "image/svg+xml", svg.read_bytes(),
                    self.doc.part.package)
        rid = self.doc.part.relate_to(part, RT.IMAGE)
        blip = shape._inline.xpath(".//a:blip")[0]
        ext_lst = parse_xml(
            '<a:extLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}">'
            '<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" '
            f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="{rid}"/>'
            "</a:ext></a:extLst>")
        blip.append(ext_lst)

    def _frame(self, par, width_in: float):
        """Marco flotante arriba de la página y a lo ancho de las dos columnas."""
        fp = OxmlElement("w:framePr")
        for k, v in {"w:w": str(int(width_in * 1440)), "w:wrap": "notBeside", "w:vAnchor": "margin",
                     "w:hAnchor": "margin", "w:xAlign": "center", "w:yAlign": "top"}.items():
            fp.set(qn(k), v)
        ppr = par._p.get_or_add_pPr()
        ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, fp)

    def figure(self, path: Path, caption: str, width: float):
        wide = width > COLUMN_WIDTH + 0.1
        p = self._new_par("Normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.keep_with_next = True
        shape = p.add_run().add_picture(str(path), width=Inches(width))
        svg = path.with_suffix(".svg")
        if svg.exists():
            self._attach_svg(shape, svg)
        cap = self._new_par("figure caption")
        # Numeración manual: la automática de la plantilla cuenta aparte los marcos flotantes
        num_pr = cap._p.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = 0
        num_pr.get_or_add_numId().val = 0
        self.n_fig = getattr(self, "n_fig", 0) + 1
        cap.add_run(f"Fig. {self.n_fig}.  ")
        add_runs(cap, caption)
        if wide:  # figura a lo ancho de la página: marco flotante en la parte superior
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(10)
            self._frame(p, width)
            self._frame(cap, width)

    def table(self, title: str, rows: list[list[str]]):
        add_runs(self._new_par("table head"), title)
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        self.anchor.addprevious(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "000000")
            borders.append(el)
        t._tbl.tblPr.append(borders)
        # Cada columna recibe al menos el ancho de su palabra más larga (para no partir palabras) y el
        # resto del ancho de la columna IEEE se reparte en proporción a la longitud de su texto.
        ncol = len(rows[0])
        clean = [[re.sub(r"[*`]", "", c) for c in r] for r in rows]
        min_w = [max(len(w) for r in clean for w in r[j].split() or [""]) * 0.062 + 0.1 for j in range(ncol)]
        need = [sum(len(r[j]) for r in clean) / len(clean) + 2 for j in range(ncol)]
        spare = max(COLUMN_WIDTH - sum(min_w), 0)
        widths = [m + spare * n / sum(need) for m, n in zip(min_w, need)]
        widths = [w * COLUMN_WIDTH / sum(widths) for w in widths]
        t.autofit = False
        for j, w in enumerate(widths):
            t.columns[j].width = Inches(w)
            for cell in t.columns[j].cells:
                cell.width = Inches(w)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                par = cell.paragraphs[0]
                par.style = self.doc.styles["table col head" if i == 0 else "table copy"]
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                add_runs(par, val)
        spacer = self._new_par("Normal")
        spacer.paragraph_format.space_after = Pt(2)


def set_authors(par, authors) -> None:
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    for k, (name, role) in enumerate(authors):
        run = par.add_run(name)
        run.font.size = Pt(11)
        for line in (role, *AFFILIATION):
            run.add_break()
            run = par.add_run(line)
            run.font.size = Pt(9)
            run.italic = line == role
        if k < len(authors) - 1:
            run.add_break(WD_BREAK.COLUMN)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]


def build() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        doc = docx.Document(normalized_template(Path(tmp)))
    pars = doc.paragraphs
    body = doc.element.body

    # 1) Título y autores: como en la plantilla IEEE, hasta 3 por fila (fila 1 a 3 columnas, fila 2 con el resto)
    title_par, note_par, author_par = pars[0], pars[1], pars[3]
    row1, row2 = AUTHORS[:AUTHORS_PER_ROW], AUTHORS[AUTHORS_PER_ROW:]
    author_sect = pars[4]._p  # fin de la sección de autores
    author_sect.find(qn("w:pPr")).find(qn("w:sectPr")).find(qn("w:cols")).set(qn("w:num"), str(len(row1)))
    author_par2 = None
    if row2:
        p2, s2 = deepcopy(author_par._p), deepcopy(author_sect)
        author_sect.addnext(p2)
        p2.addnext(s2)
        s2.find(qn("w:pPr")).find(qn("w:sectPr")).find(qn("w:cols")).set(qn("w:num"), str(len(row2)))
        author_par2 = Paragraph(p2, author_par._parent)
    sect_body = next(p for p in pars[20:] if p._p.xpath("./w:pPr/w:sectPr"))  # fin de la sección a 2 columnas
    abstract_par, keywords_par = pars[20], pars[21]
    # Se eliminan los bloques de autores 5 y 6 y sus secciones (párrafos 5-19)
    for p in pars[5:20]:
        body.remove(p._p)
    body.remove(note_par._p)
    # Se elimina todo el texto de ejemplo de la plantilla entre las palabras clave y el fin de sección
    el = keywords_par._p.getnext()
    while el is not None and el is not sect_body._p:
        nxt = el.getnext()
        body.remove(el)
        el = nxt
    for p in list(doc.paragraphs):  # texto de advertencia de la plantilla tras la sección
        if p._p is not sect_body._p and "IEEE conference templates" in p.text:
            body.remove(p._p)
    # Párrafos vacíos tras el fin de la sección a 2 columnas: abrían una última página en blanco
    el = sect_body._p.getnext()
    while el is not None and el.tag == qn("w:p") and not "".join(el.itertext()).strip():
        nxt = el.getnext()
        body.remove(el)
        el = nxt

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    b = Builder(doc, sect_body._p)
    i, in_refs = 0, False
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln or ln.startswith("<!--"):
            i += 1
            continue
        if ln.startswith("# "):
            for r in list(title_par.runs):
                r._r.getparent().remove(r._r)
            title_par.add_run(ln[2:])
            set_authors(author_par, row1)
            if author_par2 is not None:
                set_authors(author_par2, row2)
        elif ln.startswith("Resumen—"):
            for r in list(abstract_par.runs):
                r._r.getparent().remove(r._r)
            head = abstract_par.add_run("Resumen—"); head.bold = True; head.italic = True
            tail = abstract_par.add_run(ln[len("Resumen—"):].strip()); tail.bold = True
        elif ln.startswith("Palabras clave—"):
            for r in list(keywords_par.runs):
                r._r.getparent().remove(r._r)
            head = keywords_par.add_run("Palabras clave—"); head.bold = True; head.italic = True
            tail = keywords_par.add_run(ln[len("Palabras clave—"):].strip()); tail.bold = True; tail.italic = True
        elif ln.startswith("## "):
            in_refs = ln[3:].strip().lower() == "referencias"
            if in_refs:
                b.heading("Referencias", 5)
            else:
                b.heading(ln[3:], 1)
        elif ln.startswith("### "):
            b.heading(ln[4:], 2)
        elif ln.startswith("![") :
            m = re.match(r"!\[(.+)\]\((.+?)\)(\{ancho=([\d.]+)\})?", ln)
            b.figure(ROOT / "reports" / m.group(2), m.group(1), float(m.group(4) or 3.3))
        elif ln.startswith("Tabla: "):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            k = j
            while k < len(lines) and lines[k].strip().startswith("|"):
                k += 1
            b.table(ln[len("Tabla: "):], parse_table(lines[j:k]))
            i = k
            continue
        elif ln.startswith("$$"):  # bloque $$ ... \qquad (n) $$, en una o varias líneas
            text, j = ln[2:], i
            while "$$" not in text:
                j += 1
                text += " " + lines[j].strip()
            latex = text[:text.index("$$")]
            # el número va al final como «\qquad (n)»: GitHub dibuja mal las ecuaciones con \tag{n}
            num = re.search(r"\\qquad\s*\((\d+)\)\s*$", latex.strip())
            b.equation(latex.strip()[:num.start()].strip(), int(num.group(1)))
            i = j
        elif in_refs and re.match(r"\[\d+\]", ln):
            b.reference(re.sub(r"^\[\d+\]\s*", "", ln))
        elif ln.startswith("- "):
            b.bullet(ln[2:])
        else:  # párrafo: une líneas consecutivas
            buf = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"(#|- |!\[|Tabla: |\$\$|\[\d+\]|<!--)", lines[i + 1]):
                i += 1
                buf.append(lines[i].strip())
            b.body(" ".join(buf))
        i += 1

    # Pie de página de la plantilla («XXX-X-XXXX… ©20XX IEEE»): no aplica a un informe de curso
    for sec in doc.sections:
        for footer in (sec.footer, sec.first_page_footer, sec.even_page_footer):
            for fp in footer.paragraphs:
                for r in list(fp.runs):
                    r._r.getparent().remove(r._r)
    # Rótulo automático de tablas de la plantilla («TABLE I.») en español
    for lvl in doc.part.numbering_part.element.iter(qn("w:lvlText")):
        if "TABLE" in lvl.get(qn("w:val"), ""):
            lvl.set(qn("w:val"), lvl.get(qn("w:val")).replace("TABLE", "TABLA"))
    # La sección a 2 columnas pasa a ser la última del documento: si termina en un párrafo propio,
    # LibreOffice abre después una página en blanco para la sección final vacía.
    final_sect = body.find(qn("w:sectPr"))
    cols = sect_body._p.find(qn("w:pPr")).find(qn("w:sectPr")).find(qn("w:cols"))
    old_cols = final_sect.find(qn("w:cols"))
    if old_cols is not None:
        final_sect.remove(old_cols)
    final_sect.find(qn("w:pgMar")).addnext(cols)
    body.remove(sect_body._p)
    doc.core_properties.title = title_par.text
    doc.core_properties.author = "; ".join(a for a, rol in AUTHORS if rol != "Profesor")
    doc.save(OUT_DOCX)
    return OUT_DOCX


def to_pdf(path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
                   check=True, capture_output=True)
    return path.with_suffix(".pdf")


def main() -> None:
    if not shutil.which("soffice"):
        sys.exit("Se necesita LibreOffice (soffice) para generar el informe.")
    out = build()
    pdf = to_pdf(out)
    pages = None
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(pdf).pages)
    except ImportError:
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        pages = int(m.group(1)) if m else None
    print(f"{out.relative_to(ROOT)} y {pdf.relative_to(ROOT)} generados · páginas: {pages}")


if __name__ == "__main__":
    main()
