"""Ecuaciones del informe en formato nativo de Word (OMML).

``informe.md`` escribe las fórmulas en LaTeX (``$...$`` en línea y bloques ``$$ ... \\qquad (n) $$``),
que se ven bien en GitHub y en la vista previa de Markdown. Word no entiende LaTeX, así que
``build_informe.py`` reemplaza cada fórmula por su versión OMML definida aquí.

Si se cambia o se agrega una fórmula en ``informe.md``, hay que actualizar también la entrada
correspondiente de ``DISPLAY`` o ``INLINE``: el generador se detiene si el LaTeX no coincide,
para que el .md y el .docx no se desincronicen.
"""

from __future__ import annotations

NS = ('xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')
_RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


def r(t: str, up: bool = False) -> str:
    """Texto matemático (cursiva); ``up=True`` para texto recto (palabras, funciones, números)."""
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if up else ""
    t = t.replace(", ", ", ").replace("arg max", "arg max").replace(" ", "")  # Word espacia los operadores
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<m:r>{sty}{_RPR}<m:t xml:space="preserve">{t}</m:t></m:r>'


def _e(*xs: str) -> str:
    return "<m:e>" + "".join(xs) + "</m:e>"


def sub(base: str, s: str) -> str:
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(base: str, s: str) -> str:
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>"


def subsup(base: str, b: str, t: str) -> str:
    return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{b}</m:sub><m:sup>{t}</m:sup></m:sSubSup>"


def frac(n: str, d: str, lin: bool = False) -> str:
    pr = '<m:fPr><m:type m:val="lin"/></m:fPr>' if lin else ""
    return f"<m:f>{pr}<m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def nary(lo: str, hi: str, body: str, chr_: str = "∑") -> str:
    pr = (f'<m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>'
          + ("" if hi else '<m:supHide m:val="1"/>') + "</m:naryPr>")
    return f"<m:nary>{pr}<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup>{_e(body)}</m:nary>"


def paren(body: str, beg: str = "(", end: str = ")") -> str:
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>{_e(body)}</m:d>'


def acc(base: str, ch: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="{ch}"/></m:accPr>{_e(base)}</m:acc>'


def rad(body: str) -> str:
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>{_e(body)}</m:rad>'


def func(name: str, arg: str) -> str:
    return f"<m:func><m:fName>{name}</m:fName>{_e(arg)}</m:func>"


def lim_low(name: str, lim: str) -> str:
    return f"<m:limLow><m:e>{r(name, up=True)}</m:e><m:lim>{lim}</m:lim></m:limLow>"


def eqarr(*rows: str) -> str:
    return '<m:eqArr><m:eqArrPr><m:baseJc m:val="center"/></m:eqArrPr>' + "".join(_e(x) for x in rows) + "</m:eqArr>"


def inline(body: str) -> str:
    """Ecuación dentro de un párrafo."""
    return f"<m:oMath {NS}>{body}</m:oMath>"


def numbered(body: str, n: int) -> str:
    """Ecuación centrada con su número (n) alineado a la derecha, como en la plantilla IEEE."""
    return (f'<m:oMathPara {NS}><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr><m:oMath>'
            f'<m:eqArr><m:eqArrPr><m:maxDist m:val="1"/></m:eqArrPr><m:e>{body}{r("#")}{paren(r(str(n), up=True))}'
            f"</m:e></m:eqArr></m:oMath></m:oMathPara>")


# ---------------------------------------------------------------- piezas comunes
XT = acc(r("x"), "̃")                        # x̃
XTI = acc(sub(r("x"), r("i")), "̃")          # x̃ᵢ
BAR = r("∣")
GE, LE = r("≥", up=True), r("≤", up=True)
TAU_A = sub(r("τ"), r("alto", up=True))
TAU_B = sub(r("τ"), r("bajo", up=True))

# ---------------------------------------------------------------- ecuaciones numeradas: n -> (LaTeX del .md, OMML)
DISPLAY = {
    1: (r"L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log P_\theta(y_i \mid \tilde{x}_i) + \lambda \lVert \theta \rVert_2^2",
        r("L") + paren(r("θ")) + r("=−") + frac(r("1", up=True), r("N"))
        + nary(r("i") + r("=1", up=True), r("N"),
               sub(r("w"), sub(r("y"), r("i"))) + r(" ", up=True)
               + func(r("log", up=True), sub(r("P"), r("θ")) + paren(sub(r("y"), r("i")) + BAR + XTI)))
        + r("+λ") + subsup(paren(r("θ"), "‖", "‖"), r("2", up=True), r("2", up=True))),
    2: (r"\begin{aligned} p(\tilde{x}) &= \sum_{f} P(f, \text{buena} \mid \tilde{x}) \\ "
        r"\hat{f} &= \arg\max_{f} \sum_{q} P(f, q \mid \tilde{x}) \end{aligned}",
        eqarr(r("p") + paren(XT) + r("=") + nary(r("f"), "", r("P") + paren(r("f") + r(", ", up=True) + r("buena", up=True) + BAR + XT)),
              acc(r("f"), "̂") + r("=")
              + func(lim_low("arg max", r("f")), nary(r("q"), "", r("P") + paren(r("f") + r(", ", up=True) + r("q") + BAR + XT))))),
    3: (r"\begin{aligned} \tau_{\text{alto}} &= \min \{ \tau : P(\text{buena} \mid p \ge \tau) \ge \pi \} \\ "
        r"\tau_{\text{bajo}} &= \max \{ \tau : P(\text{mala} \mid p \le \tau) \ge \pi \} \end{aligned}",
        eqarr(TAU_A + r("=") + func(r("min", up=True), paren(r("τ") + r(":", up=True) + r("P")
                                                             + paren(r("buena", up=True) + BAR + r("p") + GE + r("τ")) + GE + r("π"), "{", "}")),
              TAU_B + r("=") + func(r("max", up=True), paren(r("τ") + r(":", up=True) + r("P")
                                                             + paren(r("mala", up=True) + BAR + r("p") + LE + r("τ")) + GE + r("π"), "{", "}")))),
    4: (r"d_{\text{norm}} = \frac{2\sqrt{A/\pi}}{\min(H, W)} \in (0, 1]",
        sub(r("d"), r("norm", up=True)) + r("=")
        + frac(r("2", up=True) + rad(frac(r("A"), r("π"))), func(r("min", up=True), paren(r("H") + r(", ", up=True) + r("W"))))
        + r("∈", up=True) + paren(r("0, 1", up=True), "(", "]")),
}

# ---------------------------------------------------------------- fórmulas en línea: LaTeX del .md -> OMML
INLINE = {
    r"w_k = N/(K N_k)": sub(r("w"), r("k")) + r("=") + frac(r("N"), paren(r("K") + sub(r("N"), r("k"))), lin=True),
    r"p \ge \tau_{\text{alto}}": r("p") + GE + TAU_A,
    r"p \le \tau_{\text{bajo}}": r("p") + LE + TAU_B,
    r"K(x, x') = \exp(-\gamma \lVert x - x' \rVert^2)":
        r("K") + paren(r("x") + r(", ", up=True) + r("x′")) + r("=")
        + func(r("exp", up=True), paren(r("−γ") + sup(paren(r("x") + r("−") + r("x′"), "‖", "‖"), r("2", up=True)))),
}


def _norm(s: str) -> str:
    return " ".join(s.split())


def display_omml(latex: str, n: int) -> str:
    want, body = DISPLAY[n]
    if _norm(latex) != _norm(want):
        raise SystemExit(f"La ecuación ({n}) de informe.md cambió: actualiza DISPLAY[{n}] en reports/ecuaciones.py\n"
                         f"  .md:          {_norm(latex)}\n  ecuaciones.py: {_norm(want)}")
    return numbered(body, n)


def inline_omml(latex: str) -> str:
    key = _norm(latex)
    for k, body in INLINE.items():
        if _norm(k) == key:
            return inline(body)
    raise SystemExit(f"Fórmula en línea sin versión para Word: ${latex}$ (agrégala a INLINE en reports/ecuaciones.py)")
