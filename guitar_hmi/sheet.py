"""Engrave MusicXML (standard notation over tab) as a PDF with Verovio.

    python -m guitar_hmi.sheet tabs\\tab_20261116_221500.musicxml     writes the .pdf beside it
"""
import re
import sys
from pathlib import Path

# Letter paper, in Verovio's units (tenths of a millimetre).
PAGE_WIDTH, PAGE_HEIGHT = 2159, 2794
MARGIN = 120
SCALE = 45


def render_pdf(musicxml, pdf_path, title=None):
    """Write the score as a PDF, one page per Verovio page; returns the number of pages."""
    import pymupdf
    import verovio

    toolkit = verovio.toolkit()
    toolkit.setOptions({
        "pageWidth": PAGE_WIDTH, "pageHeight": PAGE_HEIGHT, "scale": SCALE,
        "pageMarginLeft": MARGIN, "pageMarginRight": MARGIN,
        "pageMarginTop": MARGIN, "pageMarginBottom": MARGIN,
        "adjustPageHeight": False, "footer": "none", "breaks": "auto",
    })
    if title:
        musicxml = musicxml.replace("<work-title>Guitar</work-title>", f"<work-title>{title}</work-title>", 1)
    # The metronome's note symbol comes from a font MuPDF can't load; words print anywhere.
    musicxml = re.sub(r"<metronome>.*?<per-minute>(\d+)</per-minute></metronome>",
                      r"<words>about \1 BPM</words>", musicxml)
    if not toolkit.loadData(musicxml):
        raise ValueError("Verovio could not read the MusicXML")
    pdf = pymupdf.open()
    for page in range(1, toolkit.getPageCount() + 1):
        svg = pymupdf.open(stream=_for_mupdf(toolkit.renderToSVG(page)).encode("utf-8"), filetype="svg")
        pdf.insert_pdf(pymupdf.open("pdf", svg.convert_to_pdf()))
    pdf.save(pdf_path)
    return pdf.page_count


def _for_mupdf(svg):
    """MuPDF ignores Verovio's CSS (strokes, hidden placeholders, the embedded music font) and
    draws its inner <svg> at zero size unless it is given the page's size."""
    width, height = re.search(r'<svg width="([^"]+)" height="([^"]+)"', svg).groups()
    svg = svg.replace('<svg class="definition-scale"',
                      f'<svg class="definition-scale" width="{width}" height="{height}"', 1)
    svg = re.sub(r'<g [^>]*class="[^"]*placeholder[^"]*"[^>]*>.*?</g>', "", svg, flags=re.S)
    return re.sub(r"<(path|rect|polygon|polyline|ellipse)\b(?![^>]*\bstroke=)", r'<\1 stroke="black"', svg)


def main():
    for name in sys.argv[1:]:
        path = Path(name)
        pages = render_pdf(path.read_text(encoding="utf-8"), path.with_suffix(".pdf"), path.stem)
        print(f"{path.with_suffix('.pdf')}: {pages} page(s)")


if __name__ == "__main__":
    main()
