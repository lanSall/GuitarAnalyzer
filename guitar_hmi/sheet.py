"""Engrave MusicXML (standard notation over tab) with Verovio: a PDF for printing, or a PNG of a
few bars on one line for the monitor's live sheet.

    python -m guitar_hmi.sheet tabs\\tab_20261116_221500.musicxml     writes the .pdf beside it
"""
import re
import sys
from pathlib import Path

# Letter paper, in Verovio's units (tenths of a millimetre).
PAGE_WIDTH, PAGE_HEIGHT = 2159, 2794
MARGIN = 120
SCALE = 45
LIVE_SCALE = 40
LIVE_MARGIN = 15
LIVE_STAFF_SPACING = 6


def render_pdf(musicxml, pdf_path, title=None):
    """Write the score as a PDF, one page per Verovio page; returns the number of pages."""
    import pymupdf

    toolkit = _load(musicxml, title, {
        "pageWidth": PAGE_WIDTH, "pageHeight": PAGE_HEIGHT, "scale": SCALE,
        "pageMarginLeft": MARGIN, "pageMarginRight": MARGIN,
        "pageMarginTop": MARGIN, "pageMarginBottom": MARGIN,
        "adjustPageHeight": False, "footer": "none", "breaks": "auto",
    })
    pdf = pymupdf.open()
    for page in range(1, toolkit.getPageCount() + 1):
        svg = pymupdf.open(stream=_for_mupdf(toolkit.renderToSVG(page)).encode("utf-8"), filetype="svg")
        pdf.insert_pdf(pymupdf.open("pdf", svg.convert_to_pdf()))
    pdf.save(pdf_path)
    return pdf.page_count


def render_png(musicxml, zoom=1.0, dark=False):
    """PNG bytes of the whole score on one line, without title, cropped to the music.
    dark=True inverts it onto the monitor's background (#111827) with light ink."""
    import pymupdf

    toolkit = _load(musicxml, None, {
        "scale": LIVE_SCALE, "breaks": "none", "adjustPageWidth": True, "adjustPageHeight": True,
        "pageMarginLeft": LIVE_MARGIN, "pageMarginRight": LIVE_MARGIN,
        "pageMarginTop": LIVE_MARGIN, "pageMarginBottom": LIVE_MARGIN,
        "header": "none", "footer": "none", "spacingStaff": LIVE_STAFF_SPACING,
    })
    page = pymupdf.open(stream=_for_mupdf(toolkit.renderToSVG(1)).encode("utf-8"), filetype="svg")[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    if dark:
        _darken(pix)
    return pix.tobytes("png")


def render_live_png(musicxml):
    """Same engraving as the PDF, restyled for the monitor's dark panel. Called from a worker."""
    return render_png(musicxml.replace("<part-name>Guitar</part-name>", "<part-name></part-name>"),
                      zoom=1.6, dark=True)


def _darken(pix):
    """Black-on-white score -> light ink on #111827, matching the tab panel."""
    import numpy as np

    pix.invert_irect(pix.irect)
    ink = np.ndarray((pix.h, pix.w, pix.n), dtype=np.uint8, buffer=pix.samples_mv)[:, :, :3]
    paper = ink.max(axis=2) < 18
    ink[paper] = (0x11, 0x18, 0x27)


def _load(musicxml, title, options):
    import verovio

    toolkit = verovio.toolkit()
    toolkit.setOptions(options)
    if title:
        musicxml = musicxml.replace("<work-title>Guitar</work-title>", f"<work-title>{title}</work-title>", 1)
    # The metronome's note symbol comes from a font MuPDF can't load; words print anywhere.
    musicxml = re.sub(r"<metronome>.*?<per-minute>(\d+)</per-minute></metronome>",
                      r"<words>about \1 BPM</words>", musicxml)
    if not toolkit.loadData(musicxml):
        raise ValueError("Verovio could not read the MusicXML")
    return toolkit


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
