"""
PDF Highlighter Module
Annotates matched keyword bounding boxes directly in a PDF using PyMuPDF.
OCR coordinates are in image pixels (at a given DPI); this module scales them
back to PDF point space before drawing the highlights.
"""

import os
from typing import List, Dict, Any
from pathlib import Path


# Default highlight color: bright yellow (R, G, B) in 0-1 range
_HIGHLIGHT_COLOR = (1.0, 0.95, 0.0)   # yellow
_STROKE_COLOR    = (0.0, 0.7, 0.0)    # green border


def highlight_page_by_coords(
    page: Any,
    matches: List[Dict[str, Any]],
    scale: float,
) -> int:
    """
    Apply coordinate-based highlights to a specific PyMuPDF page.
    Used for scanned/OCR matches.
    """
    import fitz
    count = 0
    for m in matches:
        try:
            x1 = float(m["x1"]) * scale
            y1 = float(m["y1"]) * scale
            x2 = float(m["x2"]) * scale
            y2 = float(m["y2"]) * scale

            rect = fitz.Rect(x1, y1, x2, y2)
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=_STROKE_COLOR, fill=_HIGHLIGHT_COLOR)
            annot.set_border(width=1)
            annot.set_opacity(0.4)
            annot.set_info(
                title="Keyword Match (OCR)",
                content=f"Keyword: {m.get('keyword', '')}\nText: {m.get('text', '')}",
            )
            annot.update()
            count += 1
        except Exception as e:
            print(f"    [WARN] Could not annotate word at {m.get('x1')},{m.get('y1')}: {e}")
    return count


def highlight_page_by_search(
    page: Any,
    keywords: List[str],
) -> int:
    """
    Apply search-based highlights to a specific PyMuPDF page.
    Used for native text matches.
    """
    import fitz
    count = 0
    for keyword in keywords:
        instances = page.search_for(keyword, quads=True, flags=fitz.TEXT_INHIBIT_SPACES)
        if not instances:
            instances = page.search_for(keyword.lower(), quads=True, flags=fitz.TEXT_INHIBIT_SPACES)
        if not instances:
            instances = page.search_for(keyword.upper(), quads=True, flags=fitz.TEXT_INHIBIT_SPACES)
        
        for quad in instances:
            annot = page.add_highlight_annot(quad)
            annot.set_colors(stroke=_STROKE_COLOR)
            annot.set_opacity(0.5)
            annot.set_info(title="Keyword Match (Native)", content=f"Keyword: {keyword}")
            annot.update()
            count += 1
    return count


def highlight_pdf(
    pdf_path: str,
    matches: List[Dict[str, Any]],
    output_path: str,
    dpi: int = 300,
) -> str:
    """Draw highlight annotations on a PDF for every keyword match (Legacy/Manual)."""
    import fitz
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    scale = 72.0 / dpi
    doc = fitz.open(pdf_path)
    
    page_matches = {}
    for m in matches:
        p = int(m.get("page", 1)) - 1
        page_matches.setdefault(p, []).append(m)

    total = 0
    for p_idx, p_matches in page_matches.items():
        if 0 <= p_idx < doc.page_count:
            total += highlight_page_by_coords(doc[p_idx], p_matches, scale)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[OK] PDF highlighted: {total} annotations -> {output_path}")
    return os.path.abspath(output_path)


def highlight_pdf_by_search(
    pdf_path: str,
    keywords: List[str],
    output_path: str,
) -> str:
    """Alternative: use PyMuPDF's search to highlight native PDF text (Legacy/Manual)."""
    import fitz
    doc = fitz.open(pdf_path)
    total = 0
    for page in doc:
        total += highlight_page_by_search(page, keywords)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[OK] Native-text PDF highlighted: {total} annotations -> {output_path}")
    return os.path.abspath(output_path)
