"""
Keyword Classifier Module
Loads keywords from a text file and matches them against OCR-extracted text.
"""

import os
import re
from typing import List, Dict, Any


def load_keywords(keywords_path: str) -> List[str]:
    """
    Load keywords from a text file (one keyword per line).
    Lines starting with '#' are treated as comments and ignored.

    Args:
        keywords_path: Path to the keywords text file

    Returns:
        List of keyword strings (stripped, non-empty)
    """
    if not os.path.exists(keywords_path):
        raise FileNotFoundError(f"Keywords file not found: {keywords_path}")

    keywords = []
    with open(keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.append(line)

    print(f"[OK] Loaded {len(keywords)} keywords from: {keywords_path}")
    return keywords


def classify_ocr_results(
    ocr_results: List[Dict[str, Any]],
    keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Match OCR results against a list of keywords and return matches with
    WORD-LEVEL bounding boxes (not the full line box).

    PaddleOCR returns one bounding box per text line. When a keyword appears
    inside a longer line (e.g. "OCR-based extracting text"), we narrow the
    box to cover only the matched word by estimating its pixel position using
    character-proportion scaling along the line width.

    Returns:
        List of match dicts -- one per keyword occurrence, with precise x1/x2.
    """
    matches = []

    for result in ocr_results:
        text     = result.get("text", "")
        x1_line  = result.get("x1", 0)
        y1_line  = result.get("y1", 0)
        x2_line  = result.get("x2", 0)
        y2_line  = result.get("y2", 0)
        line_width = x2_line - x1_line
        total_chars = max(len(text), 1)

        for keyword in keywords:
            pattern = re.escape(keyword)

            # Find every occurrence of the keyword in the line (case-insensitive)
            for m in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = m.group()     # actual matched string
                char_start   = m.start()     # char index start in line
                char_end     = m.end()       # char index end (exclusive)

                # Estimate pixel x-positions by character proportion
                word_x1 = x1_line + int((char_start / total_chars) * line_width)
                word_x2 = x1_line + int((char_end   / total_chars) * line_width)

                # Clamp to line bounds
                word_x1 = max(word_x1, x1_line)
                word_x2 = min(word_x2, x2_line)

                match = result.copy()
                match["keyword"]        = keyword
                match["classification"] = _get_classification(keyword)
                match["text"]           = matched_text   # just the matched word
                match["full_line_text"] = text           # original full line
                match["x1"]            = word_x1
                match["x2"]            = word_x2
                # y1/y2 stay the same (full line height)
                matches.append(match)

    return matches


def _get_classification(keyword: str) -> str:
    """
    Derive a high-level classification label from a keyword.
    Can be extended to map keyword → category via a config file.

    Currently returns a normalized title-case version of the keyword,
    or the keyword itself if it is already a category name.
    """
    # Simple rule: classification = keyword (can be replaced with a mapping dict)
    return keyword.title()


def group_matches_by_document(
    matches: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Group a flat list of matches into:
        { document_name: { keyword: { pages: [...], matches: [...] } } }

    Useful for building the summary Excel sheet.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for match in matches:
        doc = match.get("document", "unknown")
        kw = match.get("keyword", "unknown")
        page = match.get("page", 0)

        if doc not in grouped:
            grouped[doc] = {}
        if kw not in grouped[doc]:
            grouped[doc][kw] = {"pages": [], "matches": []}

        if page not in grouped[doc][kw]["pages"]:
            grouped[doc][kw]["pages"].append(page)

        grouped[doc][kw]["matches"].append(match)

    return grouped
