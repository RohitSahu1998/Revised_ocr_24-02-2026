"""
classify_and_highlight.py
=========================
Main entry point for the Keyword Classification & Highlighting pipeline.

Usage:
    python classify_and_highlight.py --input <pdf_or_image_or_folder> [options]

Examples:
    # Process a single PDF with default keywords.txt
    python classify_and_highlight.py --input "report.pdf"

    # Process all PDFs in current directory
    python classify_and_highlight.py --input .

    # Use a custom keyword file and output folder
    python classify_and_highlight.py --input "scan.pdf" --keywords my_keywords.txt --output results/

    # Process a standalone image
    python classify_and_highlight.py --input "page.png"

    # Skip PDF highlighting (Excel only)
    python classify_and_highlight.py --input "report.pdf" --no-highlight
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Suppress noisy PaddleOCR logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# Tesseract Configuration
# ──────────────────────────────────────────────────────────────────────────────
import pytesseract
from PIL import Image, ImageOps
# Update this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ──────────────────────────────────────────────────────────────────────────────
# Supported extensions
# ──────────────────────────────────────────────────────────────────────────────
PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# OCR helper
# ──────────────────────────────────────────────────────────────────────────────
def detect_and_correct_orientation(image_path: str) -> int:
    """
    Detects orientation and rotates the image if needed.
    Returns the angle (0, 90, 180, 270) the image was rotated BY to correct it.
    """
    try:
        img = Image.open(image_path)
        # 1. Preprocess: Grayscale + EXIF Transpose
        # Tesseract OSD is often more reliable on grayscale high-contrast images.
        img_prep = ImageOps.exif_transpose(img).convert('L')
        
        # 2. Detect orientation using Tesseract OSD
        try:
            osd = pytesseract.image_to_osd(img_prep, output_type=pytesseract.Output.DICT)
            rotate_angle = osd["rotate"]
            confidence = osd["orientation_conf"]
            
            # Reduce threshold slightly to be more aggressive but check for low conf
            if rotate_angle != 0 and confidence > 5.0: 
                print(f"    [OSD] Detected {osd['orientation']}° orientation. Rotating by {rotate_angle}° (Conf: {confidence:.1f})")
                corrected_img = img.rotate(-rotate_angle, expand=True)
                corrected_img.save(image_path)
                return rotate_angle
            else:
                print(f"    [OSD] Status: Upright (no rotation needed).")
            return 0
        except Exception as osd_e:
            # Low-confidence or too little text - don't rotate
            if "Too few characters" in str(osd_e):
                print(f"    [OSD] Warning: Too few characters to detect orientation. Skipping rotation.")
            else:
                print(f"    [OSD] Orientation detection skipped: {osd_e}")
            return 0
            
    except Exception as e:
        print(f"    [ERROR] Orientation correction failed: {e}")
        return 0


def run_ocr_on_images(
    image_paths: List[str],
    document_name: str,
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """
    Run PaddleOCR on a list of images and return structured result dicts.

    Each dict contains:
        document, source_image, page, text, confidence, x1, y1, x2, y2
    """
    from paddleocr import PaddleOCR

    ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
    all_results = []

    for page_idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        print(f"  OCR on {filename} ...")
        result = ocr_engine.ocr(img_path, cls=True)

        if not result or not result[0]:
            continue

        for line in result[0]:
            box  = line[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            text = line[1][0]
            conf = line[1][1]

            x1 = int(box[0][0]);  y1 = int(box[0][1])
            x2 = int(box[2][0]);  y2 = int(box[2][1])

            all_results.append({
                "document":     document_name,
                "source_image": filename,
                "page":         page_idx,
                "text":         text,
                "confidence":   round(conf, 4),
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
            })

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ──────────────────────────────────────────────────────────────────────────────
def is_page_native_text(page: Any, min_chars: int = 50) -> bool:
    """
    Returns True if the specific PyMuPDF page contains enough selectable text
    to process natively.
    """
    try:
        text = page.get_text()
        return len(text.strip()) >= min_chars
    except Exception:
        return False


def extract_native_page_words(
    page: Any,
    page_idx: int,
    document_name: str,
    pdf_stem: str,
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """
    Extract words from a single native-text PDF page with exact bounding boxes.
    """
    scale = dpi / 72.0   # PDF points -> image pixels
    results = []
    
    # get_text("words") returns (x0,y0,x1,y1,word,block_no,line_no,word_no)
    words = page.get_text("words")
    img_filename = f"{pdf_stem}_page_{page_idx:03d}.png"

    for w in words:
        x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
        if not word_text.strip():
            continue
        results.append({
            "document":     document_name,
            "source_image": img_filename,
            "page":         page_idx,
            "text":         word_text.strip(),
            "confidence":   1.0,           # native extraction is exact
            "x1": int(x0 * scale),
            "y1": int(y0 * scale),
            "x2": int(x1 * scale),
            "y2": int(y1 * scale),
        })

    return results


def images_to_pdf(image_paths: List[str], output_pdf_path: str) -> None:
    """
    Stitch a list of image files (in order) into a single PDF.
    Every image becomes one page. Uses PyMuPDF — no extra dependency needed.
    """
    import fitz
    out_doc = fitz.open()
    for img_path in image_paths:
        img_doc = fitz.open(img_path)
        pdf_bytes = img_doc.convert_to_pdf()
        img_pdf = fitz.open("pdf", pdf_bytes)
        out_doc.insert_pdf(img_pdf)
        img_doc.close()
        img_pdf.close()
    out_doc.save(output_pdf_path, garbage=4, deflate=True)
    out_doc.close()
    print(f"  [OK] PDF assembled from {len(image_paths)} page(s) -> {output_pdf_path}")


def process_pdf(
    pdf_path: str,
    keywords: List[str],
    output_folder: str,
    dpi: int,
    do_highlight: bool,
    all_matches: List[Dict[str, Any]],
):
    """
    Image-first PDF pipeline — for every page:
      1. Render page  -> PNG
      2. Detect & correct rotation  (overwrites the PNG in-place)
      3. OCR on the upright image
      4. Classify keywords
      5. If matches: highlight the PNG -> save to highlighted_images/<doc>/
         If no matches: use the clean rendered PNG as-is
    After all pages:
      6. Stitch ALL page PNGs (highlighted or plain) into one output PDF
         so every page is always present in the correct order.
    """
    import fitz
    from keyword_classifier import classify_ocr_results
    from image_highlighter import highlight_image

    doc_name = Path(pdf_path).stem
    print(f"\n{'='*60}")
    print(f"[PDF] {os.path.basename(pdf_path)}")
    print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    pdf_matches = []

    # Zoom for rendering pages to images
    zoom   = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    # Folder for per-page rendered PNGs
    image_folder = os.path.join(output_folder, "page_images", doc_name)
    os.makedirs(image_folder, exist_ok=True)

    # Folder for highlighted (or plain) page PNGs that go into the final PDF
    hi_img_folder = os.path.join(output_folder, "highlighted_images", doc_name)
    os.makedirs(hi_img_folder, exist_ok=True)

    # Ordered list of page images to stitch into the final PDF
    final_page_images: List[str] = []

    for page_idx, page in enumerate(doc, start=1):
        print(f"  Page {page_idx}/{doc.page_count}:")

        # ── Step 1: Render page to PNG ──────────────────────────────────────
        img_path = os.path.join(image_folder, f"{doc_name}_page_{page_idx:03d}.png")
        pix = page.get_pixmap(matrix=matrix)
        pix.save(img_path)
        print(f"    [1] Rendered -> {os.path.basename(img_path)}")

        # ── Step 2: Detect & correct rotation ──────────────────────────────
        use_native = is_page_native_text(page)
        if use_native:
            # Native-text pages are already correctly oriented by PyMuPDF.
            print(f"    [2] Native text page — skipping rotation check")
        else:
            detect_and_correct_orientation(img_path)   # overwrites img_path if rotated

        # ── Step 3: OCR ────────────────────────────────────────────────────
        if use_native:
            print(f"    [3] Extracting native words")
            page_results = extract_native_page_words(
                page, page_idx, doc_name, doc_name, dpi
            )
        else:
            print(f"    [3] Running PaddleOCR on corrected image")
            page_results = run_ocr_on_images(
                [img_path], document_name=doc_name, dpi=dpi
            )

        # ── Step 4: Classify keywords ───────────────────────────────────────
        matches = classify_ocr_results(page_results, keywords)
        print(f"    [4] Keyword matches: {len(matches)}")

        # ── Step 5: Highlight image or keep clean page ──────────────────────
        fname = os.path.basename(img_path)
        hi_path = os.path.join(hi_img_folder, fname.replace(".", "_highlighted."))

        if do_highlight and matches:
            pdf_matches.extend(matches)
            highlight_image(
                image_path=img_path,
                matches=matches,
                output_path=hi_path,
                keywords=keywords,
            )
            final_page_images.append(hi_path)
            print(f"    [5] Highlighted -> {os.path.basename(hi_path)}")
        else:
            # No keyword matches (or highlighting disabled): use the clean image.
            if matches:
                pdf_matches.extend(matches)
            final_page_images.append(img_path)
            print(f"    [5] No matches — clean page included as-is")

    doc.close()

    # ── Step 6: Stitch ALL page images into one PDF ─────────────────────────
    if do_highlight:
        highlighted_pdf = os.path.join(output_folder, f"{doc_name}_highlighted.pdf")
        images_to_pdf(final_page_images, highlighted_pdf)

    if pdf_matches:
        all_matches.extend(pdf_matches)


def process_image(
    image_path: str,
    keywords: List[str],
    output_folder: str,
    all_matches: List[Dict[str, Any]],
):
    """Run OCR on a standalone image → classify → highlight."""
    from keyword_classifier import classify_ocr_results
    from image_highlighter import highlight_image

    doc_name = Path(image_path).stem
    print(f"\n{'='*60}")
    print(f"[IMG] {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # 1. Orientation Correction
    detect_and_correct_orientation(image_path)

    # 2. OCR
    ocr_results = run_ocr_on_images([image_path], document_name=doc_name)
    print(f"  [OK] OCR extracted {len(ocr_results)} text tokens")

    # Classify
    matches = classify_ocr_results(ocr_results, keywords)
    print(f"  [OK] Found {len(matches)} keyword matches")

    if matches:
        all_matches.extend(matches)

        # Highlight image
        hi_path = os.path.join(
            output_folder,
            Path(image_path).stem + "_highlighted" + Path(image_path).suffix,
        )
        highlight_image(
            image_path=image_path,
            matches=matches,
            output_path=hi_path,
            keywords=keywords,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    from keyword_classifier import load_keywords
    from excel_reporter import save_classification_excel

    # ── Load keywords ──────────────────────────────────────────────────────
    keywords = load_keywords(args.keywords)
    if not keywords:
        print("[ERROR] No keywords loaded. Please add keywords to keywords.txt")
        sys.exit(1)

    print(f"\nKeywords to match: {keywords}\n")

    # ── Resolve input files ────────────────────────────────────────────────
    input_path = os.path.abspath(args.input)
    pdf_files:   List[str] = []
    image_files: List[str] = []

    if os.path.isfile(input_path):
        ext = Path(input_path).suffix.lower()
        if ext in PDF_EXTS:
            pdf_files.append(input_path)
        elif ext in IMAGE_EXTS:
            image_files.append(input_path)
        else:
            print(f"[ERROR] Unsupported file type: {ext}")
            sys.exit(1)

    elif os.path.isdir(input_path):
        for ext in PDF_EXTS:
            pdf_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        for ext in IMAGE_EXTS:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))

    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    total = len(pdf_files) + len(image_files)
    if total == 0:
        print("[ERROR] No PDF or image files found at the given input path.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) and {len(image_files)} image(s) to process.\n")

    # ── Prepare output folder ──────────────────────────────────────────────
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)

    # ── Process files ──────────────────────────────────────────────────────
    all_matches: List[Dict[str, Any]] = []

    for pdf in pdf_files:
        try:
            process_pdf(
                pdf_path=pdf,
                keywords=keywords,
                output_folder=output_folder,
                dpi=args.dpi,
                do_highlight=not args.no_highlight,
                all_matches=all_matches,
            )
        except Exception as e:
            print(f"  [ERROR] processing {pdf}: {e}")
            import traceback; traceback.print_exc()

    for img in image_files:
        try:
            process_image(
                image_path=img,
                keywords=keywords,
                output_folder=output_folder,
                all_matches=all_matches,
            )
        except Exception as e:
            print(f"  [ERROR] processing {img}: {e}")
            import traceback; traceback.print_exc()

    # ── Save Excel report ──────────────────────────────────────────────────
    excel_path = os.path.join(output_folder, "classification_results.xlsx")
    if all_matches:
        save_classification_excel(all_matches, excel_path)
        print(f"\n[OK] Excel report: {excel_path}")
    else:
        print("\n[WARN] No keyword matches found across all files. Excel not generated.")

    print(f"\n[DONE] Pipeline complete. Output folder: {output_folder}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Keyword Classification & Highlighting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", "-i",
        default=".",
        help="Path to a PDF, image file, or folder to scan (default: current directory)",
    )
    p.add_argument(
        "--keywords", "-k",
        default="keywords.txt",
        help="Path to keywords file (default: keywords.txt)",
    )
    p.add_argument(
        "--output", "-o",
        default="output",
        help="Output folder for highlighted files and Excel (default: output/)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when rendering PDF pages to images (default: 300)",
    )
    p.add_argument(
        "--no-highlight",
        action="store_true",
        default=False,
        help="Skip generating highlighted PDFs/images (Excel report only)",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
