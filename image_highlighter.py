"""
Image Highlighter Module
Draws colored bounding boxes and keyword labels on images for matched keywords.
Uses OpenCV (preferred) with a Pillow fallback.
"""

import os
from typing import List, Dict, Any


# Color palette for different keywords (BGR for OpenCV, RGB for Pillow)
_COLORS_BGR = [
    (0, 200, 0),     # green
    (0, 128, 255),   # orange
    (255, 0, 128),   # pink/magenta
    (0, 255, 255),   # yellow
    (255, 0, 0),     # blue
    (128, 0, 255),   # purple
]

_COLORS_RGB = [(r, g, b) for (b, g, r) in _COLORS_BGR]


def _keyword_color_bgr(keyword: str, keywords: List[str]):
    idx = keywords.index(keyword) % len(_COLORS_BGR) if keyword in keywords else 0
    return _COLORS_BGR[idx]


def _keyword_color_rgb(keyword: str, keywords: List[str]):
    idx = keywords.index(keyword) % len(_COLORS_RGB) if keyword in keywords else 0
    return _COLORS_RGB[idx]


def highlight_image(
    image_path: str,
    matches: List[Dict[str, Any]],
    output_path: str,
    keywords: List[str] = None,
    box_thickness: int = 2,
) -> str:
    """
    Draw colored bounding boxes around matched keywords on an image.

    Args:
        image_path:     Path to the source image
        matches:        List of match dicts (from keyword_classifier), each needs:
                        x1, y1, x2, y2, keyword, text
        output_path:    Where to save the annotated image
        keywords:       Full keyword list (for consistent color assignment)
        box_thickness:  Thickness of the bounding box border in pixels

    Returns:
        Absolute path to the highlighted image
    """
    if keywords is None:
        keywords = list({m.get("keyword", "") for m in matches})

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        return _highlight_with_opencv(image_path, matches, output_path, keywords, box_thickness)
    except (ImportError, Exception) as e:
        print(f"  OpenCV failed ({e}), falling back to Pillow...")
        return _highlight_with_pillow(image_path, matches, output_path, keywords, box_thickness)


def _highlight_with_opencv(
    image_path: str,
    matches: List[Dict[str, Any]],
    output_path: str,
    keywords: List[str],
    box_thickness: int,
) -> str:
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    overlay = img.copy()

    for m in matches:
        x1, y1 = int(m["x1"]), int(m["y1"])
        x2, y2 = int(m["x2"]), int(m["y2"])
        kw = m.get("keyword", "")
        color = _keyword_color_bgr(kw, keywords)

        # Filled semi-transparent rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Edge border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Label text above the box
        label = kw[:25]  # truncate if too long
        font_scale = 0.45
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, label_y - th - 2), (x1 + tw + 2, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 1, label_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay for transparency
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.imwrite(output_path, img)
    print(f"[OK] Image highlighted ({len(matches)} matches) -> {output_path}")
    return os.path.abspath(output_path)


def _highlight_with_pillow(
    image_path: str,
    matches: List[Dict[str, Any]],
    output_path: str,
    keywords: List[str],
    box_thickness: int,
) -> str:
    from PIL import Image, ImageDraw, ImageFont
    import os

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(img)

    for m in matches:
        x1, y1 = int(m["x1"]), int(m["y1"])
        x2, y2 = int(m["x2"]), int(m["y2"])
        kw = m.get("keyword", "")
        r, g, b = _keyword_color_rgb(kw, keywords)

        # Semi-transparent fill
        draw_overlay.rectangle([x1, y1, x2, y2], fill=(r, g, b, 90))
        # Solid border
        draw_main.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=box_thickness)

        # Label above box
        label = kw[:25]
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw_main.textbbox((x1, max(y1 - 16, 0)), label, font=font)
        draw_main.rectangle(bbox, fill=(r, g, b))
        draw_main.text((x1, max(y1 - 16, 0)), label, fill=(255, 255, 255), font=font)

    # Composite overlay
    img = Image.alpha_composite(img, overlay).convert("RGB")
    img.save(output_path)
    print(f"[OK] Image highlighted (Pillow, {len(matches)} matches) -> {output_path}")
    return os.path.abspath(output_path)
