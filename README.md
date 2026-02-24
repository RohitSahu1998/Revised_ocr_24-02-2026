# AI Keyword Classification & Orientation Correction System

A professional-grade tool to automatically detect, rotate, and highlight specific keywords in PDFs and images.

## Features
- **Automatic Orientation Fix**: Detects 90°, 180°, and 270° rotations using Tesseract OSD and corrects them before processing.
- **Hybrid OCR Engine**: Uses PaddleOCR for high-accuracy text extraction on scanned documents and native extraction for digital PDFs.
- **Precise Highlighting**: Draws exact bounding boxes over matched keywords.
- **Excel Reporting**: Generates a detailed audit trail of all findings.
- **Batch Processing**: Process entire folders with one click.

## Installation

1. **Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Tesseract OCR (Required for Rotation Detection)**:
   - Install Tesseract OCR from [here](https://github.com/UB-Mannheim/tesseract/wiki).
   - Ensure `tesseract.exe` is in your PATH or update the path in `classify_and_highlight.py`.

## Usage

### One-Click (Windows)
1. Place your files in the `input_files` folder.
2. Double-click `run_pipeline.bat`.

### Command Line
```bash
python classify_and_highlight.py --input <file_or_folder> --output results
```

## Configuration
- Add your target keywords to `keywords.txt` (one per line).
- Results are saved in the `output/` directory by default.
