from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os
from pathlib import Path

def ocr_pdf_to_text(path_str: str, dpi=300) -> str:
    # Convert each page to an image, OCR, join
    # Requires poppler (for convert_from_path) and tesseract installed
    pages = convert_from_path(path_str, dpi=dpi)
    texts = []
    for img in pages:
        txt = pytesseract.image_to_string(img)
        texts.append(txt)
    return "\n\n".join(texts)

def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    if ext in {".md", ".markdown"}:
        from markdown import markdown
        from bs4 import BeautifulSoup
        html = markdown(data.decode("utf-8", errors="ignore"))
        return "".join(BeautifulSoup(html, "html.parser").stripped_strings)
    if ext == ".rtf":
        from striprtf.striprtf import rtf_to_text
        return rtf_to_text(data.decode("utf-8", errors="ignore"))
    if ext == ".pdf":
        # 1) try digital text extraction
        txt = extract_text(str(path)) or ""
        if txt.strip():
            return txt
        # 2) fallback to OCR if empty
        print(f"⚠️  No extractable text in {path.name}. Falling back to OCR…")
        ocr_txt = ocr_pdf_to_text(str(path))
        return ocr_txt or ""
    raise ValueError(f"Unsupported format: {ext}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert documents to plain text")
    parser.add_argument("--src", required=True, help="Source document path")
    parser.add_argument("--dst", required=True, help="Destination directory for normalized text")
    args = parser.parse_args()
    
    src_path = Path(args.src)
    dst_dir = Path(args.dst)
    
    # Ensure destination directory exists
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    try:
        text_content = load_text(src_path)
        
        # Create output filename
        output_file = dst_dir / f"{src_path.stem}.txt"
        
        # Save the text content
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text_content)
            
        print(f"✓ Successfully converted {src_path.name} to {output_file}")
        
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")