"""
Extract Documents Pipeline

Batch processes PDFs from data/raw/pdf/ using document_loader.py
and saves page-marked text files to data/processed/text/.
"""

import logging
from pathlib import Path
import sys

# Ensure root directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.document_loader import load_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("extract_documents")


def process_all_pdfs(raw_dir: Path, output_dir: Path) -> None:
    """
    Finds all PDF files in raw_dir, extracts text using load_pdf(),
    and saves output to output_dir preserving page markers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir.resolve()}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {raw_dir.resolve()}")

    for pdf_path in pdf_files:
        txt_filename = pdf_path.stem + ".txt"
        output_path = output_dir / txt_filename
        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            pages = load_pdf(pdf_path)
            total_pages = len(pages)
            total_chars = 0

            formatted_pages = []
            for page_info in pages:
                page_num = page_info["page_number"]
                text = page_info["text"]
                total_chars += len(text)
                formatted_pages.append(f"--- PAGE {page_num} ---\n{text}")

            full_content = "\n\n".join(formatted_pages)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            logger.info(
                f"Successfully extracted {pdf_path.name}: "
                f"{total_pages} pages, {total_chars} chars -> {output_path.resolve()}"
            )

        except Exception as e:
            logger.error(f"Failed processing {pdf_path.name}: {e}")
            raise


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_pdf_dir = base_dir / "data" / "raw" / "pdf"
    processed_text_dir = base_dir / "data" / "processed" / "text"

    process_all_pdfs(raw_pdf_dir, processed_text_dir)
