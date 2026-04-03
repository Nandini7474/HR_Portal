
import os
# Must be set before torch/easyocr are imported to prevent OpenMP segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pdfplumber

#lazy_singletons
_ocr_reader = None

# Returns a singleton EasyOCR reader instance
def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("[OCR] Initialising EasyOCR (first OCR call)...", flush=True)
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        print("[OCR] EasyOCR ready.", flush=True)
    return _ocr_reader


#core_function
 # Extracts text from a PDF file, using OCR if needed
def extract_text_from_pdf(file_path: str) -> str:
 
    text = _try_pdfplumber(file_path)
    if text.strip():
        return text

    print(f"[OCR] No text layer in {file_path!r} -> falling back to OCR...", flush=True)
    return _ocr_with_pypdfium2(file_path)


 # Attempts to extract text from a PDF using pdfplumber
def _try_pdfplumber(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] pdfplumber error on {file_path!r}: {e}", flush=True)
    return text


 # Uses pypdfium2 and EasyOCR to extract text from image-based PDFs
def _ocr_with_pypdfium2(file_path: str) -> str:
    """Render each PDF page to an image via pypdfium2, then run EasyOCR."""
    text = ""
    try:
        import pypdfium2 as pdfium
        reader = _get_ocr_reader()

        MAX_DIM = 1600  # cap longest edge to keep memory under ~30 MB per page

        pdf = pdfium.PdfDocument(file_path)
        for page_index in range(len(pdf)):
            page   = pdf[page_index]
            bitmap = page.render(scale=1)
            pil_img = bitmap.to_pil().convert("RGB")

            # Downscale if the page image is too large (prevents OOM in PyTorch)
            w, h = pil_img.size
            if max(w, h) > MAX_DIM:
                ratio   = MAX_DIM / max(w, h)
                pil_img = pil_img.resize(
                    (max(1, int(w * ratio)), max(1, int(h * ratio)))
                )

            img_np = np.array(pil_img)

            results = reader.readtext(img_np, detail=0, paragraph=True)
            text += " ".join(results) + "\n"

        pdf.close()
    except Exception as e:
        print(f"[ERROR] OCR failed on {file_path!r}: {e}", flush=True)

    return text
