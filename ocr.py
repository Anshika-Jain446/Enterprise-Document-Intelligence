import io
import os

from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# ============================================================
# OCR PROCESSOR
# ============================================================

class OCRProcessor:

    def __init__(self, tesseract_cmd=None):

        if pytesseract is None:
            raise RuntimeError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract"
            )

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed. "
                "Install it with: pip install pymupdf"
            )

        # ----------------------------------------------------
        # WINDOWS TESSERACT PATH
        # ----------------------------------------------------

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_cmd
            )

        elif os.name == "nt":

            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]

            for path in possible_paths:

                if os.path.exists(path):

                    pytesseract.pytesseract.tesseract_cmd = (
                        path
                    )

                    break

    # ========================================================
    # SUPPORTED IMAGE FORMATS
    # ========================================================

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
    }

    SUPPORTED_PDF_EXTENSIONS = {
        ".pdf",
    }

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    def is_image_file(self, file_path):

        extension = os.path.splitext(
            str(file_path)
        )[1].lower()

        return extension in self.SUPPORTED_IMAGE_EXTENSIONS

    # ========================================================
    # CHECK PDF
    # ========================================================

    def is_pdf_file(self, file_path):

        extension = os.path.splitext(
            str(file_path)
        )[1].lower()

        return extension in self.SUPPORTED_PDF_EXTENSIONS

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    def preprocess_image(self, image):

        if not isinstance(image, Image.Image):

            raise ValueError(
                "Expected a PIL Image."
            )

        # Convert to RGB
        image = image.convert("RGB")

        # Grayscale
        image = ImageOps.grayscale(image)

        # Increase contrast
        image = ImageOps.autocontrast(image)

        # Sharpen
        image = image.filter(
            ImageFilter.SHARPEN
        )

        return image

    # ========================================================
    # OCR PIL IMAGE
    # ========================================================

    def extract_text_from_pil(
        self,
        image,
        preprocess=True,
        lang="eng",
    ):

        if preprocess:

            image = self.preprocess_image(
                image
            )

        try:

            text = pytesseract.image_to_string(
                image,
                lang=lang,
            )

        except Exception as e:

            raise RuntimeError(
                f"OCR failed: {e}"
            ) from e

        return str(
            text or ""
        ).strip()

    # ========================================================
    # OCR IMAGE BYTES
    # ========================================================

    def extract_text_from_bytes(
        self,
        image_bytes,
        preprocess=True,
        lang="eng",
    ):

        if not image_bytes:

            return ""

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

        except Exception as e:

            raise ValueError(
                f"Could not open image: {e}"
            ) from e

        return self.extract_text_from_pil(
            image=image,
            preprocess=preprocess,
            lang=lang,
        )

    # ========================================================
    # OCR IMAGE FILE
    # ========================================================

    def extract_text_from_image(
        self,
        image_path,
        preprocess=True,
        lang="eng",
    ):

        if not image_path:

            raise ValueError(
                "Image path cannot be empty."
            )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not self.is_image_file(
            image_path
        ):

            raise ValueError(
                "Unsupported image format."
            )

        try:

            image = Image.open(
                image_path
            )

        except Exception as e:

            raise ValueError(
                f"Could not open image: {e}"
            ) from e

        return self.extract_text_from_pil(
            image=image,
            preprocess=preprocess,
            lang=lang,
        )

    # ========================================================
    # OCR PDF PAGE
    # ========================================================

    def extract_text_from_pdf_page(
        self,
        page,
        dpi=200,
        lang="eng",
    ):

        if page is None:

            raise ValueError(
                "PDF page cannot be None."
            )

        try:

            zoom = dpi / 72

            matrix = fitz.Matrix(
                zoom,
                zoom,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            return self.extract_text_from_pil(
                image=image,
                preprocess=True,
                lang=lang,
            )

        except Exception as e:

            raise RuntimeError(
                f"PDF page OCR failed: {e}"
            ) from e

    # ========================================================
    # OCR SCANNED PDF
    # ========================================================

    def extract_text_from_pdf(
        self,
        pdf_path,
        dpi=200,
        lang="eng",
    ):

        if not pdf_path:

            raise ValueError(
                "PDF path cannot be empty."
            )

        if not os.path.exists(pdf_path):

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not self.is_pdf_file(
            pdf_path
        ):

            raise ValueError(
                "File is not a PDF."
            )

        try:

            document = fitz.open(
                pdf_path
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not open PDF: {e}"
            ) from e

        pages = []

        try:

            for page_number in range(
                len(document)
            ):

                page = document[
                    page_number
                ]

                text = self.extract_text_from_pdf_page(
                    page=page,
                    dpi=dpi,
                    lang=lang,
                )

                pages.append(
                    {
                        "page": page_number + 1,
                        "text": text,
                    }
                )

        finally:

            document.close()

        return pages

    # ========================================================
    # AUTO PROCESS FILE
    # ========================================================

    def process_file(
        self,
        file_path,
        dpi=200,
        lang="eng",
    ):

        if not file_path:

            raise ValueError(
                "File path cannot be empty."
            )

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if self.is_image_file(
            file_path
        ):

            text = self.extract_text_from_image(
                image_path=file_path,
                preprocess=True,
                lang=lang,
            )

            return {
                "file": file_path,
                "file_type": "image",
                "text": text,
                "pages": [
                    {
                        "page": 1,
                        "text": text,
                    }
                ],
            }

        if self.is_pdf_file(
            file_path
        ):

            pages = self.extract_text_from_pdf(
                pdf_path=file_path,
                dpi=dpi,
                lang=lang,
            )

            combined_text = "\n\n".join(
                page["text"]
                for page in pages
                if page.get("text")
            )

            return {
                "file": file_path,
                "file_type": "pdf",
                "text": combined_text,
                "pages": pages,
            }

        raise ValueError(
            f"Unsupported file format: {file_path}"
        )


# ============================================================
# SIMPLE HELPER FUNCTIONS
# ============================================================

def extract_image_text(
    image_path,
    lang="eng",
):

    processor = OCRProcessor()

    return processor.extract_text_from_image(
        image_path=image_path,
        lang=lang,
    )


def extract_pdf_ocr(
    pdf_path,
    dpi=200,
    lang="eng",
):

    processor = OCRProcessor()

    return processor.extract_text_from_pdf(
        pdf_path=pdf_path,
        dpi=dpi,
        lang=lang,
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python ocr.py <image-or-pdf-path>"
        )

        sys.exit(1)

    file_path = sys.argv[1]

    processor = OCRProcessor()

    try:

        result = processor.process_file(
            file_path
        )

        print(
            "\n=================================================="
        )

        print(
            "OCR RESULT"
        )

        print(
            "==================================================\n"
        )

        print(
            result["text"]
        )

    except Exception as e:

        print(
            f"OCR ERROR: {e}"
        )

        sys.exit(1)

