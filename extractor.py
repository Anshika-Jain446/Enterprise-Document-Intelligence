import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd


class DocumentExtractor:

    def __init__(self, pdf_path):

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.pdf_path = pdf_path


    # ---------------------------------
    # Extract Metadata
    # ---------------------------------
    def extract_metadata(self):

        try:
            doc = fitz.open(self.pdf_path)

            metadata = doc.metadata

            info = {
                "file_name": os.path.basename(self.pdf_path),
                "title": metadata.get("title") or "Unknown",
                "author": metadata.get("author") or "Unknown",
                "subject": metadata.get("subject") or "Unknown",
                "creator": metadata.get("creator") or "Unknown",
                "producer": metadata.get("producer") or "Unknown",
                "pages": len(doc)
            }

            doc.close()

            return info

        except Exception as e:
            return {
                "error": f"Metadata extraction failed: {str(e)}"
            }


    # ---------------------------------
    # Extract Complete Text
    # ---------------------------------
    def extract_text(self):

        try:
            doc = fitz.open(self.pdf_path)

            pages = []
            text_chunks = []

            for page_number, page in enumerate(doc):

                text = page.get_text()

                pages.append({
                    "page": page_number + 1,
                    "text": text
                })

                text_chunks.append(text)

            doc.close()

            full_text = "\n".join(text_chunks)

            return full_text, pages

        except Exception as e:
            return "", [{
                "error": f"Text extraction failed: {str(e)}"
            }]


    # ---------------------------------
    # Extract Tables
    # ---------------------------------
    def extract_tables(self):

        tables = []

        try:

            with pdfplumber.open(self.pdf_path) as pdf:

                for page_number, page in enumerate(pdf.pages):

                    extracted_tables = page.extract_tables() or []

                    for table in extracted_tables:

                        if not table or len(table) < 2:
                            continue


                        headers = []

                        for i, col in enumerate(table[0]):

                            if col is None or str(col).strip() == "":
                                headers.append(f"Column_{i}")

                            else:
                                headers.append(
                                    f"{str(col).strip()}_{i}"
                                )


                        try:

                            df = pd.DataFrame(
                                table[1:],
                                columns=headers
                            )


                            tables.append({

                                "page": page_number + 1,

                                "rows": len(df),

                                "columns": len(df.columns),

                                "table": df.to_dict(
                                    orient="records"
                                )

                            })


                        except Exception as e:

                            tables.append({

                                "page": page_number + 1,

                                "error": str(e)

                            })


        except Exception as e:

            tables.append({

                "error": f"Table extraction failed: {str(e)}"

            })


        return tables



    # ---------------------------------
    # Extract Images
    # ---------------------------------
    def extract_images(self):

        images = []

        try:

            doc = fitz.open(self.pdf_path)


            for page_number, page in enumerate(doc):

                image_list = page.get_images(full=True)


                images.append({

                    "page": page_number + 1,

                    "count": len(image_list)

                })


            doc.close()


        except Exception as e:

            images.append({

                "error": f"Image extraction failed: {str(e)}"

            })


        return images



    # ---------------------------------
    # Document Statistics
    # ---------------------------------
    def document_statistics(self, text):

        return {

            "words": len(text.split()),

            "characters": len(text),

            "lines": len(text.splitlines())

        }



    # ---------------------------------
    # Complete Document Extraction
    # ---------------------------------
    def extract_document(self):

        text, pages = self.extract_text()

        document = {

            "metadata": self.extract_metadata(),

            "statistics": self.document_statistics(text),

            "text": text,

            "pages": pages,

            "tables": self.extract_tables(),

            "images": self.extract_images()

        }


        return document



# ---------------------------------
# Example Usage
# ---------------------------------

if __name__ == "__main__":

    pdf_file = "sample.pdf"


    extractor = DocumentExtractor(pdf_file)


    result = extractor.extract_document()


    print("\nMETADATA")
    print(result["metadata"])


    print("\nSTATISTICS")
    print(result["statistics"])


    print("\nTOTAL TABLES:",
          len(result["tables"]))


    print("\nTOTAL IMAGE PAGES:",
          len(result["images"]))


    print("\nTEXT PREVIEW:")
    print(result["text"][:500])