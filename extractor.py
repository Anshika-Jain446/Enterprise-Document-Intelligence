import os
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from openpyxl import load_workbook


class DocumentExtractor:

    def __init__(self, file_path: str):

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Document file not found: {file_path}"
            )

        self.file_path = file_path

        # Compatibility with existing application
        self.pdf_path = file_path

        self.extension = os.path.splitext(
            file_path
        )[1].lower()

        self.file_name = os.path.basename(
            file_path
        )

        self.base_name = os.path.splitext(
            self.file_name
        )[0]

        self.image_output_dir = os.path.join(
            "outputs",
            "images",
            self.base_name
        )

        self.visual_output_dir = os.path.join(
            "outputs",
            "visuals",
            self.base_name
        )

        os.makedirs(
            self.image_output_dir,
            exist_ok=True
        )

        os.makedirs(
            self.visual_output_dir,
            exist_ok=True
        )

    # ==========================================================
    # Metadata
    # ==========================================================

    def extract_metadata(self):

        try:

            metadata = {
                "file_name": self.file_name,
                "file_type": self.extension,
                "title": self.base_name,
                "author": "Unknown",
                "subject": "Unknown",
                "creator": "Unknown",
                "producer": "Unknown",
                "pages": 1
            }

            # --------------------------------------------------
            # PDF
            # --------------------------------------------------

            if self.extension == ".pdf":

                with fitz.open(
                    self.file_path
                ) as doc:

                    pdf_metadata = (
                        doc.metadata or {}
                    )

                    metadata.update({
                        "title":
                            pdf_metadata.get(
                                "title"
                            ) or self.base_name,

                        "author":
                            pdf_metadata.get(
                                "author"
                            ) or "Unknown",

                        "subject":
                            pdf_metadata.get(
                                "subject"
                            ) or "Unknown",

                        "creator":
                            pdf_metadata.get(
                                "creator"
                            ) or "Unknown",

                        "producer":
                            pdf_metadata.get(
                                "producer"
                            ) or "Unknown",

                        "pages":
                            len(doc),

                        "document_type":
                            "PDF Document"
                    })

            # --------------------------------------------------
            # DOCX
            # --------------------------------------------------

            elif self.extension == ".docx":

                metadata["document_type"] = (
                    "Microsoft Word Document"
                )

                with zipfile.ZipFile(
                    self.file_path,
                    "r"
                ) as archive:

                    try:

                        xml_data = archive.read(
                            "docProps/core.xml"
                        )

                        root = ET.fromstring(
                            xml_data
                        )

                        namespaces = {
                            "dc":
                                "http://purl.org/dc/elements/1.1/"
                        }

                        title = root.find(
                            "dc:title",
                            namespaces
                        )

                        creator = root.find(
                            "dc:creator",
                            namespaces
                        )

                        subject = root.find(
                            "dc:subject",
                            namespaces
                        )

                        if (
                            title is not None
                            and title.text
                        ):
                            metadata["title"] = (
                                title.text
                            )

                        if (
                            creator is not None
                            and creator.text
                        ):
                            metadata["author"] = (
                                creator.text
                            )

                        if (
                            subject is not None
                            and subject.text
                        ):
                            metadata["subject"] = (
                                subject.text
                            )

                    except KeyError:
                        pass

            # --------------------------------------------------
            # PPTX
            # --------------------------------------------------

            elif self.extension == ".pptx":

                metadata["document_type"] = (
                    "PowerPoint Presentation"
                )

                with zipfile.ZipFile(
                    self.file_path,
                    "r"
                ) as archive:

                    slide_files = [
                        name
                        for name in archive.namelist()
                        if name.startswith(
                            "ppt/slides/slide"
                        )
                        and name.endswith(".xml")
                    ]

                    metadata["pages"] = len(
                        slide_files
                    )

            # --------------------------------------------------
            # XLSX
            # --------------------------------------------------

            elif self.extension == ".xlsx":

                metadata["document_type"] = (
                    "Excel Workbook"
                )

                workbook = load_workbook(
                    self.file_path,
                    read_only=True,
                    data_only=True
                )

                metadata["sheets"] = len(
                    workbook.sheetnames
                )

                metadata["sheet_names"] = (
                    workbook.sheetnames
                )

                metadata["pages"] = max(
                    1,
                    len(workbook.sheetnames)
                )

                workbook.close()

            # --------------------------------------------------
            # CSV
            # --------------------------------------------------

            elif self.extension == ".csv":

                metadata["document_type"] = (
                    "CSV Dataset"
                )

            # --------------------------------------------------
            # TXT / MD
            # --------------------------------------------------

            elif self.extension in [
                ".txt",
                ".md"
            ]:

                metadata["document_type"] = (
                    "Markdown Document"
                    if self.extension == ".md"
                    else "Text Document"
                )

            return metadata

        except Exception as e:

            return {
                "file_name": self.file_name,
                "file_type": self.extension,
                "pages": 1,
                "error":
                    f"Metadata extraction failed: {e}"
            }

    # ==========================================================
    # PDF Text
    # ==========================================================

    def _extract_pdf_text(self):

        pages = []
        text_chunks = []

        with fitz.open(
            self.file_path
        ) as doc:

            for page_num, page in enumerate(
                doc,
                start=1
            ):

                text = (
                    page.get_text("text")
                    or ""
                )

                pages.append({
                    "page": page_num,
                    "text": text
                })

                text_chunks.append(
                    text
                )

        return (
            "\n".join(text_chunks),
            pages
        )

    # ==========================================================
    # DOCX Text
    # ==========================================================

    def _extract_docx_text(self):

        text_parts = []

        with zipfile.ZipFile(
            self.file_path,
            "r"
        ) as archive:

            xml_data = archive.read(
                "word/document.xml"
            )

            root = ET.fromstring(
                xml_data
            )

            namespace = {
                "w":
                    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            }

            paragraphs = root.findall(
                ".//w:p",
                namespace
            )

            for paragraph in paragraphs:

                words = []

                for text_node in paragraph.findall(
                    ".//w:t",
                    namespace
                ):

                    if text_node.text:
                        words.append(
                            text_node.text
                        )

                paragraph_text = "".join(
                    words
                ).strip()

                if paragraph_text:
                    text_parts.append(
                        paragraph_text
                    )

        full_text = "\n".join(
            text_parts
        )

        pages = [{
            "page": 1,
            "text": full_text
        }]

        return (
            full_text,
            pages
        )

    # ==========================================================
    # TXT / Markdown
    # ==========================================================

    def _extract_text_file(self):

        with open(
            self.file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            text = file.read()

        pages = [{
            "page": 1,
            "text": text
        }]

        return (
            text,
            pages
        )

    # ==========================================================
    # CSV
    # ==========================================================

    def _extract_csv(self):

        tables = []

        dataframe = pd.read_csv(
            self.file_path
        )

        table_records = (
            dataframe
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

        tables.append({
            "page": 1,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "table": table_records,
            "source": "CSV"
        })

        text = dataframe.to_csv(
            index=False
        )

        pages = [{
            "page": 1,
            "text": text
        }]

        return (
            text,
            pages,
            tables
        )

    # ==========================================================
    # XLSX
    # ==========================================================

    def _extract_xlsx(self):

        tables = []
        text_parts = []

        workbook = load_workbook(
            self.file_path,
            data_only=True
        )

        for sheet in workbook.worksheets:

            rows = list(
                sheet.iter_rows(
                    values_only=True
                )
            )

            if not rows:
                continue

            headers = []

            for index, value in enumerate(
                rows[0]
            ):

                if value is None:
                    headers.append(
                        f"Column_{index}"
                    )

                else:
                    headers.append(
                        str(value)
                    )

            records = []

            for row in rows[1:]:

                record = {}

                for index, value in enumerate(
                    row
                ):

                    if index < len(headers):

                        record[
                            headers[index]
                        ] = (
                            ""
                            if value is None
                            else value
                        )

                records.append(
                    record
                )

            tables.append({
                "page": sheet.title,
                "sheet": sheet.title,
                "rows": len(records),
                "columns": len(headers),
                "table": records,
                "source": "XLSX"
            })

            text_parts.append(
                f"Sheet: {sheet.title}"
            )

            for row in rows:

                text_parts.append(
                    " | ".join(
                        ""
                        if value is None
                        else str(value)
                        for value in row
                    )
                )

        workbook.close()

        text = "\n".join(
            text_parts
        )

        pages = [{
            "page": 1,
            "text": text
        }]

        return (
            text,
            pages,
            tables
        )

    # ==========================================================
    # PPTX Text
    # ==========================================================

    def _extract_pptx_text(self):

        pages = []
        text_parts = []

        namespace = {
            "a":
                "http://schemas.openxmlformats.org/drawingml/2006/main"
        }

        with zipfile.ZipFile(
            self.file_path,
            "r"
        ) as archive:

            slide_files = [
                name
                for name in archive.namelist()
                if name.startswith(
                    "ppt/slides/slide"
                )
                and name.endswith(".xml")
            ]

            def slide_number(name):

                filename = os.path.basename(
                    name
                )

                number = (
                    os.path.splitext(
                        filename
                    )[0]
                    .replace(
                        "slide",
                        ""
                    )
                )

                try:
                    return int(number)
                except ValueError:
                    return 0

            slide_files.sort(
                key=slide_number
            )

            for slide_file in slide_files:

                slide_num = slide_number(
                    slide_file
                )

                xml_data = archive.read(
                    slide_file
                )

                root = ET.fromstring(
                    xml_data
                )

                texts = []

                for node in root.findall(
                    ".//a:t",
                    namespace
                ):

                    if node.text:
                        texts.append(
                            node.text
                        )

                slide_text = "\n".join(
                    texts
                )

                pages.append({
                    "page": slide_num,
                    "text": slide_text
                })

                text_parts.append(
                    f"Slide {slide_num}"
                )

                text_parts.append(
                    slide_text
                )

        return (
            "\n".join(text_parts),
            pages
        )

    # ==========================================================
    # Main Text Extraction
    # ==========================================================

    def extract_text(self):

        try:

            if self.extension == ".pdf":

                return (
                    self._extract_pdf_text()
                )

            elif self.extension == ".docx":

                return (
                    self._extract_docx_text()
                )

            elif self.extension in [
                ".txt",
                ".md"
            ]:

                return (
                    self._extract_text_file()
                )

            elif self.extension == ".csv":

                text, pages, _ = (
                    self._extract_csv()
                )

                return (
                    text,
                    pages
                )

            elif self.extension == ".xlsx":

                text, pages, _ = (
                    self._extract_xlsx()
                )

                return (
                    text,
                    pages
                )

            elif self.extension == ".pptx":

                return (
                    self._extract_pptx_text()
                )

            else:

                return (
                    "",
                    [{
                        "error":
                            "Unsupported file type: "
                            f"{self.extension}"
                    }]
                )

        except Exception as e:

            return (
                "",
                [{
                    "error":
                        f"Text extraction failed: {e}"
                }]
            )

    # ==========================================================
    # PDF Tables
    # ==========================================================

    def _extract_pdf_tables(self):

        tables = []

        with pdfplumber.open(
            self.file_path
        ) as pdf:

            for page_num, page in enumerate(
                pdf.pages,
                start=1
            ):

                extracted_tables = (
                    page.extract_tables()
                    or []
                )

                for table in extracted_tables:

                    if len(table) < 2:
                        continue

                    headers = []

                    for index, column in enumerate(
                        table[0]
                    ):

                        header = (
                            str(column).strip()
                            if column
                            else f"Column_{index}"
                        )

                        headers.append(
                            header
                        )

                    try:

                        dataframe = pd.DataFrame(
                            table[1:],
                            columns=headers
                        )

                        tables.append({
                            "page": page_num,
                            "rows": len(dataframe),
                            "columns": len(
                                dataframe.columns
                            ),
                            "table":
                                dataframe
                                .fillna("")
                                .to_dict(
                                    orient="records"
                                ),
                            "source": "PDF"
                        })

                    except Exception as error:

                        tables.append({
                            "page": page_num,
                            "error": str(error),
                            "source": "PDF"
                        })

        return tables

    # ==========================================================
    # Tables
    # ==========================================================

    def extract_tables(self):

        try:

            if self.extension == ".pdf":

                return (
                    self._extract_pdf_tables()
                )

            elif self.extension == ".csv":

                _, _, tables = (
                    self._extract_csv()
                )

                return tables

            elif self.extension == ".xlsx":

                _, _, tables = (
                    self._extract_xlsx()
                )

                return tables

            else:

                return []

        except Exception as e:

            return [{
                "error":
                    f"Table extraction failed: {e}"
            }]

    # ==========================================================
    # PDF Embedded Images
    # ==========================================================

    def _extract_pdf_images(self):

        images = []

        with fitz.open(
            self.file_path
        ) as doc:

            for page_num, page in enumerate(
                doc,
                start=1
            ):

                page_images = page.get_images(
                    full=True
                )

                image_info = []

                for index, img in enumerate(
                    page_images,
                    start=1
                ):

                    xref = img[0]

                    try:

                        image_data = (
                            doc.extract_image(
                                xref
                            )
                        )

                        image_bytes = (
                            image_data["image"]
                        )

                        image_ext = (
                            image_data["ext"]
                        )

                        image_filename = (
                            f"page_{page_num}_"
                            f"image_{index}."
                            f"{image_ext}"
                        )

                        image_path = os.path.join(
                            self.image_output_dir,
                            image_filename
                        )

                        with open(
                            image_path,
                            "wb"
                        ) as image_file:

                            image_file.write(
                                image_bytes
                            )

                        image_info.append({
                            "image_no": index,
                            "xref": xref,
                            "width": img[2],
                            "height": img[3],
                            "bits_per_component":
                                img[4],
                            "colorspace": img[5],
                            "format": image_ext,
                            "path": image_path
                        })

                    except Exception as image_error:

                        image_info.append({
                            "image_no": index,
                            "xref": xref,
                            "error":
                                str(image_error)
                        })

                images.append({
                    "page": page_num,
                    "count": len(page_images),
                    "images": image_info
                })

        return images

    # ==========================================================
    # DOCX Images
    # ==========================================================

    def _extract_docx_images(self):

        images = []

        with zipfile.ZipFile(
            self.file_path,
            "r"
        ) as archive:

            media_files = [
                name
                for name in archive.namelist()
                if name.startswith(
                    "word/media/"
                )
            ]

            for index, media_file in enumerate(
                media_files,
                start=1
            ):

                extension = os.path.splitext(
                    media_file
                )[1].lower()

                output_name = (
                    f"docx_image_{index}"
                    f"{extension}"
                )

                output_path = os.path.join(
                    self.image_output_dir,
                    output_name
                )

                with open(
                    output_path,
                    "wb"
                ) as output:

                    output.write(
                        archive.read(
                            media_file
                        )
                    )

                images.append({
                    "image_no": index,
                    "source": media_file,
                    "path": output_path
                })

        return [{
            "page": 1,
            "count": len(images),
            "images": images
        }]

    # ==========================================================
    # PPTX Images
    # ==========================================================

    def _extract_pptx_images(self):

        images = []

        with zipfile.ZipFile(
            self.file_path,
            "r"
        ) as archive:

            media_files = [
                name
                for name in archive.namelist()
                if name.startswith(
                    "ppt/media/"
                )
            ]

            for index, media_file in enumerate(
                media_files,
                start=1
            ):

                extension = os.path.splitext(
                    media_file
                )[1].lower()

                output_name = (
                    f"pptx_image_{index}"
                    f"{extension}"
                )

                output_path = os.path.join(
                    self.image_output_dir,
                    output_name
                )

                with open(
                    output_path,
                    "wb"
                ) as output:

                    output.write(
                        archive.read(
                            media_file
                        )
                    )

                images.append({
                    "image_no": index,
                    "source": media_file,
                    "path": output_path
                })

        return [{
            "page": 1,
            "count": len(images),
            "images": images
        }]

    # ==========================================================
    # Main Image Extraction
    # ==========================================================

    def extract_images(self):

        try:

            if self.extension == ".pdf":

                return (
                    self._extract_pdf_images()
                )

            elif self.extension == ".docx":

                return (
                    self._extract_docx_images()
                )

            elif self.extension == ".pptx":

                return (
                    self._extract_pptx_images()
                )

            else:

                return []

        except Exception as e:

            return [{
                "error":
                    f"Image extraction failed: {e}"
            }]

    # ==========================================================
    # PDF Visual / Flowchart / Diagram Extraction
    # ==========================================================

    def _extract_pdf_visuals(self):

        visuals = []

        with fitz.open(
            self.file_path
        ) as doc:

            for page_num, page in enumerate(
                doc,
                start=1
            ):

                embedded_image_count = len(
                    page.get_images(
                        full=True
                    )
                )

                try:

                    drawing_count = len(
                        page.get_drawings()
                    )

                except Exception:

                    drawing_count = 0

                # Render pages that contain either
                # embedded images or vector drawings.
                has_visual_content = (
                    embedded_image_count > 0
                    or drawing_count > 0
                )

                if not has_visual_content:
                    continue

                output_name = (
                    f"page_{page_num}_visual.png"
                )

                output_path = os.path.join(
                    self.visual_output_dir,
                    output_name
                )

                try:

                    matrix = fitz.Matrix(
                        2.0,
                        2.0
                    )

                    pixmap = page.get_pixmap(
                        matrix=matrix,
                        alpha=False
                    )

                    pixmap.save(
                        output_path
                    )

                    visuals.append({
                        "page": page_num,
                        "type":
                            "PDF rendered visual",
                        "path":
                            output_path,
                        "embedded_images":
                            embedded_image_count,
                        "drawings":
                            drawing_count
                    })

                except Exception as render_error:

                    visuals.append({
                        "page": page_num,
                        "type":
                            "PDF rendered visual",
                        "error":
                            str(render_error),
                        "embedded_images":
                            embedded_image_count,
                        "drawings":
                            drawing_count
                    })

        return visuals

    # ==========================================================
    # Main Visual Extraction
    # ==========================================================

    def extract_visuals(self):

        try:

            if self.extension == ".pdf":

                return (
                    self._extract_pdf_visuals()
                )

            return []

        except Exception as e:

            return [{
                "error":
                    f"Visual extraction failed: {e}"
            }]

    # ==========================================================
    # Statistics
    # ==========================================================

    def document_statistics(
        self,
        text,
        tables=None,
        images=None,
        visuals=None
    ):

        if tables is None:
            tables = self.extract_tables()

        if images is None:
            images = self.extract_images()

        if visuals is None:
            visuals = self.extract_visuals()

        total_images = 0

        for item in images:

            if isinstance(
                item,
                dict
            ):

                total_images += item.get(
                    "count",
                    0
                )

        real_tables = [
            table
            for table in tables
            if isinstance(
                table,
                dict
            )
            and "table" in table
        ]

        return {
            "words":
                len(text.split()),

            "characters":
                len(text),

            "lines":
                len(text.splitlines()),

            "tables":
                len(real_tables),

            "images":
                total_images,

            "visuals":
                len([
                    item
                    for item in visuals
                    if isinstance(
                        item,
                        dict
                    )
                    and "path" in item
                ])
        }

    # ==========================================================
    # Complete Document Extraction
    # ==========================================================

    def extract_document(self):

        text, pages = (
            self.extract_text()
        )

        tables = (
            self.extract_tables()
        )

        images = (
            self.extract_images()
        )

        visuals = (
            self.extract_visuals()
        )

        metadata = (
            self.extract_metadata()
        )

        statistics = (
            self.document_statistics(
                text=text,
                tables=tables,
                images=images,
                visuals=visuals
            )
        )

        return {
            "metadata": metadata,
            "statistics": statistics,
            "text": text,
            "pages": pages,
            "tables": tables,
            "images": images,
            "visuals": visuals
        }


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    file_path = "sample.pdf"

    try:

        extractor = DocumentExtractor(
            file_path
        )

        result = (
            extractor.extract_document()
        )

        print(
            "\n========== METADATA =========="
        )

        print(
            result["metadata"]
        )

        print(
            "\n========== STATISTICS =========="
        )

        print(
            result["statistics"]
        )

        print(
            "\n========== TABLES =========="
        )

        print(
            f"Tables Found: "
            f"{len(result['tables'])}"
        )

        print(
            "\n========== IMAGES =========="
        )

        total_images = sum(
            page.get(
                "count",
                0
            )
            for page in result["images"]
            if isinstance(
                page,
                dict
            )
        )

        print(
            f"Total Images: "
            f"{total_images}"
        )

        print(
            "\n========== VISUALS =========="
        )

        print(
            f"Visual Pages: "
            f"{len(result['visuals'])}"
        )

        print(
            "\n========== TEXT PREVIEW =========="
        )

        print(
            result["text"][:500]
        )

    except Exception as e:

        print(
            f"Error: {e}"
        )