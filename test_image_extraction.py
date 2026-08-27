"""Focused test for image (OCR) extraction in app.extract_document."""

import io
import sys
import types
from pathlib import Path

from PIL import Image, ImageDraw


MARKER = "# MAIN APPLICATION"


def load_app_module():
    source = Path("app.py").read_text(encoding="utf-8")

    index = source.find(MARKER)

    if index != -1:
        source = source[:index]

    module = types.ModuleType("app_under_test")
    module.__file__ = "app.py"

    sys.modules["app_under_test"] = module

    exec(compile(source, "app.py", "exec"), module.__dict__)  # noqa: S102

    return module


def build_image(extension, text):
    image = Image.new("RGB", (900, 220), "white")

    draw = ImageDraw.Draw(image)
    draw.text((20, 90), text, fill="black")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format={
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".bmp": "BMP",
            ".tif": "TIFF",
            ".tiff": "TIFF",
            ".webp": "WEBP",
        }[extension],
    )

    return buffer.getvalue()


def main():
    app = load_app_module()

    failures = []

    for extension in sorted(app.IMAGE_EXTENSIONS):
        if extension not in app.SUPPORTED_EXTENSIONS:
            failures.append(
                f"{extension} is missing from SUPPORTED_EXTENSIONS."
            )

    text = "INVOICE TOTAL 4210"

    for extension in sorted(app.IMAGE_EXTENSIONS):
        image_bytes = build_image(extension, text)

        extracted = app.extract_document(
            f"scan{extension}",
            image_bytes,
        )

        if not extracted:
            failures.append(f"{extension}: extraction returned no sections.")
            continue

        section = extracted[0]

        if section.get("chunk_type") != "Image":
            failures.append(f"{extension}: chunk_type is not 'Image'.")

        if section.get("page") != 1:
            failures.append(f"{extension}: page is not 1.")

        if "INVOICE" not in str(section.get("content", "")).upper():
            failures.append(
                f"{extension}: OCR text did not contain the expected words "
                f"(got {section.get('content')!r})."
            )

    blank = build_image(".png", "")

    if app.extract_document("blank.png", blank):
        failures.append("A blank image should extract zero sections.")

    chunks = app.chunk_document(
        app.extract_document("scan.png", build_image(".png", text)),
        "Fixed Size",
        1000,
        100,
        "scan.png",
    )

    if not chunks:
        failures.append("Chunking an image extraction produced no chunks.")

    elif chunks[0].get("chunk_type") != "Image":
        failures.append("Image chunks lost their 'Image' chunk type.")

    if failures:
        print("FAILED")

        for failure in failures:
            print(" -", failure)

        return 1

    print("PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
