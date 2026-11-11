"""Extractors for supported submission formats."""

from .pdf_extractor import extract_pdf
from .docx_extractor import extract_docx
from .mp4_extractor import extract_mp4

EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".doc": extract_docx,
    ".mp4": extract_mp4,
    ".mov": extract_mp4,
    ".mkv": extract_mp4,
    ".avi": extract_mp4,
    ".wav": extract_mp4,  # bare audio still goes through the ASR path
    ".mp3": extract_mp4,
}


def supported_extensions():
    return sorted(EXTRACTORS.keys())
