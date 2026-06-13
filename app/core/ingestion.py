"""
Document loaders for PDF, URL, and plain text.
Returns list of {text, metadata} dicts.
"""
import io
import re
from pathlib import Path

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup


def load_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text page by page from a PDF."""
    pages = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "source_type": "pdf"
                }
            })
    doc.close()
    return pages


def load_url(url: str) -> list[dict]:
    """Fetch and parse a URL, returning cleaned text."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Bot/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav, footer, scripts
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Try to get article content first, fallback to body
    article = soup.find("article") or soup.find("main") or soup.find("body")
    text = article.get_text(separator="\n") if article else soup.get_text(separator="\n")

    # Clean whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return [{
        "text": text,
        "metadata": {
            "source": url,
            "page": 1,
            "source_type": "url",
            "title": soup.title.string if soup.title else url
        }
    }]


def load_text(text: str, filename: str = "plain_text") -> list[dict]:
    """Wrap plain text as a single document."""
    return [{
        "text": text.strip(),
        "metadata": {
            "source": filename,
            "page": 1,
            "source_type": "text"
        }
    }]
