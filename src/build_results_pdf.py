"""Render results/results.md into results/results.pdf with figures embedded.

Reproducible packaging step (not part of the eval pipeline). Converts the
results.md markdown to styled HTML (figures are embedded inline in the source)
and prints to PDF with headless Chrome. Re-run with:

    python -m src.build_results_pdf

Dependencies: the `markdown` package (falls back to /tmp/pdflibs if not in the
venv) and Google Chrome. Verifies the output by re-extracting text + images.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import markdown
except ImportError:  # convenience fallback for an out-of-venv install
    sys.path.insert(0, "/tmp/pdflibs")
    import markdown

from . import config

OUT = config.ROOT / "results"
SRC = OUT / "results.md"
PDF = OUT / "results.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 0.8in 0.9in; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 22pt; color: #111; border-bottom: 3px solid #2563eb; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #1e3a8a; margin-top: 26px; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
h3 { font-size: 12.5pt; color: #111; margin-top: 20px; page-break-after: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto 18px;
      border: 1px solid #e5e7eb; border-radius: 4px; page-break-inside: avoid; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt; color: #b91c1c; }
pre { background: #f3f4f6; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 9pt;
      page-break-inside: avoid; }
pre code { background: none; color: #1a1a1a; padding: 0; }
blockquote { border-left: 4px solid #f59e0b; background: #fffbeb; margin: 10px 0; padding: 8px 14px;
             color: #444; font-style: italic; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #d1d5db; padding: 5px 8px; text-align: left; }
th { background: #eff6ff; }
strong { color: #111; }
ul, ol { margin: 6px 0; }
"""


def embed_figures(md_text: str) -> str:
    """After any heading line that names a `figures/*.png` path, embed that image."""
    out_lines = []
    for line in md_text.splitlines():
        out_lines.append(line)
        if line.startswith("#"):
            m = re.search(r"`(figures/[^`]+\.png)`", line)
            if m:
                out_lines.append("")
                out_lines.append(f"![]({m.group(1)})")
    return "\n".join(out_lines)


def build():
    md_text = embed_figures(SRC.read_text())
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    # write the HTML inside results/ so relative figure paths resolve
    html_path = OUT / "_build.html"
    html_path.write_text(html)
    profile = tempfile.mkdtemp(prefix="chrome-pdf-")
    before = PDF.stat().st_mtime if PDF.exists() else 0
    proc = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
        f"--print-to-pdf={PDF}", f"file://{html_path}",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # headless Chrome often writes the PDF but does not self-exit; don't block on it
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        html_path.unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)
    if not (PDF.exists() and PDF.stat().st_mtime > before):
        raise SystemExit("ERROR: Chrome did not produce a new PDF")

    # verify
    sys.path.insert(0, "/tmp/pdflibs")
    import pypdf
    r = pypdf.PdfReader(str(PDF))
    full_text = "\n".join(p.extract_text() for p in r.pages)
    imgs = sum(len(p.images) for p in r.pages)
    has_section = "Consolidated response-mode table" in full_text and "Response mode on the XSafety baseline" in full_text
    print(f"built {PDF}")
    print(f"  pages: {len(r.pages)} | embedded image tiles: {imgs}")
    print(f"  contains headline + response-data sections: {has_section}")
    if not has_section:
        raise SystemExit("ERROR: expected sections not found in rendered PDF — build did not pick up results.md")


if __name__ == "__main__":
    build()
