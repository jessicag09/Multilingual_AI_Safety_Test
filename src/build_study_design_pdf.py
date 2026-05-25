"""Render README.md into STUDY_DESIGN.pdf with styled tables.

Re-run with:

    python -m src.build_study_design_pdf
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.path.insert(0, "/tmp/pdflibs")
    import markdown

from . import config

SRC = config.ROOT / "README.md"
PDF = config.ROOT / "STUDY_DESIGN.pdf"
HTML = config.ROOT / "_study_design_build.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 0.75in 0.85in; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.8pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 21pt; color: #111; border-bottom: 3px solid #2563eb; padding-bottom: 6px; }
h2 { font-size: 14.5pt; color: #1e3a8a; margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
h3 { font-size: 12pt; color: #111; margin-top: 18px; }
h4 { font-size: 11pt; color: #111; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 8.8pt; }
th, td { border: 1px solid #d1d5db; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #eff6ff; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; color: #b91c1c; }
pre { background: #f3f4f6; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 9pt; }
pre code { background: none; color: #1a1a1a; padding: 0; }
ul, ol { margin: 6px 0; }
strong { color: #111; }
"""


def build():
    md_text = SRC.read_text()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML.write_text(html)
    profile = tempfile.mkdtemp(prefix="chrome-pdf-")
    before = PDF.stat().st_mtime if PDF.exists() else 0
    proc = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
        f"--print-to-pdf={PDF}", f"file://{HTML}",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        HTML.unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)
    if not (PDF.exists() and PDF.stat().st_mtime > before):
        raise SystemExit("ERROR: Chrome did not produce a new study design PDF")
    sys.path.insert(0, "/tmp/pdflibs")
    import pypdf
    r = pypdf.PdfReader(str(PDF))
    text = "\n".join(p.extract_text() or "" for p in r.pages)
    has_sections = "Metric glossary" in text and "Prompt inventory" in text and "Research questions" in text
    print(f"built {PDF}")
    print(f"  pages: {len(r.pages)}")
    print(f"  contains expected sections: {has_sections}")
    if not has_sections:
        raise SystemExit("ERROR: expected sections not found in study design PDF")


if __name__ == "__main__":
    build()
