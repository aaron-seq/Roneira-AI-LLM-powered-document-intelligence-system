# Sample documents

A small synthetic corpus so you can see the whole workflow before uploading
anything of your own. Load it with:

```bash
python scripts/load_samples.py
```

Everything here is **fabricated**. The companies, people, amounts and policies
do not exist. Nothing in this directory contains real personal or commercial
data.

---

## What is in here

### Invoices — `INV-2025-1001.pdf` … `INV-2025-1010.pdf`

Ten single-page invoices from different fictional vendors, each with a vendor
address, invoice number, dates, line items and totals. Useful for testing
field extraction and numeric questions.

Questions worth asking once they are indexed:

- *"Which invoices are due in January 2026?"*
- *"What is the total amount owed to Quantum Industrial Systems?"*
- *"Which vendor is based in New York?"*

### HR policies — `hr_2024_001.pdf` … `hr_2024_010.pdf`

Ten two-page policy documents from fictional companies across different
industries — PTO, remote work, and similar. Longer prose, so they exercise
chunking and multi-page citation.

- *"What is the remote work policy at Cascade Manufacturing?"*
- *"How many PTO days do employees get in their first year?"*
- *"Which policies were last updated in 2024?"*

### Long-form text — `case_study_nextgen_mfg.txt`, `eng_specs_nextgen_robotics.txt`

Plain-text case study and engineering specification. Good for testing
summarization and for questions that span several passages.

---

## A note on what these will and will not show you

With the default install, search matches **keywords**. Asking *"how much do we
owe?"* will not find *"total amount due"* until you install the embedding
model:

```bash
pip install sentence-transformers
```

That difference is the clearest way to see what semantic retrieval buys you:
run the same question before and after.

Similarly, questions are answered with citations but **summaries and prose
answers need Ollama running**. Without it you still get search results with
page references; you just do not get an answer written in sentences.

---

## Adding your own

Drop files into this directory and re-run the loader. Supported formats are
listed by `GET /api/documents/formats/supported`.

Scanned pages and images are read with OCR, which needs the **tesseract**
binary — a system package, not a pip install:

```bash
sudo apt-get install tesseract-ocr     # or: brew install tesseract
```

Without it, a scan is refused with that reason rather than indexed empty, and
the document's metadata carries `ocr_unavailable_reason`.

---

## A note on the invoice text layer

The ten invoice PDFs were generated with markup that reached their text layer:
`<b>TOTAL DUE: $8385.00</b>` is literally in the page. Extraction strips a
closed list of inline formatting tags (`b`, `i`, `u`, `em`, `strong`, `br`), so
you will not see it in summaries, search results or exports — but it is in the
files, and it is why that stripping exists. A document *about* markup keeps its
content: only those six tags are removed.
