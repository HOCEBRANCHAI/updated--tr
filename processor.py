# processor.py
# Library of robust extraction, OCR, FX, validation, mapping.
# Used by app.py. No server here.

import io
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta

# Money / math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

# OCR & PDF
import PyPDF2
import boto3
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import pytesseract

# LLM + HTTP
from openai import OpenAI
import requests

# Env
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("invoice-processor")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- Money helpers --------------------
CENT = Decimal("0.01")
RATE_PREC = Decimal("0.0001")

def q_money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

def q_rate(x) -> Decimal:
    return Decimal(str(x)).quantize(RATE_PREC, rounding=ROUND_HALF_UP)

def nearly_equal_money(a: Decimal, b: Decimal, tol: Decimal = CENT) -> bool:
    return abs(q_money(a) - q_money(b)) <= tol

# -------------------- Dates & currency --------------------
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def ensure_iso_date(s: Optional[str], field: str, errors: List[str]) -> Optional[date]:
    if not s or not ISO_DATE.match(s):
        errors.append(f"{field} must be YYYY-MM-DD (got {s!r}).")
        return None
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        errors.append(f"{field} is not a valid calendar date (got {s!r}).")
        return None

KNOWN_CURRENCIES = {
    "EUR","USD","GBP","INR","EGP","AED","SAR","CAD","AUD","NZD",
    "JPY","CNY","DKK","SEK","NOK","CHF","PLN","CZK","HUF"
}

def normalize_currency(cur: Optional[str], errors: List[str]) -> Optional[str]:
    cur = (cur or "").strip().upper()
    if cur not in KNOWN_CURRENCIES:
        errors.append(f"Unknown or missing currency {cur!r}.")
        return None
    return cur

def _prev_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d -= timedelta(days=1)
    return d

def get_eur_rate(invoice_date: date, ccy: str) -> Tuple[Decimal, str]:
    """
    Return (rate, rate_date_str) for 1 CCY -> EUR using exchangerate.host (ECB).
    Strategy:
      1) Try direct:  base=CCY&symbols=EUR
      2) Fallback:    base=EUR&symbols=CCY, then invert
    Look back up to 7 business days.
    """
    ccy = (ccy or "").upper().strip()
    if ccy == "EUR":
        return Decimal("1"), invoice_date.isoformat()

    d = _prev_business_day(invoice_date)
    for _ in range(7):
        # direct
        url1 = f"https://api.exchangerate.host/{d.isoformat()}?base={ccy}&symbols=EUR"
        try:
            r1 = requests.get(url1, timeout=8)
            js1 = r1.json() if r1.content else {}
            rate = (js1.get("rates") or {}).get("EUR")
            if r1.status_code == 200 and rate:
                return q_rate(rate), d.isoformat()
            log.warning(f"FX miss (direct) {url1} status={r1.status_code} body={str(js1)[:160]}")
        except Exception as ex:
            log.warning(f"FX direct failed {url1}: {ex}")

        # invert
        url2 = f"https://api.exchangerate.host/{d.isoformat()}?base=EUR&symbols={ccy}"
        try:
            r2 = requests.get(url2, timeout=8)
            js2 = r2.json() if r2.content else {}
            base_rate = (js2.get("rates") or {}).get(ccy)
            if r2.status_code == 200 and base_rate and float(base_rate) != 0.0:
                inv = Decimal("1") / Decimal(str(base_rate))
                return q_rate(inv), d.isoformat()
            log.warning(f"FX miss (invert) {url2} status={r2.status_code} body={str(js2)[:160]}")
        except Exception as ex2:
            log.warning(f"FX invert failed {url2}: {ex2}")

        d = _prev_business_day(d - timedelta(days=1))
    raise ValueError(f"No EUR rate found for {ccy} near {invoice_date.isoformat()}.")

# -------------------- OCR (robust, supports scanned PDFs) --------------------
def _preprocess_for_tesseract(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("L"))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    return Image.fromarray(img)

def _textract_analyze_image(img_bytes: bytes) -> str:
    textract = boto3.client('textract', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
    resp = textract.analyze_document(
        Document={'Bytes': img_bytes},
        FeatureTypes=['TABLES', 'FORMS']
    )
    blocks = resp.get("Blocks", [])
    text_lines = []
    block_map = {b["Id"]: b for b in blocks}

    for b in blocks:
        if b.get("BlockType") == "LINE" and b.get("Text"):
            text_lines.append(b["Text"])

    kv_pairs = []
    for b in blocks:
        if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in (b.get("EntityTypes") or []):
            key_words, val_words = [], []
            for rel in b.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for cid in rel.get("Ids", []):
                        w = block_map.get(cid)
                        if w and w.get("BlockType") == "WORD" and w.get("Text"):
                            key_words.append(w["Text"])
                if rel["Type"] == "VALUE":
                    for vid in rel.get("Ids", []):
                        v = block_map.get(vid)
                        if not v: continue
                        for rel2 in v.get("Relationships", []):
                            if rel2["Type"] == "CHILD":
                                for vcid in rel2.get("Ids", []):
                                    w = block_map.get(vcid)
                                    if w and w.get("BlockType") == "WORD" and w.get("Text"):
                                        val_words.append(w["Text"])
            k = " ".join(key_words).strip()
            v = " ".join(val_words).strip()
            if k or v:
                kv_pairs.append(f"{k}: {v}")

    combined = "\n".join(text_lines)
    if kv_pairs:
        combined += "\n--- Key-Value Pairs ---\n" + "\n".join(kv_pairs)
    return combined

def _tesseract_ocr(pil_img: Image.Image) -> str:
    pre = _preprocess_for_tesseract(pil_img)
    return pytesseract.image_to_string(pre, config="--psm 6")

def get_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    OCR strategy:
      1) PyPDF2 text
      2) Textract detect_document_text (whole PDF)
      3) PDF → images → Textract AnalyzeDocument (per page)
      4) PDF → images → Tesseract (last resort)
    """
    # 1) PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = "".join([(p.extract_text() or "") for p in reader.pages])
        if len(text.strip()) > 100:
            log.info(f"[PyPDF2] {filename}")
            return text
        log.warning(f"[PyPDF2] minimal for {filename}; try Textract detect.")
    except Exception as e:
        log.warning(f"[PyPDF2] failed for {filename}: {e}")

    # 2) Textract detect
    try:
        textract = boto3.client('textract', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        resp = textract.detect_document_text(Document={'Bytes': pdf_bytes})
        text = "\n".join([b.get("Text","") for b in resp.get("Blocks", []) if b.get("BlockType")=="LINE"])
        if len(text.strip()) > 40:
            log.info(f"[Textract.detect] {filename}")
            return text
        log.warning(f"[Textract.detect] minimal; try Analyze per page.")
    except Exception as e:
        log.warning(f"[Textract.detect] failed for {filename}: {e}; Analyze per page.")

    # 3) Textract Analyze per-page (images)
    try:
        images: List[Image.Image] = convert_from_bytes(pdf_bytes, dpi=300)
        texts = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            page_text = _textract_analyze_image(buf.getvalue())
            if page_text:
                texts.append(page_text)
        combined = "\n\n--- PAGE BREAK ---\n\n".join(texts)
        if len(combined.strip()) > 40:
            log.info(f"[Textract.analyze IMG] {filename}")
            return combined
        log.warning(f"[Textract.analyze IMG] minimal; trying Tesseract.")
    except Exception as e:
        log.warning(f"[Textract.analyze IMG] failed for {filename}: {e}; Tesseract fallback.")

    # 4) Tesseract fallback
    try:
        images: List[Image.Image] = convert_from_bytes(pdf_bytes, dpi=300)
        ocr = []
        for im in images:
            ocr.append(_tesseract_ocr(im))
        combined = "\n\n--- PAGE BREAK ---\n\n".join(ocr)
        if len(combined.strip()) > 10:
            log.info(f"[Tesseract] {filename}")
            return combined
    except Exception as e:
        log.error(f"[Tesseract] failed for {filename}: {e}")

    raise ValueError(
        f"PDF text extraction failed for {filename}: "
        f"PyPDF2, Textract detect, Textract analyze (per image), and Tesseract all returned minimal text."
    )

# -------------------- LLM extraction --------------------
SECTION_LABELS = [
    "invoice", "total", "subtotal", "tax", "vat", "btw", "reverse charge",
    "verlegd", "omgekeerde heffing", "bill to", "payer", "customer", "vendor",
    "supplier", "line items", "description", "due", "payment terms", "amount"
]

def reduce_invoice_text(raw_text: str, window: int = 300) -> str:
    text = raw_text or ""
    text_low = text.lower()
    spans: List[Tuple[int, int]] = []
    for label in SECTION_LABELS:
        for m in re.finditer(re.escape(label), text_low):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            spans.append((start, end))
    if not spans:
        return text[:8000]
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 50:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    chunks = [text[s:e] for s, e in merged]
    reduced = "\n---\n".join(chunks)
    return reduced[:12000]

LLM_PROMPT = """
You are an expert, high-accuracy financial data extraction model. Your sole task is to extract structured data from the provided invoice text and respond with a single, minified JSON object. You must think like an accountant.

RULES:
1) JSON ONLY. Include all top-level keys even if null. Dates = YYYY-MM-DD. Numbers = floats (no symbols).
2) Use the invoice currency shown in the official TOTAL block; ignore "for reference" currencies.
3) subtotal = goods/services only; total_vat = all taxes; total_amount = subtotal + total_vat.
4) VAT category:
   - "Import-VAT" for import VAT.
   - "Reverse-Charge" if reverse-charge applies (e.g., verlegd, omgekeerde heffing).
   - "Standard" for normal % VAT charged.
   - "Zero-Rated" if 0% VAT and not reverse charge.
   - "Out-of-Scope" if outside tax scope (e.g., Article 44).
5) Line items:
   - Extract goods/services only; do NOT include taxes as a line item.
   - unit_price only if explicitly printed. Do not compute it.
6) VAT percentage:
   - If the invoice explicitly shows a single VAT rate (e.g., 21%), put that in vat_breakdown.rate.
   - If text states "VAT out of scope / Article 44 / reverse charge / 0%", use 0.0 in vat_breakdown.
   - If multiple rates exist, list them; total_vat must equal the sum of tax_amount.

SCHEMA:
{
  "invoice_number": "string | null",
  "invoice_date": "YYYY-MM-DD | null",
  "due_date": "YYYY-MM-DD | null",
  "vendor_name": "string | null",
  "vendor_vat_id": "string | null",
  "vendor_address": "string | null",
  "customer_name": "string | null",
  "customer_vat_id": "string | null",
  "customer_address": "string | null",
  "currency": "string | null",
  "vat_category": "string | null",
  "subtotal": "float | null",
  "total_amount": "float | null",
  "total_vat": "float | null",
  "vat_breakdown": [
    {"rate": "float | 'import'", "base_amount": "float | null", "tax_amount": "float"}
  ],
  "line_items": [
    {"description": "string", "quantity": "float | null", "unit_price": "float | null", "line_total": "float | null"}
  ],
  "payment_terms": "string | null"
}
"""

def structure_text_with_llm(invoice_text: str, filename: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    reduced = reduce_invoice_text(invoice_text)
    try:
        log.info(f"LLM extracting {filename}...")
        r = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": f"**INVOICE TEXT TO PARSE (reduced):**\n{reduced}"}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except json.JSONDecodeError as e:
        log.error(f"LLM JSON error {filename}: {e}")
        raise ValueError(f"Extraction failed for {filename}: malformed JSON.")
    except Exception as e:
        log.error(f"LLM API error {filename}: {e}")
        raise ValueError("Extraction failed: LLM API error.")

# -------------------- Validation & mapping --------------------
def validate_extraction(data: dict, filename: str) -> Tuple[date, str, Decimal, Decimal, Decimal]:
    errors: List[str] = []
    for f in ["invoice_number","invoice_date","vendor_name","customer_name",
              "vat_category","currency","subtotal","total_amount","total_vat"]:
        if data.get(f) in (None, ""):
            errors.append(f"Missing {f!r}.")

    inv_date = ensure_iso_date(data.get("invoice_date"), "invoice_date", errors)
    if data.get("due_date"):
        _ = ensure_iso_date(data.get("due_date"), "due_date", errors)

    currency = normalize_currency(data.get("currency"), errors)

    try:
        sub = q_money(data["subtotal"])
        vat = q_money(data["total_vat"])
        tot = q_money(data["total_amount"])
        if not nearly_equal_money(sub + vat, tot):
            errors.append(f"Subtotal({sub}) + VAT({vat}) != Total({tot}).")
    except Exception:
        errors.append("Invalid numeric values for subtotal/total_vat/total_amount.")

    lis = data.get("line_items") or []
    try:
        li_sum = sum((q_money(li.get("line_total", 0)) for li in lis if li.get("line_total") is not None),
                     Decimal("0.00"))
        if lis and not nearly_equal_money(li_sum, sub):
            # Warn only (telecom/discount patterns often don't sum neatly)
            log.warning(f"{filename}: line_items sum({li_sum}) != subtotal({sub}). Using header totals.")
    except Exception:
        log.warning(f"{filename}: invalid line_items; continuing with header totals.")

    if errors:
        msg = f"Validation failed for {filename}: {' | '.join(errors)}"
        log.error(msg)
        raise ValueError(msg)

    assert inv_date and currency
    return inv_date, currency, sub, vat, tot

def _normalize_company_name(name: str) -> str:
    if not name: return ""
    return re.sub(r"[\s\-_/.,()]+", " ", name.casefold()).strip()

def _split_company_list(raw: str) -> List[str]:
    if not raw: return []
    return [p.strip() for p in re.split(r"[,\n;]", raw) if p and p.strip()]

def _derive_vat_rate_percent(llm_data: Dict[str, Any]) -> Optional[float]:
    vcat = (llm_data.get("vat_category") or "").strip().lower()
    for v in (llm_data.get("vat_breakdown") or []):
        r = v.get("rate")
        if isinstance(r, (int, float)):
            try:
                return float(r)
            except Exception:
                pass
    if vcat in {"reverse-charge", "out-of-scope", "zero-rated"}:
        return 0.0
    return None

def _map_llm_output_to_register_entry(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    description = ""
    if llm_data.get("line_items"):
        description = llm_data["line_items"][0].get("description", "")

    return {
        "Date": llm_data.get("invoice_date"),
        "Invoice Number": llm_data.get("invoice_number"),
        "Vendor Name": llm_data.get("vendor_name"),
        "Customer Name": llm_data.get("customer_name"),
        "Type": "Unclassified",
        "VAT Category": llm_data.get("vat_category"),
        "VAT Rate (%)": _derive_vat_rate_percent(llm_data),   # NEW
        "Nett Amount": float(q_money(llm_data.get("subtotal") or 0.0)),
        "VAT Amount": float(q_money(llm_data.get("total_vat") or 0.0)),
        "Gross Amount": float(q_money(llm_data.get("total_amount") or 0.0)),
        "Currency": llm_data.get("currency"),
        "Description": description,
        "Full_Extraction_Data": llm_data
    }

def _classify_type(register_entry: Dict[str, Any], our_companies_list: List[str]) -> str:
    v = _normalize_company_name(register_entry.get("Vendor Name") or "")
    c = _normalize_company_name(register_entry.get("Customer Name") or "")
    ours = [_normalize_company_name(x) for x in our_companies_list]
    if any(o and o in c for o in ours): return "Purchase"
    if any(o and o in v for o in ours): return "Sales"
    if "dutch food solutions" in c or "mohamed soliman" in c: return "Purchase"
    if "dutch food solutions" in v: return "Sales"
    return "Unclassified"

def _convert_to_eur_fields(entry: dict) -> dict:
    try:
        ccy = (entry.get("Currency") or "").upper()
        inv_date_str = entry.get("Date")
        inv_dt = date.fromisoformat(inv_date_str)
        if ccy == "EUR":
            entry["FX Rate (ccy->EUR)"] = "1.0000"
            entry["FX Rate Date"] = inv_dt.isoformat()
            for k_src, k_dst in [("Nett Amount","Nett Amount (EUR)"),
                                 ("VAT Amount","VAT Amount (EUR)"),
                                 ("Gross Amount","Gross Amount (EUR)")]:
                entry[k_dst] = float(q_money(entry.get(k_src, 0)))
            return entry

        rate, used_date = get_eur_rate(inv_dt, ccy)
        entry["FX Rate (ccy->EUR)"] = str(rate)
        entry["FX Rate Date"] = used_date
        for k_src, k_dst in [("Nett Amount","Nett Amount (EUR)"),
                             ("VAT Amount","VAT Amount (EUR)"),
                             ("Gross Amount","Gross Amount (EUR)")]:
            entry[k_dst] = float(q_money(q_money(entry.get(k_src, 0)) * rate))
        return entry
    except Exception as ex:
        entry["FX Rate (ccy->EUR)"] = None
        entry["FX Rate Date"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Error"] = f"EUR conversion failed: {ex}"
        return entry

# -------------------- Main pipeline --------------------
def robust_invoice_processor(pdf_bytes: bytes, filename: str) -> dict:
    invoice_text = get_text_from_pdf(pdf_bytes, filename)
    llm_data = structure_text_with_llm(invoice_text, filename)
    _ = validate_extraction(llm_data, filename)
    return llm_data

# -------------------- Public API for app.py --------------------
__all__ = [
    "robust_invoice_processor",
    "_map_llm_output_to_register_entry",
    "_classify_type",
    "_split_company_list",
    "_convert_to_eur_fields",
    "q_money",
]
