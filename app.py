import logging
import os
import re
import json
import io
from typing import List, Dict, Any, Optional
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta
import requests

# PDF to Image conversion (for scanned PDFs)
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. Image-based PDF fallback will be disabled.")

# --- FastAPI Imports ---
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- PDF & OCR Imports ---
import PyPDF2
import boto3  # For AWS Textract (OCR)

# --- LLM Imports ---
from openai import OpenAI

# --- Standard Imports ---
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Configuration Flags
# ----------------------------------------------------------------------------
conversion_enabled: bool = True  # Toggle currency conversion on/off

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
# Added proper logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------------
# Robust LLM Prompt (The new "Processor")
# ----------------------------------------------------------------------------

LLM_PROMPT = """
You are an expert, high-accuracy financial data extraction model. Your sole task is to extract structured data from the provided invoice text and respond with a single, minified JSON object.

**CRITICAL RULES:**
1.  **JSON ONLY:** Respond with NOTHING but a single, minified JSON object. Do not add "Here is the JSON..." or any other text.
2.  **STRICT SCHEMA:** Adhere strictly to the JSON schema defined below.
3.  **ALL FIELDS:** Include all top-level keys from the schema, even if the value is `null`.
4.  **DATES:** All dates MUST be in "YYYY-MM-DD" format.
5.  **NUMBERS:** All monetary values MUST be a standard float (e.g., 123.45). Do not include currency symbols or commas.
6.  **VAT/TAX:** The `vat_breakdown` array is for summarizing tax rates. If the invoice shows multiple VAT rates, create an object for each.
7.  **LINE ITEMS:** Extract all individual line items. If no line items are obvious, create a single line item using the invoice's main description.
8.  **SANITY CHECK:** Before outputting, ensure that `subtotal` + the sum of `vat_breakdown.amount` is approximately equal to the `total_amount`.

**JSON SCHEMA TO FOLLOW:**
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
  "currency": "string (e.g., 'EUR', 'USD', 'EGP') | null",
  "subtotal": "float | null",
  "total_amount": "float | null",
  "total_vat": "float | null",
  "vat_breakdown": [
    {
      "rate": "float (e.g., 21.0, 14.0, 0.0)",
      "base_amount": "float (amount the tax is applied to)",
      "tax_amount": "float (the resulting tax amount)"
    }
  ],
  "line_items": [
    {
      "description": "string",
      "quantity": "float | null",
      "unit_price": "float | null",
      "line_total": "float | null"
    }
  ],
  "payment_terms": "string | null"
}
"""

# ----------------------------------------------------------------------------
# Robust Extraction Pipeline (Replaces processor.py)
# ----------------------------------------------------------------------------

def _extract_basic_financial_data_from_text(text: str) -> dict:
    """
    Fallback: Extract basic financial data (gross amount, VAT amount, VAT percentage) 
    from minimal text using regex patterns. Used when full extraction fails.
    """
    if not text or len(text.strip()) < 10:
        return None
    
    result = {
        "total_amount": None,
        "total_vat": None,
        "vat_rate": None,
        "subtotal": None
    }
    
    # Normalize text for pattern matching
    text_upper = text.upper()
    text_clean = re.sub(r'[^\d\s.,€$£¥%:()\-]', ' ', text)
    
    # Pattern 1: Find "Total" or "Totaal" followed by amount
    total_patterns = [
        r'(?:total|totaal|totaalbedrag|totaal\s+bedrag)[:\s]*([\d.,]+)',
        r'([\d.,]+)\s*(?:€|EUR|euro)',
        r'total[:\s]*€?\s*([\d.,]+)',
    ]
    
    for pattern in total_patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        if matches:
            try:
                # Take the largest number found (likely the total)
                amounts = [float(m.replace(',', '.').replace(' ', '')) for m in matches if m.replace(',', '').replace('.', '').isdigit()]
                if amounts:
                    result["total_amount"] = max(amounts)
                    break
            except:
                continue
    
    # Pattern 2: Find VAT/BTW amount
    vat_patterns = [
        r'(?:btw|vat|belasting|tax)[:\s]*([\d.,]+)',
        r'btw[:\s]*€?\s*([\d.,]+)',
        r'vat[:\s]*€?\s*([\d.,]+)',
    ]
    
    for pattern in vat_patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        if matches:
            try:
                vat_amounts = [float(m.replace(',', '.').replace(' ', '')) for m in matches if m.replace(',', '').replace('.', '').isdigit()]
                if vat_amounts:
                    result["total_vat"] = max(vat_amounts)
                    break
            except:
                continue
    
    # Pattern 3: Find VAT percentage (21%, 0%, etc.)
    vat_rate_patterns = [
        r'(?:btw|vat)[:\s]*(\d+(?:[.,]\d+)?)\s*%',
        r'(\d+(?:[.,]\d+)?)\s*%\s*(?:btw|vat)',
        r'btw\s*(\d+(?:[.,]\d+)?)',
    ]
    
    for pattern in vat_rate_patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        if matches:
            try:
                rates = [float(m.replace(',', '.')) for m in matches]
                if rates:
                    result["vat_rate"] = max(rates)
                    break
            except:
                continue
    
    # Calculate subtotal if we have total and VAT
    if result["total_amount"] and result["total_vat"]:
        result["subtotal"] = result["total_amount"] - result["total_vat"]
    
    # Return result if we found at least one field
    if any(result.values()):
        return result
    return None

def get_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Step 1: Extracts text from a PDF with multiple fallback methods.
    Tries: PyPDF2 -> Textract detect_document_text -> Textract analyze_document
    Returns text and logs which extraction method was used.
    """
    # Validate PDF first
    if not pdf_bytes or len(pdf_bytes) < 100:
        raise ValueError(f"Invalid PDF file {filename}: File is too small or empty.")
    
    # Check if it's actually a PDF (starts with %PDF)
    if not pdf_bytes[:4].startswith(b'%PDF'):
        raise ValueError(f"Invalid PDF file {filename}: File does not appear to be a valid PDF format.")
    
    # Check for password protection
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            logging.warning(f"PDF {filename} appears to be password-protected. Attempting to extract anyway...")
            try:
                reader.decrypt("")  # Try empty password
            except:
                raise ValueError(f"PDF {filename} is password-protected. Please remove password protection first.")
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError(f"PDF {filename} is password-protected. Please remove password protection first.")
    
    # 1. Try PyPDF2 first (fastest, cheapest)
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        if len(text.strip()) > 100:
            extraction_method = "PyPDF2"
            logging.info(f"[{extraction_method}] Successfully extracted text from {filename}.")
            return text
            
        logging.warning(f"PyPDF2 found minimal text in {filename}. Falling back to Textract detect_document_text.")
            
    except Exception as e:
        logging.warning(f"PyPDF2 failed for {filename}: {e}. Falling back to Textract.")

    # 2. Fallback to Textract detect_document_text
    try:
        logging.info(f"Extracting text from {filename} using AWS Textract detect_document_text...")
        
        # Check if AWS is configured
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key or not aws_secret_key:
            logging.warning(f"AWS credentials not configured. Skipping Textract for {filename}.")
            raise ValueError("AWS credentials not configured")
        
        try:
            textract = boto3.client('textract', region_name=aws_region)
        except Exception as cred_error:
            logging.warning(f"AWS Textract client initialization failed: {cred_error}")
            raise ValueError("AWS Textract client initialization failed")
        
        # Check file size (Textract has limits)
        if len(pdf_bytes) > 500 * 1024 * 1024:  # 500MB limit
            raise ValueError(f"PDF {filename} is too large ({len(pdf_bytes) / 1024 / 1024:.1f}MB). Maximum size is 500MB.")
        
        response = textract.detect_document_text(
            Document={'Bytes': pdf_bytes}
        )
        
        text = ""
        blocks = response.get("Blocks", [])
        for item in blocks:
            if item["BlockType"] == "LINE":
                line_text = item.get("Text", "")
                if line_text:
                    text += line_text + "\n"
        
        # Also extract WORD blocks if LINE blocks are insufficient
        if len(text.strip()) < 20:
            for item in blocks:
                if item["BlockType"] == "WORD":
                    word_text = item.get("Text", "")
                    if word_text:
                        text += word_text + " "
        
        text = text.strip()
        
        if len(text) > 20:
            extraction_method = "Textract_detect_document_text"
            logging.info(f"[{extraction_method}] Successfully extracted text from {filename} ({len(text)} chars).")
            return text
        else:
            logging.warning(f"Textract detect_document_text returned minimal text ({len(text)} chars). Trying analyze_document...")
            
    except ValueError as ve:
        # Re-raise our intentional ValueErrors
        if "AWS" in str(ve) or "credentials" in str(ve).lower():
            logging.warning(f"AWS Textract not available: {ve}. Trying analyze_document...")
        else:
            raise ve
    except Exception as e:
        logging.warning(f"AWS Textract detect_document_text failed for {filename}: {e}. Trying analyze_document...")

    # 3. Final fallback: Textract analyze_document (for scanned PDFs with tables/forms)
    try:
        logging.info(f"Extracting text from {filename} using AWS Textract analyze_document (TABLES, FORMS)...")
        
        # Check AWS credentials
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key or not aws_secret_key:
            logging.error(f"AWS credentials not configured for {filename}")
            raise ValueError(f"AWS Textract not configured. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION environment variables.")
        
        try:
            textract = boto3.client('textract', region_name=aws_region)
        except Exception as cred_error:
            logging.error(f"AWS Textract client initialization failed: {cred_error}")
            raise ValueError(f"AWS Textract client initialization failed: {cred_error}")
        
        # Check file size (Textract has limits)
        if len(pdf_bytes) > 500 * 1024 * 1024:  # 500MB limit
            raise ValueError(f"PDF {filename} is too large ({len(pdf_bytes) / 1024 / 1024:.1f}MB). Maximum size is 500MB.")
        
        response = textract.analyze_document(
            Document={'Bytes': pdf_bytes},
            FeatureTypes=['TABLES', 'FORMS']
        )
        
        # Extract text from all block types
        text = ""
        key_value_pairs = {}
        
        # Process blocks to extract text and key-value pairs
        blocks = response.get("Blocks", [])
        if not blocks:
            logging.warning(f"Textract analyze_document returned no blocks for {filename}")
            raise ValueError("Textract analyze_document returned no blocks")
        
        block_map = {block["Id"]: block for block in blocks}
        
        # Extract all text from different block types
        for block in blocks:
            block_type = block.get("BlockType")
            
            # Extract LINE blocks (complete lines of text)
            if block_type == "LINE":
                line_text = block.get("Text", "")
                if line_text:
                    text += line_text + "\n"
            
            # Extract WORD blocks directly (fallback if LINE not available)
            elif block_type == "WORD":
                word_text = block.get("Text", "")
                if word_text:
                    text += word_text + " "
            
            # Extract CELL blocks from tables
            elif block_type == "CELL":
                cell_text = ""
                for relationship in block.get("Relationships", []):
                    if relationship["Type"] == "CHILD":
                        for child_id in relationship["Ids"]:
                            child_block = block_map.get(child_id)
                            if child_block:
                                if child_block.get("BlockType") == "WORD":
                                    cell_text += child_block.get("Text", "") + " "
                                elif child_block.get("BlockType") == "LINE":
                                    cell_text += child_block.get("Text", "") + " "
                if cell_text.strip():
                    text += cell_text.strip() + "\n"
            
            # Extract KEY_VALUE_SET blocks (forms)
            elif block_type == "KEY_VALUE_SET":
                entity_type = block.get("EntityTypes", [])
                if "KEY" in entity_type:
                    # Extract key text
                    key_text = ""
                    for relationship in block.get("Relationships", []):
                        if relationship["Type"] == "CHILD":
                            for child_id in relationship["Ids"]:
                                child_block = block_map.get(child_id)
                                if child_block:
                                    if child_block.get("BlockType") == "WORD":
                                        key_text += child_block.get("Text", "") + " "
                                    elif child_block.get("BlockType") == "LINE":
                                        key_text += child_block.get("Text", "") + " "
                    key_text = key_text.strip()
                    
                    # Find the corresponding value
                    for relationship in block.get("Relationships", []):
                        if relationship["Type"] == "VALUE":
                            for value_id in relationship["Ids"]:
                                value_block = block_map.get(value_id)
                                if value_block:
                                    value_text = ""
                                    for rel in value_block.get("Relationships", []):
                                        if rel["Type"] == "CHILD":
                                            for child_id in rel["Ids"]:
                                                child_block = block_map.get(child_id)
                                                if child_block:
                                                    if child_block.get("BlockType") == "WORD":
                                                        value_text += child_block.get("Text", "") + " "
                                                    elif child_block.get("BlockType") == "LINE":
                                                        value_text += child_block.get("Text", "") + " "
                                    if value_text.strip():
                                        key_value_pairs[key_text] = value_text.strip()
        
        # Clean up text (preserve newlines, normalize spaces)
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\n')
        cleaned_lines = [re.sub(r' +', ' ', line).strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        # Remove completely empty lines
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = text.strip()
        
        # Combine extracted text and key-value pairs
        combined_text = text
        if key_value_pairs:
            combined_text += "\n\n--- Key-Value Pairs ---\n"
            for key, value in key_value_pairs.items():
                combined_text += f"{key}: {value}\n"
        
        # Log extraction details for debugging
        logging.info(f"Textract analyze_document extracted {len(text)} characters, {len(key_value_pairs)} key-value pairs from {filename}")
        
        if len(combined_text.strip()) > 20:
            extraction_method = "Textract_analyze_document"
            logging.info(f"[{extraction_method}] Successfully extracted text from {filename} ({len(combined_text)} chars).")
            return combined_text
        else:
            logging.warning(f"Textract analyze_document returned insufficient text ({len(combined_text)} chars) for {filename}")
            raise ValueError("Textract analyze_document returned no significant text.")
            
    except ValueError as ve:
        # Re-raise ValueError as-is (these are our intentional errors)
        raise ve
    except Exception as e:
        error_details = str(e)
        logging.error(f"AWS Textract analyze_document failed for {filename}: {error_details}")
        logging.error(f"Error type: {type(e).__name__}")
        
        # Provide more helpful error messages
        if "UnsupportedDocumentException" in error_details:
            # Try extracting images from PDF and processing those
            if PDF2IMAGE_AVAILABLE:
                logging.info(f"Textract rejected PDF format. Attempting to extract images from {filename} and process them...")
                try:
                    # Re-create textract client in case it was lost in exception
                    textract_client = boto3.client('textract', region_name=aws_region)
                    return _extract_text_from_pdf_images(pdf_bytes, filename, textract_client, aws_region)
                except Exception as img_error:
                    logging.warning(f"Image extraction fallback failed for {filename}: {img_error}")
                    # Continue to minimal extraction fallback
            
            # Last resort: Try to get ANY text and extract basic financial data
            logging.info(f"Attempting basic text extraction from {filename} for minimal data extraction...")
            minimal_text = _try_minimal_text_extraction(pdf_bytes, filename)
            if minimal_text and len(minimal_text.strip()) > 10:
                logging.info(f"Extracted minimal text ({len(minimal_text)} chars) from {filename}. Will attempt basic financial data extraction.")
                return minimal_text
            
            raise ValueError(
                f"PDF {filename} is in an unsupported format for AWS Textract. "
                f"Could not extract any usable text. "
                f"Please try: (1) Re-scanning the document at higher quality, "
                f"(2) Converting to a standard PDF format, or (3) Using a different PDF file."
            )
        elif "InvalidParameterException" in error_details or "InvalidS3ObjectException" in error_details:
            raise ValueError(f"Invalid PDF file format for {filename}. Please ensure the file is a valid PDF.")
        elif "AccessDeniedException" in error_details or "UnauthorizedOperation" in error_details:
            raise ValueError(f"AWS Textract access denied. Please check AWS credentials and permissions for {filename}.")
        elif "ThrottlingException" in error_details:
            raise ValueError(f"AWS Textract rate limit exceeded. Please try again later for {filename}.")
        elif "DocumentTooLargeException" in error_details:
            raise ValueError(f"PDF {filename} is too large for AWS Textract. Maximum size is 500 pages or 500MB.")
        elif "ProvisionedThroughputExceededException" in error_details:
            raise ValueError(f"AWS Textract throughput limit exceeded. Please try again later for {filename}.")
        else:
            raise ValueError(f"PDF text extraction failed for {filename}: {error_details}. Check file format and AWS configuration.")

def _extract_text_from_pdf_images(pdf_bytes: bytes, filename: str, textract_client, aws_region: str) -> str:
    """
    Fallback method: Extract images from PDF and send each page image to Textract.
    This works for scanned PDFs that Textract rejects in PDF format.
    """
    try:
        logging.info(f"Converting PDF {filename} to images for Textract processing...")
        # Convert PDF pages to images (300 DPI for good quality)
        images = convert_from_bytes(pdf_bytes, dpi=300)
        
        if not images:
            raise ValueError("No images could be extracted from PDF")
        
        all_text = ""
        
        for page_num, pil_image in enumerate(images, 1):
            logging.info(f"Processing page {page_num} of {filename} as image...")
            
            # Convert PIL Image to bytes (PNG format)
            img_bytes_io = io.BytesIO()
            pil_image.save(img_bytes_io, format='PNG')
            img_bytes = img_bytes_io.getvalue()
            
            # Send image to Textract analyze_document
            try:
                response = textract_client.analyze_document(
                    Document={'Bytes': img_bytes},
                    FeatureTypes=['TABLES', 'FORMS']
                )
                
                # Extract text from blocks
                blocks = response.get("Blocks", [])
                block_map = {block["Id"]: block for block in blocks}
                
                page_text = ""
                key_value_pairs = {}
                
                for block in blocks:
                    block_type = block.get("BlockType")
                    
                    if block_type == "LINE":
                        page_text += block.get("Text", "") + "\n"
                    elif block_type == "WORD":
                        page_text += block.get("Text", "") + " "
                    elif block_type == "CELL":
                        cell_text = ""
                        for relationship in block.get("Relationships", []):
                            if relationship["Type"] == "CHILD":
                                for child_id in relationship["Ids"]:
                                    child_block = block_map.get(child_id)
                                    if child_block:
                                        if child_block.get("BlockType") == "WORD":
                                            cell_text += child_block.get("Text", "") + " "
                                        elif child_block.get("BlockType") == "LINE":
                                            cell_text += child_block.get("Text", "") + " "
                        if cell_text.strip():
                            page_text += cell_text.strip() + "\n"
                    elif block_type == "KEY_VALUE_SET":
                        entity_type = block.get("EntityTypes", [])
                        if "KEY" in entity_type:
                            key_text = ""
                            for relationship in block.get("Relationships", []):
                                if relationship["Type"] == "CHILD":
                                    for child_id in relationship["Ids"]:
                                        child_block = block_map.get(child_id)
                                        if child_block:
                                            if child_block.get("BlockType") == "WORD":
                                                key_text += child_block.get("Text", "") + " "
                                            elif child_block.get("BlockType") == "LINE":
                                                key_text += child_block.get("Text", "") + " "
                            key_text = key_text.strip()
                            
                            for relationship in block.get("Relationships", []):
                                if relationship["Type"] == "VALUE":
                                    for value_id in relationship["Ids"]:
                                        value_block = block_map.get(value_id)
                                        if value_block:
                                            value_text = ""
                                            for rel in value_block.get("Relationships", []):
                                                if rel["Type"] == "CHILD":
                                                    for child_id in rel["Ids"]:
                                                        child_block = block_map.get(child_id)
                                                        if child_block:
                                                            if child_block.get("BlockType") == "WORD":
                                                                value_text += child_block.get("Text", "") + " "
                                                            elif child_block.get("BlockType") == "LINE":
                                                                value_text += child_block.get("Text", "") + " "
                                            if value_text.strip():
                                                key_value_pairs[key_text] = value_text.strip()
                
                # Clean up page text
                page_text = page_text.strip()
                if key_value_pairs:
                    page_text += "\n\n--- Key-Value Pairs ---\n"
                    for key, value in key_value_pairs.items():
                        page_text += f"{key}: {value}\n"
                
                all_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                
            except Exception as page_error:
                logging.warning(f"Failed to process page {page_num} of {filename}: {page_error}")
                continue
        
        if len(all_text.strip()) > 20:
            logging.info(f"[Textract_image_fallback] Successfully extracted text from {filename} ({len(all_text)} chars from {len(images)} pages).")
            return all_text.strip()
        else:
            raise ValueError("Image extraction returned insufficient text")
            
    except Exception as e:
        logging.error(f"PDF image extraction fallback failed for {filename}: {e}")
        raise ValueError(f"Failed to extract text from PDF images: {str(e)}")

def _try_minimal_text_extraction(pdf_bytes: bytes, filename: str) -> str:
    """
    Last resort: Try to extract ANY text from PDF using PyPDF2, even if minimal.
    Returns whatever text we can get, even if it's just a few words.
    """
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logging.warning(f"Minimal text extraction failed for {filename}: {e}")
        return ""

# ----------------------------------------------------------------------------
# Currency Conversion Functions (FX)
# ----------------------------------------------------------------------------

CENT = Decimal("0.01")
RATE_PREC = Decimal("0.0001")

def q_money(x) -> Decimal:
    """Quantize money to 2 decimal places."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

def q_rate(x) -> Decimal:
    """Quantize rate to 4 decimal places."""
    return Decimal(str(x)).quantize(RATE_PREC, rounding=ROUND_HALF_UP)

def _prev_business_day(d: date) -> date:
    """Get previous business day (skip weekends)."""
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d

def get_eur_rate(invoice_date: date, ccy: str) -> Decimal:
    """
    Returns rate to convert 1 CCY -> EUR on (or before) invoice_date.
    Uses exchangerate.host (mirrors ECB; no API key).
    Fetches historical rate for invoice_date (or previous business day if weekend).
    """
    ccy = (ccy or "").upper().strip()
    if ccy == "EUR":
        return Decimal("1")

    d = _prev_business_day(invoice_date)
    for _ in range(7):  # look back up to 7 calendar days to skip holidays
        url = f"https://api.exchangerate.host/{d.isoformat()}?base={ccy}&symbols=EUR"
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            js = r.json()
            rate = js.get("rates", {}).get("EUR")
            if rate:
                return q_rate(rate)
        except Exception as ex:
            logging.warning(f"FX fetch failed {url}: {ex}")
        d = d - timedelta(days=1)
        d = _prev_business_day(d)
    raise ValueError(f"No EUR rate found for {ccy} near {invoice_date.isoformat()}.")

def _convert_to_eur_fields(entry: dict, conversion_enabled: bool = True) -> dict:
    """
    Adds EUR-converted fields based on invoice date & currency using exchangerate.host.
    Adds: FX Rate (ccy->EUR), Nett Amount (EUR), VAT Amount (EUR), Gross Amount (EUR)
    """
    if not conversion_enabled:
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Conversion Note"] = "Currency conversion disabled"
        return entry
    
    try:
        ccy = (entry.get("Currency") or "").upper()
        inv_date_str = entry.get("Date")
        
        if not inv_date_str or not ccy or ccy == "EUR":
            # No conversion needed for EUR or missing data
            if ccy == "EUR":
                entry["FX Rate (ccy->EUR)"] = "1.0000"
                entry["Nett Amount (EUR)"] = round(float(entry.get("Nett Amount", 0)), 2)
                entry["VAT Amount (EUR)"] = round(float(entry.get("VAT Amount", 0)), 2)
                entry["Gross Amount (EUR)"] = round(float(entry.get("Gross Amount", 0)), 2)
            else:
                entry["FX Rate (ccy->EUR)"] = None
                entry["Nett Amount (EUR)"] = None
                entry["VAT Amount (EUR)"] = None
                entry["Gross Amount (EUR)"] = None
                entry["FX Error"] = "Missing date or currency for conversion"
            return entry
        
        inv_dt = date.fromisoformat(inv_date_str)
        rate = get_eur_rate(inv_dt, ccy)  # 1 CCY => EUR
        entry["FX Rate (ccy->EUR)"] = str(rate)  # keep string for exactness
        entry["FX Rate Date"] = inv_dt.isoformat()  # Date used for rate lookup

        # Convert amounts using Decimal for precision
        for k_src, k_dst in [
            ("Nett Amount", "Nett Amount (EUR)"),
            ("VAT Amount",  "VAT Amount (EUR)"),
            ("Gross Amount","Gross Amount (EUR)")
        ]:
            amt = Decimal(str(entry.get(k_src, 0)))
            converted = q_money(amt * rate)
            entry[k_dst] = round(float(converted), 2)
        
        return entry
    except Exception as ex:
        # Do not fail the whole invoice if FX fetch fails—attach note.
        logging.error(f"EUR conversion failed for entry: {ex}")
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Error"] = f"EUR conversion failed: {str(ex)}"
        return entry

def structure_text_with_llm(invoice_text: str, filename: str) -> dict:
    """
    Step 2: Sends the text and the robust prompt to the LLM to get structured JSON.
    """
    full_prompt = f"{LLM_PROMPT}\n\n**INVOICE TEXT TO PARSE:**\n{invoice_text}"
    
    try:
        # === OpenAI API Integration ===
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo" for better performance
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Return only valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,  # Deterministic output
            response_format={"type": "json_object"}  # Force JSON mode
        )
        
        json_string = response.choices[0].message.content.strip()

        
        # Clean up the response to get a pure JSON string
        cleaned_json_string = json_string.strip().lstrip("```json").rstrip("```")
        
        return json.loads(cleaned_json_string)

    except json.JSONDecodeError as e:
        logging.error(f"LLM returned invalid JSON for {filename}: {e}. Response text: {json_string}")
        raise ValueError(f"Extraction failed for {filename}: Model returned malformed JSON.")
    except Exception as e:
        logging.error(f"LLM call failed for {filename}: {e}")
        raise ValueError(f"Extraction failed for {filename}: LLM API error.")

def validate_extraction(data: dict, filename: str) -> None:
    """
    Step 3: Validates the extracted JSON for critical financial fields.
    Raises a ValueError if validation fails, which instructs the user to check.
    """
    errors = []
    
    # 1. Check for missing critical fields
    if not data.get("invoice_number"):
        errors.append("Missing 'invoice_number'.")
    if not data.get("invoice_date"):
        errors.append("Missing 'invoice_date'.")
    if data.get("total_amount") is None: # Allow 0.0 but not null
        errors.append("Missing 'total_amount'.")
    if not data.get("vendor_name"):
        errors.append("Missing 'vendor_name'.")
    if not data.get("customer_name"):
        errors.append("Missing 'customer_name'.")
        
    # 2. Financial Sanity Check
    try:
        total = float(data.get("total_amount", 0.0) or 0.0)
        subtotal = float(data.get("subtotal", 0.0) or 0.0)
        vat_sum = sum(float(item.get("tax_amount", 0.0) or 0.0) for item in data.get("vat_breakdown", []))
        
        if vat_sum == 0.0 and data.get("total_vat"):
            vat_sum = float(data.get("total_vat", 0.0) or 0.0)
            
        # Allow for a small rounding difference (e.g., 1 EUR/USD/etc.)
        if total != 0 and abs((subtotal + vat_sum) - total) > 1.0:
            errors.append(
                f"Financial Mismatch: Subtotal ({subtotal}) + VAT ({vat_sum}) "
                f"does not equal Total ({total})."
            )
    except Exception:
        errors.append("Could not perform financial sanity check due to invalid numbers.")
        
    # 3. Raise the error that instructs the user to check
    if errors:
        error_message = f"Manual review required for {filename}: {'. '.join(errors)}"
        logging.error(error_message)
        raise ValueError(error_message)

    logging.info(f"Successfully validated extraction for {filename}.")

def robust_invoice_processor(pdf_bytes: bytes, filename: str) -> dict:
    """
    Runs the complete 3-step extraction and validation pipeline.
    Falls back to basic regex extraction if LLM extraction fails.
    """
    # Step 1: Get Text (with OCR fallback)
    invoice_text = get_text_from_pdf(pdf_bytes, filename)
    
    # Step 2: Try LLM extraction first
    try:
        extracted_data = structure_text_with_llm(invoice_text, filename)
        # Step 3: Validate Data
        validate_extraction(extracted_data, filename)
        # Store invoice text for reverse charge detection
        extracted_data["_invoice_text"] = invoice_text
        # Success! Return the clean, validated data
        return extracted_data
    except Exception as llm_error:
        # LLM extraction failed - try basic regex extraction
        logging.warning(f"LLM extraction failed for {filename}: {llm_error}. Attempting basic financial data extraction...")
        basic_data = _extract_basic_financial_data_from_text(invoice_text)
        
        if basic_data and any(basic_data.values()):
            # Create a minimal valid structure from basic data
            logging.info(f"Extracted basic financial data from {filename}: {basic_data}")
            return {
                "invoice_number": f"EXTRACTED-{filename}",
                "invoice_date": date.today().isoformat(),  # Use today as fallback
                "vendor_name": "Unknown",
                "customer_name": "Unknown",
                "subtotal": basic_data.get("subtotal") or (basic_data.get("total_amount", 0) - basic_data.get("total_vat", 0)),
                "total_amount": basic_data.get("total_amount") or 0.0,
                "total_vat": basic_data.get("total_vat") or 0.0,
                "currency": "EUR",  # Default to EUR
                "vat_breakdown": [
                    {
                        "rate": basic_data.get("vat_rate") or 0.0,
                        "base_amount": basic_data.get("subtotal") or (basic_data.get("total_amount", 0) - basic_data.get("total_vat", 0)),
                        "tax_amount": basic_data.get("total_vat") or 0.0
                    }
                ] if basic_data.get("vat_rate") or basic_data.get("total_vat") else [],
                "line_items": [],
                "payment_terms": None,
                "vendor_vat_id": None,
                "customer_vat_id": None,
                "due_date": None,
                "extraction_method": "basic_regex_fallback",
                "extraction_note": f"Full extraction failed. Extracted basic financial data using regex patterns. Original error: {str(llm_error)}",
                "_invoice_text": invoice_text  # Store invoice text for reverse charge detection
            }
        else:
            # Even basic extraction failed - re-raise the original error
            raise llm_error

# ----------------------------------------------------------------------------
# Helpers: Kept your originals
# ----------------------------------------------------------------------------

def _normalize_company_name(name: str) -> str:
    if not name:
        return ""
    # Casefold (stronger than lower), strip punctuation & multiple spaces
    n = name.casefold()
    n = re.sub(r"[\s\-_/.,()]+", " ", n)
    return n.strip()


def _split_company_list(raw: str) -> List[str]:
    # Accept comma/newline/semicolon separated
    if not raw:
        return []
    parts = re.split(r"[,\n;]", raw)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned

# ----------------------------------------------------------------------------
# Mapping & Classification (Modified to fit new pipeline)
# ----------------------------------------------------------------------------

def _classify_vat_category(vat_percentage: Optional[float], invoice_text: str = "") -> str:
    """
    Classifies VAT category based on VAT percentage and invoice text.
    Returns: "Standard VAT", "Reduced Rate", "Standard Rate (from another country)", 
             "Zero Rated", "Reverse Charge", or "Unknown"
    """
    # Check for reverse charge indicators in text (case-insensitive)
    if invoice_text:
        reverse_charge_keywords = [
            "reverse charge", "btw verlegd", "vat verlegd", "omgekeerde heffing",
            "reverse charge vat", "rcm", "reverse charge mechanism"
        ]
        text_lower = invoice_text.lower()
        if any(keyword in text_lower for keyword in reverse_charge_keywords):
            return "Reverse Charge"
    
    # If no VAT percentage, return Unknown
    if vat_percentage is None:
        return "Unknown"
    
    vat_percentage = float(vat_percentage)
    
    # Classify based on percentage (with small tolerance for rounding)
    if abs(vat_percentage - 21.0) < 0.1:
        return "Standard VAT"
    elif abs(vat_percentage - 9.0) < 0.1:
        return "Reduced Rate"
    elif abs(vat_percentage - 14.0) < 0.1:
        return "Standard Rate (from another country)"
    elif abs(vat_percentage - 0.0) < 0.1:
        return "Zero Rated"
    else:
        # For other rates, try to classify
        if 20.0 <= vat_percentage <= 22.0:
            return "Standard VAT"
        elif 8.0 <= vat_percentage <= 10.0:
            return "Reduced Rate"
        elif 13.0 <= vat_percentage <= 15.0:
            return "Standard Rate (from another country)"
        else:
            return f"Other ({vat_percentage}%)"

def _map_llm_output_to_register_entry(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps the rich, validated JSON from the LLM to your final register entry format.
    This REPLACES your old _map_transaction_to_register_entry
    """
    # Get the main description from the first line item, or join all
    description = ""
    if llm_data.get("line_items"):
        description = llm_data["line_items"][0].get("description", "")
    
    # Extract VAT percentage(s) from vat_breakdown
    vat_percentage = None
    vat_breakdown = llm_data.get("vat_breakdown", [])
    if vat_breakdown:
        # If multiple VAT rates, use the primary one (highest base_amount)
        rates = [item.get("rate", 0.0) for item in vat_breakdown if item.get("rate") is not None]
        if len(rates) == 1:
            vat_percentage = rates[0]
        elif len(rates) > 1:
            # Multiple rates: use the rate with the highest base_amount as primary
            primary_item = max(vat_breakdown, key=lambda x: float(x.get("base_amount", 0) or 0))
            vat_percentage = primary_item.get("rate")
    
    # If still no VAT percentage found, try calculating from total_vat and subtotal
    if vat_percentage is None:
        total_vat = float(llm_data.get("total_vat", 0) or 0)
        subtotal = float(llm_data.get("subtotal", 0) or 0)
        if subtotal > 0:
            vat_percentage = round((total_vat / subtotal) * 100, 2)
    
    # Get invoice text for reverse charge detection
    invoice_text = llm_data.get("_invoice_text", "")
    
    # Classify VAT category
    vat_category = _classify_vat_category(vat_percentage, invoice_text)
    
    return {
        "Date": llm_data.get("invoice_date"),
        "Invoice Number": llm_data.get("invoice_number"),
        "Vendor Name": llm_data.get("vendor_name"),
        "Customer Name": llm_data.get("customer_name"),
        "Type": "Unclassified",  # This will be set by _classify_type
        "Nett Amount": llm_data.get("subtotal") or 0.0,
        "VAT Amount": llm_data.get("total_vat") or 0.0,
        "VAT %": vat_percentage,  # Added VAT percentage
        "VAT Category": vat_category,  # Added VAT category
        "Gross Amount": llm_data.get("total_amount") or 0.0,
        "Currency": llm_data.get("currency") or "EUR",
        "Description": description,
        # Optional: keep the full data for debugging
        "Full_Extraction_Data": llm_data 
    }


def _is_credit_note(invoice_text: str, invoice_number: str = None) -> bool:
    """
    Detects if a document is a credit note by checking for credit note keywords.
    """
    if not invoice_text:
        return False
    
    text_lower = invoice_text.lower()
    invoice_num_lower = (invoice_number or "").lower()
    
    # Keywords that indicate a credit note
    credit_note_keywords = [
        "credit note",
        "creditnote",
        "credit memo",
        "creditmemo",
        "creditnota",
        "credit nota",
        "nota de credito",
        "nota de crédito",
        "avoir",
        "gutschrift",
        "nota di credito",
        "crédit",
        "cn-",
        "-cn",
        "cn:",
        "credit:"
    ]
    
    # Check if any keyword appears in the text
    for keyword in credit_note_keywords:
        if keyword in text_lower or keyword in invoice_num_lower:
            return True
    
    return False

def _apply_credit_note_negative_amounts(register_entry: Dict[str, Any], invoice_text: str = None) -> Dict[str, Any]:
    """
    If the document is a credit note, ensures that Nett Amount and Gross Amount are negative.
    If they already have a negative sign, leaves them unchanged.
    If they don't have a negative sign, makes them negative.
    Also makes VAT Amount negative if net and gross are negative.
    """
    # Get invoice text from register entry if not provided
    if not invoice_text:
        full_data = register_entry.get("Full_Extraction_Data", {})
        invoice_text = full_data.get("_invoice_text", "")
    
    invoice_number = register_entry.get("Invoice Number", "")
    
    # Check if this is a credit note
    if not _is_credit_note(invoice_text, invoice_number):
        return register_entry
    
    # Get current amounts
    nett_amount = register_entry.get("Nett Amount", 0.0) or 0.0
    gross_amount = register_entry.get("Gross Amount", 0.0) or 0.0
    vat_amount = register_entry.get("VAT Amount", 0.0) or 0.0
    
    # Convert to float if they're not already
    try:
        nett_amount = float(nett_amount)
        gross_amount = float(gross_amount)
        vat_amount = float(vat_amount)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert amounts to float for credit note: {register_entry.get('Invoice Number')}")
        return register_entry
    
    # Apply negative sign if not already negative
    if nett_amount > 0:
        register_entry["Nett Amount"] = -abs(nett_amount)
        logging.info(f"Applied negative sign to Nett Amount for credit note: {register_entry.get('Invoice Number')}")
    elif nett_amount == 0:
        # Keep zero as zero
        pass
    else:
        # Already negative, keep as is
        logging.info(f"Nett Amount already negative for credit note: {register_entry.get('Invoice Number')}")
    
    if gross_amount > 0:
        register_entry["Gross Amount"] = -abs(gross_amount)
        logging.info(f"Applied negative sign to Gross Amount for credit note: {register_entry.get('Invoice Number')}")
    elif gross_amount == 0:
        # Keep zero as zero
        pass
    else:
        # Already negative, keep as is
        logging.info(f"Gross Amount already negative for credit note: {register_entry.get('Invoice Number')}")
    
    # Make VAT Amount negative if net and gross are negative (or if it's positive)
    if nett_amount < 0 or gross_amount < 0:
        if vat_amount > 0:
            register_entry["VAT Amount"] = -abs(vat_amount)
            logging.info(f"Applied negative sign to VAT Amount for credit note: {register_entry.get('Invoice Number')}")
        elif vat_amount == 0:
            # Keep zero as zero
            pass
        # If already negative, keep as is
    
    return register_entry

def _classify_type(register_entry: Dict[str, Any], our_companies_list: List[str]) -> str:
    """
    (Your original function - no changes needed)
    Decide Purchase vs Sales.
    """
    vendor_name_norm = _normalize_company_name(register_entry.get("Vendor Name") or "")
    customer_name_norm = _normalize_company_name(register_entry.get("Customer Name") or "")

    our_norms = [_normalize_company_name(x) for x in our_companies_list]

    for oc in our_norms:
        if oc and oc in customer_name_norm:
            return "Purchase"
        if oc and oc in vendor_name_norm:
            return "Sales"
    return register_entry.get("Type") or "Unclassified"


# ----------------------------------------------------------------------------
# FastAPI app (Your original code, re-wired)
# ----------------------------------------------------------------------------

app = FastAPI(
    title="Invoice Transaction Register Extractor",
    description=(
        "Upload invoice PDFs and extract structured transaction register data. "
        "Automatically classifies as Purchase or Sales."
    ),
    version="4.0.0 (FX Conversion + Enhanced OCR)", # Updated version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Robust Invoice Transaction Register Extractor API", # Updated message
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload - Upload a single invoice PDF",
            "upload_multiple": "POST /upload-multiple - Upload multiple invoice PDFs",
        },
        "supported_formats": ["PDF"],
        "pipeline": ["PyPDF2", "Textract detect_document_text", "Textract analyze_document (TABLES/FORMS)", "LLM", "Validation", "FX Conversion"]
    }


@app.post("/upload")
async def upload_and_extract(
    file: UploadFile = File(...),
    our_companies: str = Form(
        ...,
        description=(
            "Company name(s): single (e.g., 'Dutch Food Solutions B.V.') or comma-/newline-separated "
            "(e.g., 'Dutch Food Solutions B.V., Biofount (Netherlands) B.V.')"
        ),
    ),
):
    try:
        # Keep your original validation
        if not (file.content_type or "").startswith("application/pdf"):
            return JSONResponse(status_code=400, content={
                "filename": file.filename, "status": "error",
                "error": f"Unsupported content-type: {file.content_type}. Only application/pdf is accepted."
            })

        our_companies_list = _split_company_list(our_companies)
        if not our_companies_list:
            return JSONResponse(status_code=400, content={
                "filename": file.filename, "status": "error",
                "error": "No company name(s) provided. Provide a single name or a comma/newline-separated list."
            })

        file_bytes = await file.read()
        
        # --- THIS IS THE MODIFIED PART ---
        logging.info(f"Processing {file.filename} with robust pipeline...")
        
        # 1. Call the main pipeline function
        llm_data = robust_invoice_processor(file_bytes, file.filename)
        
        # 2. Map to register format
        register_entry = _map_llm_output_to_register_entry(llm_data)
        
        # 3. Apply credit note negative amounts logic
        invoice_text = llm_data.get("_invoice_text", "")
        register_entry = _apply_credit_note_negative_amounts(register_entry, invoice_text)
        
        # 4. Classify
        register_entry["Type"] = _classify_type(register_entry, our_companies_list)
        
        # 5. Apply currency conversion
        register_entry = _convert_to_eur_fields(register_entry, conversion_enabled)
        
        # 5. Add conversion info to Full_Extraction_Data for audit trace
        if "Full_Extraction_Data" in register_entry:
            register_entry["Full_Extraction_Data"]["fx_conversion"] = {
                "enabled": conversion_enabled,
                "rate": register_entry.get("FX Rate (ccy->EUR)"),
                "rate_date": register_entry.get("FX Rate Date"),
                "error": register_entry.get("FX Error")
            }
        # --- END MODIFIED PART ---
        
        return {
            "filename": file.filename,
            "status": "success",
            "register_entry": register_entry,
        }
    except Exception as e:
        # This now catches errors from the robust pipeline,
        # including the "Manual review required..." message.
        logging.exception(f"Error processing {file.filename}")
        return JSONResponse(status_code=500, content={
            "filename": file.filename,
            "status": "error",
            "error": str(e), # Pass the user-friendly error message
            "register_entry": None,
        })


@app.post("/upload-multiple")
async def upload_multiple_and_extract(
    files: List[UploadFile] = File(...),
    our_companies: str = Form(
        ..., description=(
            "Company name(s): single or comma-/newline-separated"
        )
    ),
):
    if not files:
        return JSONResponse(status_code=400, content={
            "status": "error", "error": "No files provided", "results": [],
        })

    try:
        our_companies_list = _split_company_list(our_companies)
        if not our_companies_list:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "error": "No company name(s) provided. Provide a single name or a comma/newline-separated list.",
                "results": [],
            })

        # Logic from your process_multiple_invoices is now in the endpoint
        results = []
        
        for file in files:
            if not file.filename or not (file.content_type or "").startswith("application/pdf"):
                logging.warning(f"Skipping invalid file: {file.filename}")
                results.append({
                    "file_name": file.filename or "unknown",
                    "status": "error",
                    "error": "Invalid file or content type (not PDF).",
                    "register_entry": None,
                })
                continue

            logging.info(f"— Processing file: {file.filename}")
            try:
                file_bytes = await file.read()
                if not file_bytes:
                     raise ValueError("File is empty.")

                # --- THIS IS THE MODIFIED PART ---
                # 1. Call the main pipeline function
                llm_data = robust_invoice_processor(file_bytes, file.filename)
                
                # 2. Map to register format
                register_entry = _map_llm_output_to_register_entry(llm_data)
                
                # 3. Apply credit note negative amounts logic
                invoice_text = llm_data.get("_invoice_text", "")
                register_entry = _apply_credit_note_negative_amounts(register_entry, invoice_text)
                
                # 4. Classify
                register_entry["Type"] = _classify_type(register_entry, our_companies_list)
                
                # 5. Apply currency conversion
                register_entry = _convert_to_eur_fields(register_entry, conversion_enabled)
                
                # 5. Add conversion info to Full_Extraction_Data for audit trace
                if "Full_Extraction_Data" in register_entry:
                    register_entry["Full_Extraction_Data"]["fx_conversion"] = {
                        "enabled": conversion_enabled,
                        "rate": register_entry.get("FX Rate (ccy->EUR)"),
                        "rate_date": register_entry.get("FX Rate Date"),
                        "error": register_entry.get("FX Error")
                    }
                # --- END MODIFIED PART ---
                
                results.append({
                    "file_name": file.filename,
                    "status": "success",
                    "register_entry": register_entry,
                })
                
            except Exception as e:
                # This catches errors from the pipeline (e.g., "Manual review required...")
                logging.exception(f"Failed to process {file.filename}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "error": str(e), # The user-friendly error
                    "register_entry": None,
                })

        # Summary logic with native currency totals
        total_files = len(files)
        successful_files = sum(1 for r in results if r.get("status") == "success")
        failed_files = sum(1 for r in results if r.get("status") == "error")

        total_nett = Decimal("0.00")
        total_vat = Decimal("0.00")
        total_gross = Decimal("0.00")
        
        # EUR converted totals
        total_nett_eur = Decimal("0.00")
        total_vat_eur = Decimal("0.00")
        total_gross_eur = Decimal("0.00")
        
        for r in results:
            if r.get("status") == "success" and r.get("register_entry"):
                e = r["register_entry"]
                # Native currency totals
                total_nett += q_money(e.get("Nett Amount", 0.0) or 0.0)
                total_vat += q_money(e.get("VAT Amount", 0.0) or 0.0)
                total_gross += q_money(e.get("Gross Amount", 0.0) or 0.0)
                
                # EUR converted totals (if conversion was successful)
                nett_eur = e.get("Nett Amount (EUR)")
                vat_eur = e.get("VAT Amount (EUR)")
                gross_eur = e.get("Gross Amount (EUR)")
                
                if nett_eur is not None:
                    total_nett_eur += q_money(nett_eur)
                if vat_eur is not None:
                    total_vat_eur += q_money(vat_eur)
                if gross_eur is not None:
                    total_gross_eur += q_money(gross_eur)

        response = {
            "status": "success",
            "summary": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_nett_amount": round(float(total_nett), 2),
                "total_vat_amount": round(float(total_vat), 2),
                "total_gross_amount": round(float(total_gross), 2),
                "note": "Native currency totals (summed regardless of currency)."
            },
            "results": results,
        }
        
        # Add EUR converted summary if conversion is enabled
        if conversion_enabled:
            response["eur_converted_summary"] = {
                "total_nett_amount_eur": round(float(total_nett_eur), 2),
                "total_vat_amount_eur": round(float(total_vat_eur), 2),
                "total_gross_amount_eur": round(float(total_gross_eur), 2),
                "note": "All amounts converted to EUR using historical exchange rates (ECB reference via exchangerate.host)."
            }
        
        return response
    except Exception as e:
        logging.exception("Error processing multiple files")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": str(e),
            "results": [],
        })


# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    # Check for PORT environment variable (set by Render/Heroku)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    # If PORT is set (production), use 0.0.0.0 to bind to all interfaces
    if os.getenv("PORT"):
        host = "0.0.0.0"
    logging.info(f"Starting uvicorn server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)