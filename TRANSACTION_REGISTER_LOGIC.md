# Transaction Register Logic - Explanation 

## Overview
This system automatically extracts invoice data from PDF documents and creates standardized transaction register entries suitable for accounting and VAT reporting. The system processes invoices through a multi-stage pipeline that ensures accuracy, proper classification, and currency standardization.

---

## Complete Processing Pipeline

### Stage 1: PDF Text Extraction
**Purpose:** Extract readable text from invoice PDFs (including scanned documents)

**Process:**
1. **Primary Method - PyPDF2:** Attempts direct text extraction from PDF structure
   - Fast and cost-effective for digital PDFs
   - Works when PDF contains embedded text layers

2. **Fallback Method 1 - AWS Textract (detect_document_text):** 
   - Used when PyPDF2 finds minimal text (< 100 characters)
   - OCR-based extraction for scanned PDFs
   - Extracts text line by line

3. **Fallback Method 2 - AWS Textract (analyze_document with TABLES & FORMS):**
   - Final fallback for complex scanned documents
   - Specifically designed for structured documents with tables and forms
   - Extracts key-value pairs (e.g., "Invoice Date: 2025-01-15")
   - Preserves table structure and relationships

**Audit Trail:** System logs which extraction method was used for each invoice

---

### Stage 2: AI-Powered Data Extraction (LLM)
**Purpose:** Convert unstructured text into structured financial data

**Process:**
- Uses OpenAI GPT-4o model with specialized financial extraction prompt
- Extracts the following fields:
  - **Invoice Metadata:** Invoice number, invoice date, due date
  - **Parties:** Vendor name, vendor VAT ID, vendor address
  - **Parties:** Customer name, customer VAT ID, customer address
  - **Financial Data:** Currency, subtotal, total VAT, total amount
  - **VAT Breakdown:** Individual VAT rates and amounts
  - **Line Items:** Description, quantity, unit price, line totals
  - **Payment Terms:** Any payment terms or notes

**Validation Rules Applied:**
- Dates must be in ISO format (YYYY-MM-DD)
- Monetary values must be numeric (no currency symbols)
- Subtotal + VAT must approximately equal total amount
- All required fields must be present

**Output:** Structured JSON object with validated financial data

---

### Stage 3: Data Mapping to Register Format
**Purpose:** Transform extracted data into standardized transaction register entry

**Mapping Logic:**
```
LLM Extracted Data → Register Entry Format
├── invoice_date → Date
├── invoice_number → Invoice Number
├── vendor_name → Vendor Name
├── customer_name → Customer Name
├── subtotal → Nett Amount
├── total_vat → VAT Amount
├── total_amount → Gross Amount
├── currency → Currency
└── line_items[0].description → Description
```

**Register Entry Structure:**
- **Date:** Invoice date (YYYY-MM-DD)
- **Invoice Number:** Unique invoice identifier
- **Vendor Name:** Supplier/seller name
- **Customer Name:** Buyer/client name
- **Type:** Purchase/Sales/Unclassified (determined in next stage)
- **Nett Amount:** Amount before VAT (subtotal)
- **VAT Amount:** Total VAT/tax amount
- **Gross Amount:** Total amount including VAT
- **Currency:** Original invoice currency (EUR, USD, GBP, etc.)
- **Description:** Main line item description

---

### Stage 4: Transaction Classification (Purchase vs Sales)
**Purpose:** Automatically classify transactions based on company relationship

**Classification Logic:**
1. **Input:** List of "our companies" (e.g., "Dutch Food Solutions B.V.", "Biofount (Netherlands) B.V.")
2. **Normalization:** Company names are normalized (lowercase, punctuation removed) for matching
3. **Matching Rules:**
   - **Purchase:** If "our company" appears in the **Customer Name** field
     - Logic: We are the customer → We are buying → This is a Purchase
   - **Sales:** If "our company" appears in the **Vendor Name** field
     - Logic: We are the vendor → We are selling → This is a Sales
   - **Unclassified:** If neither match is found

**Example:**
- Invoice shows: Vendor = "ABC Supplies Ltd", Customer = "Dutch Food Solutions B.V."
- System matches "Dutch Food Solutions B.V." in customer field
- Classification: **Purchase**

**Audit Note:** Classification is based on exact name matching after normalization. Manual review recommended for edge cases.

---

### Stage 5: Currency Conversion to EUR
**Purpose:** Standardize all amounts to EUR for consolidated reporting

**Conversion Process:**
1. **Rate Source:** exchangerate.host API (mirrors ECB daily reference rates)
2. **Rate Date:** Uses invoice date (or previous business day if invoice date is weekend/holiday)
3. **Historical Accuracy:** Fetches historical exchange rate for the exact invoice date
4. **Fallback Logic:** If rate not available on invoice date, looks back up to 7 days for valid business day rate

**Conversion Calculation:**
- **FX Rate:** Retrieved from API (e.g., 1 USD = 0.9234 EUR)
- **Nett Amount (EUR)** = Nett Amount × FX Rate (rounded to 2 decimals)
- **VAT Amount (EUR)** = VAT Amount × FX Rate (rounded to 2 decimals)
- **Gross Amount (EUR)** = Gross Amount × FX Rate (rounded to 2 decimals)

**Precision:**
- Uses `Decimal` type for all calculations (prevents floating-point errors)
- Rates stored with 4 decimal precision
- Final amounts rounded to 2 decimal places (standard accounting practice)

**Error Handling:**
- If conversion fails (API error, invalid currency, etc.), system:
  - Logs the error
  - Sets EUR fields to `null`
  - Adds `FX Error` field with error message
  - **Does NOT fail the entire invoice** - original currency amounts remain valid

**Audit Fields Added:**
- `FX Rate (ccy->EUR)`: Exchange rate used
- `FX Rate Date`: Date for which rate was fetched
- `FX Error`: Error message if conversion failed (if applicable)

---

### Stage 6: Final Register Entry Structure

**Complete Register Entry Contains:**

**Core Fields:**
- Date
- Invoice Number
- Vendor Name
- Customer Name
- Type (Purchase/Sales/Unclassified)
- Nett Amount (original currency)
- VAT Amount (original currency)
- Gross Amount (original currency)
- Currency (original)
- Description

**EUR Conversion Fields:**
- FX Rate (ccy->EUR)
- FX Rate Date
- Nett Amount (EUR)
- VAT Amount (EUR)
- Gross Amount (EUR)
- FX Error (if applicable)

**Audit Trail:**
- Full_Extraction_Data: Complete original LLM extraction (for verification)
- fx_conversion: Conversion metadata (rate, date, errors)

---

## Batch Processing (/upload-multiple endpoint)

**Process:**
1. Processes each invoice through Stages 1-6 individually
2. Maintains separate results for each file (success/error tracking)
3. Calculates two types of summaries:

**Summary 1: Native Currency Totals**
- Sums all amounts in their original currencies
- Note: These are mixed-currency totals (for reference only)

**Summary 2: EUR Converted Summary** (if conversion enabled)
- Sums all EUR-converted amounts
- Provides consolidated totals for reporting
- All amounts standardized to EUR using historical rates

---

## Data Quality & Validation

### Financial Validation:
- Subtotal + VAT ≈ Total Amount (within tolerance)
- All monetary values are numeric
- Dates are valid calendar dates in ISO format
- Currency codes are recognized ISO codes

### Business Logic Validation:
- Invoice number present
- At least one party (vendor or customer) identified
- Financial amounts are non-negative (unless credit note)

### Error Handling:
- Missing fields: System flags but continues processing
- Invalid dates: Raises error requiring manual review
- Math inconsistencies: Raises error requiring manual review
- FX conversion failures: Logs error but preserves original currency data

---

## Accounting Standards Compliance

### VAT Reporting:
- Separates Nett Amount, VAT Amount, and Gross Amount
- Preserves VAT breakdown by rate (if available)
- Maintains original currency for multi-currency reporting

### Audit Trail:
- Full extraction data preserved for verification
- FX conversion details (rate, date) for audit
- Extraction method logged (PyPDF2/Textract) for transparency

### Currency Handling:
- Uses ECB reference rates (standard for EU reporting)
- Historical rates ensure accurate period reporting
- Decimal precision prevents rounding errors

---

## Use Cases

### 1. Purchase Invoice Processing
- Extracts supplier invoices
- Classifies as "Purchase"
- Converts to EUR for consolidated reporting
- Ready for accounts payable entry

### 2. Sales Invoice Processing
- Extracts customer invoices
- Classifies as "Sales"
- Converts to EUR for revenue reporting
- Ready for accounts receivable entry

### 3. Multi-Currency Consolidation
- Processes invoices in various currencies (USD, GBP, EGP, etc.)
- Converts all to EUR using historical rates
- Provides EUR summary for financial statements

---

## Technical Notes for Accountants

### Why This Approach?
1. **Automation:** Reduces manual data entry errors
2. **Consistency:** Standardized format across all invoices
3. **Accuracy:** AI extraction + validation ensures data quality
4. **Auditability:** Full extraction data preserved for verification
5. **Multi-Currency:** Automatic conversion using official rates

### Limitations & Recommendations:
1. **Classification:** Purchase/Sales classification is based on name matching - manual review recommended for complex structures
2. **VAT Rates:** System extracts VAT amounts but may not always identify specific VAT categories (standard/reduced/exempt) - manual categorization may be needed
3. **Currency Conversion:** Uses ECB rates - appropriate for EU reporting, but may need adjustment for other jurisdictions
4. **Scanned Documents:** OCR accuracy depends on document quality - poor scans may require manual verification

### Integration Points:
- Register entries can be exported to accounting software
- EUR totals ready for financial statement preparation
- Original currency data preserved for multi-currency ledgers
- Full audit trail supports compliance requirements

---

## Summary

The transaction register system:
1. **Extracts** invoice data using AI and OCR
2. **Validates** financial accuracy and completeness
3. **Classifies** transactions as Purchase/Sales
4. **Converts** all amounts to EUR using historical exchange rates
5. **Preserves** original data and audit trail

Result: Standardized, EUR-converted transaction register entries ready for accounting and VAT reporting, with full audit trail for compliance.

