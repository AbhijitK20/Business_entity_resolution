"""Text normalization module for business names and addresses."""
import re
import unicodedata
import pandas as pd
from typing import Optional


# Legal suffixes to strip — covers US, India, UK, France, Germany forms
LEGAL_SUFFIXES = [
    # US
    r"\bincorporated\b", r"\binc\.?\b", r"\bcorporation\b",
    r"\bcorp\.?\b", r"\bcompany\b", r"\bco\.?\b",
    r"\bllc\b", r"\bllp\b", r"\blp\b", r"\bpllc\b", r"\bpc\b",
    # India / UK
    r"\blimited\b", r"\bltd\.?\b", r"\bplc\b", r"\bpvt\b", r"\bprivate\b",
    r"\bopc\b",
    # France
    r"\bsarl\b", r"\bsas\b", r"\bsasu\b", r"\beurl\b",
    r"\bsnc\b", r"\bsci\b", r"\bsa\b",
    # Germany / Europe
    r"\bgmbh\b", r"\bag\b", r"\bkg\b", r"\bug\b",
    r"\bbv\b", r"\bnv\b", r"\boy\b", r"\bab\b",
    # Generic
    r"\bgroup\b", r"\bholdings\b", r"\benterprises?\b",
    r"\bassociates?\b", r"\bpartners?\b", r"\bthe\b", r"\band\b",
    r"\blaboratory\b", r"\blab\b",
    # Trade-name markers
    r"\bdba\b", r"\btrading as\b",
]

# Abbreviation expansion map
ABBREVIATION_MAP = {
    r"&": "and",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\bsvcs?\b": "services",
    r"\btechs?\b": "technology",
    r"\bmgmt\b": "management",
    r"\bgrp\b": "group",
    r"\bassocs?\b": "associates",
    r"\bsys\b": "systems",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
}

# Street abbreviations
STREET_ABBREVS = {
    r"\bstreet\b": "st", r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd", r"\bdrive\b": "dr",
    r"\broad\b": "rd", r"\blane\b": "ln",
    r"\bcourt\b": "ct", r"\bparkway\b": "pkwy",
    r"\bplace\b": "pl", r"\bcircle\b": "cir",
    r"\bnorth\b": "n", r"\bsouth\b": "s",
    r"\beast\b": "e", r"\bwest\b": "w",
}


def normalize_name(name: Optional[str]) -> str:
    """10-step business name normalization pipeline.
    
    Based on techniques from:
    - due-diligence-agents (18 legal suffixes, iterative stripping)
    - business-record-matcher (abbreviation expansion)
    - UBS-ER (Unicode normalization)
    - canonmap (initialism detection)
    """
    if pd.isna(name) or name == "":
        return ""
    
    name = str(name)
    
    # Step 1: Unicode normalization (handle accented chars)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    
    # Step 2: Lowercase
    name = name.lower()
    
    # Step 3: Remove parenthesized text
    name = re.sub(r"\([^)]*\)", " ", name)
    
    # Step 4: Strip legal suffixes (iterative up to 3x for stacked suffixes)
    for _ in range(3):
        for suffix in LEGAL_SUFFIXES:
            name = re.sub(suffix, " ", name)
    
    # Step 5: Expand abbreviations
    for pattern, replacement in ABBREVIATION_MAP.items():
        name = re.sub(pattern, replacement, name)
    
    # Step 6: Replace punctuation with spaces
    for ch in "&'/,.":
        name = name.replace(ch, " ")
    
    # Step 7: Strip non-alphanumeric
    name = re.sub(r"[^\w\s]", " ", name)
    
    # Step 8: Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    
    # Step 9: Remove digit-only words (for names)
    name = re.sub(r"\w*\d\w*", "", name)
    
    # Step 10: Final whitespace collapse
    name = re.sub(r"\s+", " ", name).strip()
    
    return name


def normalize_address(addr: Optional[str]) -> str:
    """Address normalization pipeline.
    
    Based on techniques from:
    - business-record-matcher (street abbreviation expansion)
    - UBS-ER (punctuation handling)
    """
    if pd.isna(addr) or addr == "":
        return ""
    
    addr = str(addr).lower()
    
    # Expand street abbreviations
    for pattern, replacement in STREET_ABBREVS.items():
        addr = re.sub(pattern, replacement, addr)
    
    # Replace punctuation with spaces
    addr = re.sub(r"[^\w\s]", " ", addr)
    
    # Collapse whitespace
    addr = re.sub(r"\s+", " ", addr).strip()
    
    return addr


def normalize_country(country: Optional[str]) -> str:
    """Normalize country labels."""
    if pd.isna(country):
        return ""
    return str(country).strip().lower()


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to all relevant columns in a dataframe."""
    df = df.copy()
    
    if "business_name" in df.columns:
        df["business_name_clean"] = df["business_name"].apply(normalize_name)
    
    if "business_address" in df.columns:
        df["business_address_clean"] = df["business_address"].apply(normalize_address)
    
    if "country" in df.columns:
        df["country_clean"] = df["country"].apply(normalize_country)
    
    return df


def get_initialism(name: str) -> str:
    """Extract initialism from a name.
    
    From canonmap: bidirectional matching.
    If name is 2-6 chars alpha only, treat AS an initialism.
    Otherwise, extract first letters of each word.
    """
    name_clean = name.strip().upper()
    if name_clean.isalpha() and 2 <= len(name_clean) <= 6 and " " not in name_clean:
        return name_clean
    parts = re.findall(r"[A-Za-z]+", name)
    return "".join(p[0].upper() for p in parts) if parts else ""


def is_company_name(name: str) -> int:
    """Detect if a name looks like a company (vs person).
    
    From UBS-ER: keyword-based detection.
    """
    company_keywords = [
        "ltd", "inc", "co", "corp", "llc", "plc", "limited",
        "incorporated", "company", "corporation", "gmbh", "kg",
        "llp", "pte", "pty", "sa", "sarl", "bv", "nv", "ag",
        "oy", "oyj", "ab", "spa", "srl", "sas", "kft", "ks", "sp",
        "group", "holdings", "partners", "associates",
        "international", "global", "enterprise", "enterprises",
    ]
    name_lower = name.lower()
    for keyword in company_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", name_lower):
            return 1
    return 0
