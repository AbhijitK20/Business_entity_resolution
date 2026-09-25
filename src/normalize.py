"""Text normalization module for business names and addresses.

Includes Indic→Latin transliteration (ITRANS + word-final schwa deletion)
ported from the SABER team's measured approach — fixes the 22.7% of India
S1–S2 pairs that are cross-script (India recall@20 0.835 → 0.992 in their
measurements).
"""
import re
import unicodedata
import warnings
import pandas as pd
from typing import Optional

# --- Indic transliteration (optional dependency, graceful fallback) ---------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _itrans
    _HAS_INDIC = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_INDIC = False

# 9 script blocks: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
# Tamil, Telugu, Kannada, Malayalam
_INDIC_BLOCKS = [
    (0x0900, "devanagari"), (0x0980, "bengali"), (0x0A00, "gurmukhi"),
    (0x0A80, "gujarati"), (0x0B00, "oriya"), (0x0B80, "tamil"),
    (0x0C00, "telugu"), (0x0C80, "kannada"), (0x0D00, "malayalam"),
]
_INDIC_RUN = re.compile(r"[\u0900-\u0D7F\u200c\u200d]+")
_INDIC_ANY = re.compile(r"[\u0900-\u0D7F]")


def _script_of(ch: str) -> Optional[str]:
    """Return the script name for an Indic character, else None."""
    o = ord(ch)
    for base, name in _INDIC_BLOCKS:
        if base <= o < base + 0x80:
            return name
    return None


def _translit_run(match: "re.Match") -> str:
    """Transliterate one Indic run to Latin (ITRANS + schwa deletion)."""
    text = match.group(0).replace("\u200c", "").replace("\u200d", "")
    script_name = next((_script_of(c) for c in text if _script_of(c)), None)
    if script_name is None:
        return text
    if not _HAS_INDIC:
        # Graceful fallback (text is returned unchanged) — but never silently:
        # one warning per run so a missing dependency cannot masquerade as a
        # successful transliteration. Not installed, not requirements.txt edited.
        warnings.warn(
            "indic-transliteration is not installed: Indic-script text was left "
            "untransliterated (graceful fallback, NOT a successful transliteration). "
            "Install 'indic-transliteration' to enable V1 cross-script normalization.",
            RuntimeWarning,
        )
        return text
    script = getattr(sanscript, script_name.upper(), None)
    if script is None and script_name == "oriya":
        script = getattr(sanscript, "ODIA", None)  # newer scheme name
    if script is None:
        warnings.warn(
            f"indic_transliteration exposes no scheme for {script_name!r}: "
            "text left untransliterated (graceful fallback).",
            RuntimeWarning,
        )
        return text
    out = _itrans(text, script, sanscript.ITRANS)
    # Drop source-script characters the scheme has no mapping for. ITRANS
    # leaves candrabindu (U+0949), Gujarati candrabindu (U+0A89) and Odia
    # nakaaraa (U+0B3C) untouched, so they survive into the "Latin" output and
    # break exact-key blocking plus every trigram/MinHash key built from it.
    # Only characters of the SOURCE script are removed, so Latin diacritics
    # (Saint-Etienne, Cafe Zurich) are untouched. If stripping would leave no
    # Latin letters at all, the transliteration did not really happen, so the
    # original text is kept rather than emitting an empty string.
    stripped = "".join(c for c in out if _script_of(c) != script_name)
    if any(c.isalpha() and ord(c) < 128 for c in stripped):
        out = stripped
    # word-final schwa deletion: rama -> ram, marketinga -> marketing.
    # ITRANS writes the same inherent vowel as uppercase A in some outputs
    # (kRRiShNA -> krishna), so both cases are deleted. Long "I" (ii) is
    # deliberately NOT stripped: it is a real vowel, not schwa.
    out = re.sub(r"(?<=[^aeiouAEIOU\s])[aA]\b", "", out)
    # ITRANS cleanups for anusvara / conjuncts
    out = (out.replace("~N", "n").replace(".N", "n")
              .replace("M", "n").replace("JN", "gy"))
    return out


def transliterate_indic(text: str) -> str:
    """Replace every Indic-script run in `text` with Latin transliteration."""
    if not text or not _INDIC_ANY.search(text):
        return text
    return _INDIC_RUN.sub(_translit_run, text)


def is_non_latin(text: str) -> int:
    """1 if the text contains any non-Latin letters (script feature)."""
    if not text:
        return 0
    for ch in text:
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, ""):
            return 1
    return 0


# Ligature map for French/European characters (œ, æ, ß, ø, ł, đ, curly apostrophe)
_LIGATURE_MAP = str.maketrans({
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss",
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
    "’": "'",
})


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
    
    # Step 0: Indic → Latin transliteration (before ASCII folding destroys it)
    name = transliterate_indic(name)
    
    # Step 1: Ligatures + Unicode normalization (handles accented chars)
    name = name.translate(_LIGATURE_MAP)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    
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
    
    # Step 7: Strip non-alphanumeric (Unicode-aware \w keeps any residual letters)
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
    
    addr = str(addr)
    
    # Step 0: Indic → Latin transliteration, then ligatures
    addr = transliterate_indic(addr)
    addr = addr.translate(_LIGATURE_MAP)
    addr = unicodedata.normalize("NFKD", addr)
    addr = "".join(c for c in addr if not unicodedata.combining(c))
    addr = addr.lower()
    
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


def _is_non_latin_value(value) -> int:
    """is_non_latin() over one raw cell, tolerating None/NaN/numeric values."""
    if value is None or pd.isna(value):
        return 0
    return is_non_latin(str(value))


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to all relevant columns in a dataframe.

    Normalized columns are written alongside the raw ones (raw is never
    mutated). Script flags are computed from the RAW text — after
    transliteration the clean text is Latin by construction, so the flag would
    otherwise always be 0 (BLUEPRINT §2.1 "script flag feature").
    """
    df = df.copy()

    if "business_name" in df.columns:
        df["business_name_clean"] = df["business_name"].apply(normalize_name)
        df["business_name_is_non_latin"] = df["business_name"].apply(_is_non_latin_value)

    if "business_address" in df.columns:
        df["business_address_clean"] = df["business_address"].apply(normalize_address)
        df["business_address_is_non_latin"] = df["business_address"].apply(_is_non_latin_value)

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
