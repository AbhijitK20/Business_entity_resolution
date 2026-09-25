"""Baseline unit tests for src/normalize.py — EXISTING behavior only.

No production code is changed by these tests. They lock down what the module
does today, including exposing a missing dependency instead of skipping it.

Run: python tests/test_normalize.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_PKG_IMPORT_ERROR = None
try:
    import src.normalize as N
    from src.normalize import (
        normalize_name, normalize_address, apply_normalization,
        transliterate_indic,
    )
    _IMPORT_MODE = "package import (src.normalize)"
except Exception as exc:  # src/__init__.py eagerly imports blocking/model deps
    _PKG_IMPORT_ERROR = exc
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "normalize_under_test",
        Path(__file__).parent.parent / "src" / "normalize.py",
    )
    N = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(N)
    normalize_name = N.normalize_name
    normalize_address = N.normalize_address
    apply_normalization = N.apply_normalization
    transliterate_indic = N.transliterate_indic
    _IMPORT_MODE = (
        "standalone file load of src/normalize.py "
        f"(package import failed: {type(_PKG_IMPORT_ERROR).__name__}: {_PKG_IMPORT_ERROR})"
    )


def _require_indic(label: str) -> None:
    assert N._HAS_INDIC, (
        f"{label}: indic-transliteration is NOT installed -> src/normalize.py "
        "skips transliteration (_HAS_INDIC=False; a RuntimeWarning is emitted, "
        "see test_missing_indic_dependency_surfaces_warning). "
        "Missing dependency: indic-transliteration (see TASK_BREAKDOWN V1)."
    )


def _assert_transliterated(raw: str, lo: int, hi: int, label: str) -> None:
    _require_indic(label)
    out = transliterate_indic(raw)
    assert out.strip(), f"{label}: empty output for {raw!r}"
    assert out != raw, f"{label}: input returned unchanged (not transliterated): {raw!r}"
    remaining = [c for c in out if lo <= ord(c) < hi]
    assert not remaining, f"{label}: {label} script chars remain in output: {out!r}"


# 1. Latin passthrough -------------------------------------------------------
def test_latin_passthrough():
    assert normalize_name("acme robotics") == "acme robotics"
    assert normalize_name("Bright Cafe LLC") == "bright cafe"
    assert normalize_name("Krishna Sweets Pvt Ltd") == "krishna sweets"
    assert normalize_address("500 Market Street, San Jose") == "500 market st san jose"


# 2. Spec canonical French example -----------------------------------------
def test_cafe_de_la_paix_sarl():
    # TASK_BREAKDOWN V1 spec: 'Café de la Paix SARL' -> 'cafe de la paix'
    assert normalize_name("Café de la Paix SARL") == "cafe de la paix"


# 3. French / European ligatures --------------------------------------------
def test_french_ligatures():
    assert normalize_name("Cœur") == "coeur"
    assert normalize_name("Straße") == "strasse"
    assert normalize_name("Æon Œuvre") == "aeon oeuvre"
    assert normalize_name("Łódź") == "lodz"


# 4-12. Indic scripts ---------------------------------------------------------
def test_devanagari_transliteration():
    _assert_transliterated("राम", 0x0900, 0x0980, "Devanagari")


def test_bengali_transliteration():
    _assert_transliterated("বাংলা", 0x0980, 0x0A00, "Bengali")


def test_gurmukhi_transliteration():
    _assert_transliterated("ਪੰਜਾਬ", 0x0A00, 0x0A80, "Gurmukhi")


def test_gujarati_transliteration():
    _assert_transliterated("ગુજરાત", 0x0A80, 0x0B00, "Gujarati")


def test_odia_transliteration():
    _assert_transliterated("ଓଡ଼ିଶା", 0x0B00, 0x0B80, "Odia")


def test_tamil_transliteration():
    _assert_transliterated("தமிழ்", 0x0B80, 0x0C00, "Tamil")


def test_telugu_transliteration():
    _assert_transliterated("తెలుగు", 0x0C00, 0x0C80, "Telugu")


def test_kannada_transliteration():
    _assert_transliterated("ಕನ್ನಡ", 0x0C80, 0x0D00, "Kannada")


def test_malayalam_transliteration():
    _assert_transliterated("മലയാളം", 0x0D00, 0x0D80, "Malayalam")


def test_devanagari_canonical_ram_marketing():
    # MASTERPLAN/TASK_BREAKDOWN V1 spec: 'राम मार्केटिंग' -> 'ram marketing'
    _require_indic("canonical example")
    assert normalize_name("राम मार्केटिंग") == "ram marketing", (
        f"got {normalize_name('राम मार्केटिंग')!r}"
    )


def test_indic_dependency_available():
    # Fails loudly while indic-transliteration is absent (never a silent skip).
    assert N._HAS_INDIC, (
        "indic-transliteration is NOT installed -> Indic transliteration (V1) is "
        "disabled with a RuntimeWarning only: cross-script India pairs (22.7%) "
        "cannot be normalized. Missing dependency: indic-transliteration."
    )


def test_missing_indic_dependency_surfaces_warning():
    # Missing dep must NOT look like a successful transliteration: graceful
    # fallback (input unchanged) + one RuntimeWarning.
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out = transliterate_indic("राम")
    indic_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "indic-transliteration" in str(w.message)
    ]
    if N._HAS_INDIC:
        assert not indic_warnings, "no warning expected once dep is installed"
    else:
        assert out == "राम", (
            "graceful fallback must return the input unchanged "
            f"(got {out!r})"
        )
        assert indic_warnings, (
            "missing indic-transliteration must be surfaced as a RuntimeWarning"
        )


def test_is_non_latin_detects_script():
    assert N.is_non_latin("राम marketing") == 1
    assert N.is_non_latin("商店") == 1
    assert N.is_non_latin("acme robotics") == 0
    assert N.is_non_latin("café sarl") == 0   # é is a LATIN letter
    assert N.is_non_latin("") == 0
    assert N.is_non_latin(None) == 0


# 13. empty string ------------------------------------------------------------
def test_empty_string():
    assert normalize_name("") == ""
    assert normalize_address("") == ""
    assert transliterate_indic("") == ""


# 14. None ---------------------------------------------------------------------
def test_none():
    assert normalize_name(None) == ""
    assert normalize_address(None) == ""


# 15. whitespace ---------------------------------------------------------------
def test_whitespace():
    assert normalize_name("   ") == ""
    assert normalize_address("   ") == ""
    assert normalize_name(" \t \n ") == ""


# 16. punctuation ---------------------------------------------------------------
def test_punctuation():
    assert normalize_name("A & B, Inc.") == "a and b"
    assert normalize_address("500 Market St., San Jose") == "500 market st san jose"


# 17. deterministic repeated calls ---------------------------------------------
def test_deterministic_repeated_calls():
    samples = [
        "acme robotics",
        "Café de la Paix SARL",
        "A & B, Inc.",
        "500 Market Street, San Jose",
        "राम मार्केटिंग",
        "",
    ]
    for s in samples:
        assert normalize_name(s) == normalize_name(s), f"normalize_name non-deterministic for {s!r}"
        assert normalize_address(s) == normalize_address(s), f"normalize_address non-deterministic for {s!r}"


# 18. raw input is not mutated --------------------------------------------------
def test_apply_normalization_preserves_raw():
    import pandas as pd

    df = pd.DataFrame({
        "business_name": ["Café SARL"],
        "business_address": ["1 Main St"],
        "country": ["France"],
    })
    out = apply_normalization(df)

    assert out is not df
    assert "business_name_clean" not in df.columns, "input dataframe was mutated"
    assert df["business_name"].iloc[0] == "Café SARL", "raw value was mutated"

    assert out["business_name"].iloc[0] == "Café SARL", "RAW column must be kept alongside clean"
    assert out["business_name_clean"].iloc[0] == "cafe"
    assert out["business_address_clean"].iloc[0] == "1 main st"
    assert out["country_clean"].iloc[0] == "france"


def test_apply_normalization_adds_non_latin_flags():
    import pandas as pd

    df = pd.DataFrame({
        "business_name": ["राम मार्केटिंग", "acme robotics", None],
        "business_address": ["1 मुख्य सड़क", "500 market street", float("nan")],
        "country": ["India", "US", None],
    })
    out = apply_normalization(df)

    # ADDITIVE script flags, computed from the RAW text (pre-transliteration)
    assert out["business_name_is_non_latin"].tolist() == [1, 0, 0], (
        out["business_name_is_non_latin"].tolist()
    )
    assert out["business_address_is_non_latin"].tolist() == [1, 0, 0], (
        out["business_address_is_non_latin"].tolist()
    )

    # existing columns intact, raw columns unchanged, input df untouched
    for col in ("business_name_clean", "business_address_clean", "country_clean"):
        assert col in out.columns, f"{col} disappeared"
    assert out["business_name"].iloc[0] == "राम मार्केटिंग"
    assert out["business_address"].iloc[0] == "1 मुख्य सड़क"
    assert "business_name_is_non_latin" not in df.columns, "input dataframe was mutated"


def test_apply_normalization_without_optional_columns():
    # Frames without name/address/country columns must not blow up (pipeline
    # passes several frame shapes through this function).
    import pandas as pd

    out = apply_normalization(pd.DataFrame({"entity_id": ["S1-1"]}))
    assert out["entity_id"].tolist() == ["S1-1"]
    assert "business_name_clean" not in out.columns


def main():
    print("=" * 60)
    print("BASELINE TESTS — src/normalize.py")
    print(f"import mode: {_IMPORT_MODE}")
    print("=" * 60)
    tests = sorted((k, v) for k, v in globals().items() if k.startswith("test_"))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"  {len(tests) - len(failed)}/{len(tests)} passed, {len(failed)} failed")
    if failed:
        print("\n  failed tests:")
        for name, exc in failed:
            print(f"    - {name}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
