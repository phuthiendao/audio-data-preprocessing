"""Transcript text cleaning for Stage 0 (pre-extraction).

Pure-Python (stdlib only) so it stays fast and unit-testable without the heavy
pipeline deps. Entry points: clean_transcript(text) and is_empty_transcript(text).
"""
import re
import unicodedata

# ---- noise / artifact stripping ----
# Non-nested (...) or [...] tags. Looped in clean_transcript to catch adjacent runs.
_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_HTML = re.compile(r"<[^>]+>")
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols, pictographs, emoticons, supplemental
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002B00-\U00002BFF"   # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "]+",
    flags=re.UNICODE,
)

_QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "‟": '"',  # “ ” „ ‟
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # ‘ ’ ‚ ‛
    "«": '"', "»": '"',                                # « »
    "″": '"', "′": "'",                                # ″ ′
}


def _normalize_quotes(text):
    for k, v in _QUOTE_MAP.items():
        text = text.replace(k, v)
    return text


# ---- Vietnamese tone-placement normalization: new style -> traditional ----
# Move the tone mark from the 2nd vowel onto the 1st in the OPEN diphthongs
# oa / oe / uy, e.g. "hoà"->"hòa", "khoẻ"->"khỏe", "thuỷ"->"thủy".
# Only when the cluster ends the syllable ((?!\w) trailing boundary), so closed
# syllables like "toàn" / "ngoài" keep their tone on the last vowel. 'uy' after
# 'q' is left alone (quỳ stays quỳ, quý stays quý).
_OA_SHIFT = {"à": "òa", "á": "óa", "ả": "ỏa", "ã": "õa", "ạ": "ọa"}
_OE_SHIFT = {"è": "òe", "é": "óe", "ẻ": "ỏe", "ẽ": "õe", "ẹ": "ọe"}
_UY_SHIFT = {"ỳ": "ùy", "ý": "úy", "ỷ": "ủy", "ỹ": "ũy", "ỵ": "ụy"}

_OA_RE = re.compile(r"o([àáảãạ])(?!\w)")
_OE_RE = re.compile(r"o([èéẻẽẹ])(?!\w)")
_UY_RE = re.compile(r"(?<![qQ])u([ỳýỷỹỵ])(?!\w)")


def normalize_diacritics(text):
    text = unicodedata.normalize("NFC", text)
    text = _OA_RE.sub(lambda m: _OA_SHIFT[m.group(1)], text)
    text = _OE_RE.sub(lambda m: _OE_SHIFT[m.group(1)], text)
    text = _UY_RE.sub(lambda m: _UY_SHIFT[m.group(1)], text)
    return text


# ---- punctuation cleanup ----
_REPEAT_PUNCT = re.compile(r"([.!?,;:])\1+")


def collapse_punctuation(text):
    text = text.replace("…", ".")   # … -> .
    text = _REPEAT_PUNCT.sub(r"\1", text)  # ??? -> ?, !!! -> !, ... -> .
    return text


_PAREN_TAG = re.compile(r"\(([^()]+)\)")


def clean_transcript(text, strip_brackets=True):
    """Full Stage-1 cleaning of one transcript string.
    
    If strip_brackets=True (default), non-nested [...] and (...) tags are stripped.
    If strip_brackets=False, bracketed tags are preserved and standardized inside square brackets [...] (e.g. (cười) -> [cười]).
    """
    if not text:
        return ""
    text = _normalize_quotes(text.strip())
    text = _HTML.sub(" ", text)
    text = _EMOJI.sub("", text)
    if strip_brackets:
        prev = None
        while prev != text:            # loop to strip adjacent/leftover bracket runs
            prev = text
            text = _BRACKETS.sub(" ", text)
    else:
        # Convert any (tag) into [tag] for a single unified [...] tag format
        text = _PAREN_TAG.sub(r"[\1]", text)
    text = normalize_diacritics(text)
    text = collapse_punctuation(text)
    return re.sub(r"\s+", " ", text).strip()


def is_empty_transcript(text):
    """True if no word characters remain (only punctuation/whitespace, or blank)."""
    return not re.search(r"\w", text or "", re.UNICODE)
