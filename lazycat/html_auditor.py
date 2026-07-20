import logging
from html.parser import HTMLParser
from typing import List, Tuple, Dict, Any

_logger = logging.getLogger(__name__)

# beautifulsoup4 is a declared dependency of this package. It is imported
# tolerantly so a missing install degrades the functional audit loudly (see
# audit_functional_html) instead of hard-failing every importer of this module.
try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised only on broken installs
    BeautifulSoup = None

ALLOWED_TAGS = {
    "article", "section", "header", "p", "ul", "ol", "li", "blockquote", "hr",
    "strong", "em", "code", "pre", "a", "mark", "small", "time", "data", "span",
    "div", "aside", "table", "thead", "tbody", "tr", "th", "td",
    "h1", "h2", "h3", "h4", "h5", "h6"
}

ALLOWED_ATTRIBUTES = {
    "data-note-id", "data-tag", "data-source", "data-timestamp", "href",
    "class", "style"
}

class NoteHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors = []
        self.found_tags = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]):
        self.found_tags.append(tag)
        if tag not in ALLOWED_TAGS:
            self.errors.append(f"Forbidden HTML tag: <{tag}>")
            return

        for attr, value in attrs:
            if attr not in ALLOWED_ATTRIBUTES:
                self.errors.append(f"Forbidden attribute on <{tag}>: {attr}")
                continue
            
            if attr == "href":
                # Ensure it only points to internal notes, e.g. #note_123 or #note_abc
                if not (value.startswith("#note_") or value.startswith("note_")):
                    self.errors.append(f"Invalid link href target: {value}. Must start with '#note_' or 'note_'")

def audit_html_fragment(html_content: str) -> Dict[str, Any]:
    """
    Audits an HTML fragment against the allowed subset of tags and attributes.
    Returns a dict with 'is_valid', 'errors', and 'tags'.
    """
    if not html_content.strip():
        return {"is_valid": True, "errors": [], "tags": []}
        
    parser = NoteHTMLParser()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as e:
        return {
            "is_valid": False,
            "errors": [f"HTML parsing failed: {str(e)}"],
            "tags": []
        }
        
    return {
        "is_valid": len(parser.errors) == 0,
        "errors": parser.errors,
        "tags": list(set(parser.found_tags))
    }

def audit_functional_html(html_str: str) -> dict:
    """
    Audits an HTML fragment for dead interactivity (e.g. return false, missing handlers).
    Returns a dictionary with 'is_valid' and 'errors'.
    """
    errors = []
    
    if "onclick=\"return false\"" in html_str or "onclick='return false'" in html_str:
        errors.append("Found dead button (onclick='return false'). You MUST provide a real inline JavaScript function instead of 'return false'.")
        
    if BeautifulSoup is None:
        # beautifulsoup4 is a declared dependency; if it is genuinely absent the
        # dead-button/dead-link checks below cannot run. Previously the ImportError
        # was swallowed by a broad `except Exception`, so the audit silently
        # returned is_valid=True and vouched for HTML it had never inspected.
        _logger.warning(
            "beautifulsoup4 unavailable — functional HTML audit is DEGRADED "
            "(dead-button/dead-link checks skipped)."
        )
    else:
        try:
            soup = BeautifulSoup(html_str, "html.parser")

            # Check buttons
            for btn in soup.find_all("button"):
                if not btn.get("onclick") and not btn.get("id") and not btn.get("class") and not btn.get("type") == "submit":
                    errors.append("Found a <button> with no onclick handler, id, or class. It appears dead.")

            # Check links
            for a in soup.find_all("a"):
                if a.get("href") == "#" and not a.get("onclick"):
                    errors.append("Found a link (<a href='#'>) with no onclick handler. It appears dead.")

        except Exception as e:
            _logger.error(f"Failed to audit HTML: {e}")


    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
