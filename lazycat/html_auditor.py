from html.parser import HTMLParser
from typing import List, Tuple, Dict, Any

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
