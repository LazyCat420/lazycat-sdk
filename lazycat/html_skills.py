import logging
from bs4 import BeautifulSoup
from typing import Dict, Any

logger = logging.getLogger(__name__)

def canvas_read_dom(canvas_html: str) -> str:
    """Returns the current DOM structure of the canvas."""
    return f"Current Canvas HTML:\n{canvas_html}"

def canvas_modify_dom(canvas_html: str, css_selector: str, action: str, html_snippet: str = "") -> Dict[str, Any]:
    """Modifies the DOM based on a CSS selector and action."""
    try:
        soup = BeautifulSoup(canvas_html, "html.parser")
        target = soup.select_one(css_selector)
        if not target:
            return {"error": f"Selector '{css_selector}' not found", "success": False}
            
        snippet_soup = BeautifulSoup(html_snippet, "html.parser")
        if action == "append":
            target.append(snippet_soup)
        elif action == "prepend":
            target.insert(0, snippet_soup)
        elif action == "insert_before":
            target.insert_before(snippet_soup)
        elif action == "insert_after":
            target.insert_after(snippet_soup)
        elif action == "replace":
            target.replace_with(snippet_soup)
        elif action == "remove":
            target.decompose()
        else:
            return {"error": f"Unknown action '{action}'", "success": False}
            
        return {"success": True, "rendered_html": str(soup)}
    except Exception as e:
        logger.error(f"DOM modification failed: {e}")
        return {"error": str(e), "success": False}

def render_component(component_type: str, title: str, data: Dict[str, Any] = None, rendered_html: str = "") -> Dict[str, Any]:
    """
    Generic render_component tool.
    In a real app, you would pass this to a template engine.
    For the SDK, it simply formats the response so the frontend can catch it.
    """
    return {
        "success": True,
        "component_type": component_type,
        "title": title,
        "data": data or {},
        "rendered_html": rendered_html
    }
