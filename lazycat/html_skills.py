import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

COMPONENT_SKILL_LIBRARY = """
## Component Skill Library
When generating HTML components, use the following self-contained patterns to ensure full interactivity without relying on external libraries or missing event handlers.

### 1. Sortable Table
To create a table that can be sorted by clicking headers, embed the script directly:
```html
<table id="task-table-1" class="data-table">
  <thead>
    <tr>
      <th style="cursor:pointer;" onclick="sortTable('task-table-1', 0)">Task ↕</th>
      <th style="cursor:pointer;" onclick="sortTable('task-table-1', 1)">Status ↕</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Buy Milk</td><td>Pending</td></tr>
    <tr><td>Call Bob</td><td>Done</td></tr>
  </tbody>
</table>
<script>
function sortTable(tableId, col) {
  var table = document.getElementById(tableId);
  var rows = Array.from(table.rows).slice(1);
  var dir = table.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';
  table.setAttribute('data-dir', dir);
  rows.sort((a, b) => {
    var v1 = a.cells[col].innerText;
    var v2 = b.cells[col].innerText;
    return dir === 'asc' ? v1.localeCompare(v2) : v2.localeCompare(v1);
  });
  rows.forEach(r => table.tBodies[0].appendChild(r));
}
</script>
```

### 2. Interactive Calendar
To allow navigating months in a calendar:
```html
<div id="cal-widget" class="calendar-widget">
  <div class="header">
    <button onclick="changeMonth(-1)">Prev</button>
    <span id="cal-month-label">June 2026</span>
    <button onclick="changeMonth(1)">Next</button>
  </div>
  <div id="cal-grid" class="calendar-grid">...</div>
</div>
<script>
function changeMonth(offset) {
  // Logic to re-render calendar grid goes here
  document.getElementById('cal-month-label').innerText = "July 2026";
}
</script>
```

### 3. Tabbed Layout
```html
<div class="tab-container" id="tabs-1">
  <div class="tab-headers">
    <button onclick="switchTab('tabs-1', 'tab-a')">Tab A</button>
    <button onclick="switchTab('tabs-1', 'tab-b')">Tab B</button>
  </div>
  <div id="tab-a" class="tab-content" style="display:block;">Content A</div>
  <div id="tab-b" class="tab-content" style="display:none;">Content B</div>
</div>
<script>
function switchTab(containerId, tabId) {
  var container = document.getElementById(containerId);
  var contents = container.getElementsByClassName('tab-content');
  for(var i=0; i<contents.length; i++) {
    contents[i].style.display = 'none';
  }
  document.getElementById(tabId).style.display = 'block';
}
</script>
```

### 4. Search/Filter Input
```html
<input type="text" id="filter-input-1" onkeyup="filterList('filter-input-1', 'list-1')" placeholder="Search...">
<ul id="list-1">
  <li>Apple</li>
  <li>Banana</li>
</ul>
<script>
function filterList(inputId, listId) {
  var filter = document.getElementById(inputId).value.toLowerCase();
  var li = document.getElementById(listId).getElementsByTagName('li');
  for (var i = 0; i < li.length; i++) {
    var text = li[i].innerText.toLowerCase();
    li[i].style.display = text.indexOf(filter) > -1 ? "" : "none";
  }
}
</script>
```

### 5. Modal / Popup
```html
<button onclick="document.getElementById('modal-1').style.display='block'">Open Modal</button>
<div id="modal-1" style="display:none; position:fixed; top:20%; left:20%; background:#fff; padding:20px;">
  <p>Modal Content</p>
  <button onclick="document.getElementById('modal-1').style.display='none'">Close</button>
</div>
```
"""

def canvas_read_dom(canvas_html: str) -> str:
    """Returns the current DOM structure of the canvas."""
    return f"Current Canvas HTML:\n{canvas_html}"

def canvas_modify_dom(canvas_html: str, css_selector: str, action: str, html_snippet: str = "") -> Dict[str, Any]:
    """Modifies the DOM based on a CSS selector and action."""
    try:
        # Lazy import (like html_auditor) so importing this module doesn't
        # require beautifulsoup4 in environments that never call it.
        from bs4 import BeautifulSoup

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
