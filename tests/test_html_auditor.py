from lazycat.html_auditor import audit_html_fragment, audit_functional_html


class TestAuditHtmlFragment:
    def test_valid_fragment_passes(self):
        html = (
            '<article><header><p data-timestamp="2026-01-01T00:00:00Z">Title</p></header>'
            '<section><p>Some <strong>bold</strong> text.</p>'
            '<a href="#note_123">link</a></section></article>'
        )
        result = audit_html_fragment(html)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert "article" in result["tags"]

    def test_empty_fragment_is_valid(self):
        result = audit_html_fragment("   ")
        assert result["is_valid"] is True
        assert result["tags"] == []

    def test_forbidden_tag_rejected(self):
        result = audit_html_fragment("<article><script>alert(1)</script></article>")
        assert result["is_valid"] is False
        assert any("Forbidden HTML tag: <script>" in e for e in result["errors"])

    def test_forbidden_attribute_rejected(self):
        result = audit_html_fragment('<p onclick="alert(1)">hi</p>')
        assert result["is_valid"] is False
        assert any("Forbidden attribute" in e for e in result["errors"])

    def test_external_href_rejected(self):
        result = audit_html_fragment('<a href="https://evil.example">out</a>')
        assert result["is_valid"] is False
        assert any("Invalid link href target" in e for e in result["errors"])

    def test_internal_note_hrefs_allowed(self):
        assert audit_html_fragment('<a href="#note_1">a</a>')["is_valid"] is True
        assert audit_html_fragment('<a href="note_1">a</a>')["is_valid"] is True


class TestAuditFunctionalHtml:
    def test_dead_onclick_return_false_flagged(self):
        result = audit_functional_html('<button onclick="return false">Go</button>')
        assert result["is_valid"] is False
        assert any("dead button" in e.lower() for e in result["errors"])

    def test_bare_button_flagged(self):
        result = audit_functional_html("<div><button>Do nothing</button></div>")
        assert result["is_valid"] is False
        assert any("<button>" in e for e in result["errors"])

    def test_dead_anchor_flagged(self):
        result = audit_functional_html('<a href="#">dead link</a>')
        assert result["is_valid"] is False
        assert any("appears dead" in e for e in result["errors"])

    def test_wired_button_passes(self):
        result = audit_functional_html('<button onclick="doThing()">Go</button>')
        assert result["is_valid"] is True
        assert result["errors"] == []
