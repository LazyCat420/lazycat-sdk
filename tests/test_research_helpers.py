from lazycat.research import _extract_json


def test_extract_json_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_prose_and_fence():
    assert _extract_json('Here you go: {"title": "t", "n": {"x": 2}} done') == {
        "title": "t",
        "n": {"x": 2},
    }
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_none_for_no_json_or_empty():
    assert _extract_json("no json here") is None
    assert _extract_json("") is None
