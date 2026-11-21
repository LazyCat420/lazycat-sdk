"""Regression test ensuring retired `lazycat` package fails loudly with ImportError."""

import importlib
import sys

import pytest


def test_import_lazycat_raises_retired_import_error():
    sys.modules.pop("lazycat", None)
    with pytest.raises(ImportError, match="retired"):
        importlib.import_module("lazycat")
