# tests/conftest.py
import os
import sys
import types
import pytest

@pytest.fixture(autouse=True)
def mock_pyhmcode():
    if os.getenv("CI") == "true":
        try:
            import pyhmcode
        except ImportError:
            dummy_module = types.ModuleType("pyhmcode")
            sys.modules["pyhmcode"] = dummy_module
            sys.modules["pyhmcode._pyhmcode"] = types.ModuleType("pyhmcode._pyhmcode")
