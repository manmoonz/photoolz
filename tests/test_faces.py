from __future__ import annotations

import builtins
import sys

import pytest

from photoolz.indexer import faces

_real_import = builtins.__import__


class _Quit(SystemExit):
    """Mimics face_recognition's import-time failure: print to stdout, then quit()."""

    def __init__(self):
        print("Please install `face_recognition_models` ...")
        super().__init__(None)


def _patch_imports(monkeypatch, failures: dict[str, type[BaseException] | BaseException]):
    def fake_import(name, *args, **kwargs):
        if name in failures:
            exc = failures[name]
            raise exc() if isinstance(exc, type) else exc
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "face_recognition", object())
    assert faces.face_recognition_unavailable_reason() is None


def test_not_installed(monkeypatch):
    _patch_imports(monkeypatch, {"face_recognition": ModuleNotFoundError(name="face_recognition")})
    assert faces.face_recognition_unavailable_reason() == "pip install face_recognition"


@pytest.mark.parametrize("models_error, hint", [
    (ModuleNotFoundError(name="pkg_resources"), "pip install 'setuptools<81'"),
    (ModuleNotFoundError(name="face_recognition_models"), "pip install face_recognition_models"),
    (OSError("model file missing"), "fix face_recognition_models, which failed to import: model file missing"),
])
def test_quit_on_models_failure_is_caught_and_diagnosed(monkeypatch, capsys, models_error, hint):
    _patch_imports(monkeypatch, {"face_recognition": _Quit, "face_recognition_models": models_error})
    assert faces.face_recognition_unavailable_reason() == hint
    assert "Please install" not in capsys.readouterr().out
