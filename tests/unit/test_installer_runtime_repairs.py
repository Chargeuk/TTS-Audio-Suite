import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INSTALL_SPEC = importlib.util.spec_from_file_location("tts_suite_install_test_module", REPO_ROOT / "install.py")
INSTALL_MODULE = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(INSTALL_MODULE)

from engines.dots_tts.dots_tts_engine import DotsTTSEngine
from utils.runtimes.profiles import get_runtime_profile


@pytest.mark.unit
def test_nested_module_check_does_not_import_parent(tmp_path, monkeypatch):
    package = tmp_path / "explosive_parent"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('parent package must not be imported')\n",
        encoding="utf-8",
    )
    (package / "runtime.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "explosive_parent", raising=False)

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer.module_available("explosive_parent.runtime") is True
    assert installer.module_available("explosive_parent.missing") is False
    assert "explosive_parent" not in sys.modules


@pytest.mark.unit
def test_fish_source_restore_accepts_namespace_package(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_package = source_root / "fish_speech"
    (source_package / "inference_engine").mkdir(parents=True)
    (source_package / "inference_engine" / "__init__.py").write_text("", encoding="utf-8")

    site_root = tmp_path / "site-packages"
    target_package = site_root / "fish_speech"
    target_package.mkdir(parents=True)
    (target_package / "content_sequence.py").write_text("", encoding="utf-8")
    # Prefer the scratch installation without hiding Python's own stdlib.
    monkeypatch.syspath_prepend(str(site_root))

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer._restore_fish_source_package(source_root) is True
    assert (target_package / "inference_engine" / "__init__.py").is_file()


@pytest.mark.unit
def test_dots_fallback_overrides_incomplete_tn_module_temporarily(monkeypatch):
    unrelated_tn = types.ModuleType("tn")
    monkeypatch.setitem(sys.modules, "tn", unrelated_tn)
    for name in (
        "tn.chinese",
        "tn.chinese.normalizer",
        "tn.english",
        "tn.english.normalizer",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with DotsTTSEngine._text_normalizer_compat():
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer

        assert ZhNormalizer().normalize("123") == "123"
        assert EnNormalizer().normalize("test") == "test"

    assert sys.modules["tn"] is unrelated_tn
    assert "tn.chinese" not in sys.modules
    assert "tn.english" not in sys.modules


@pytest.mark.unit
def test_vibevoice_runtime_avoids_webrtc_dependency_conflict():
    profile = get_runtime_profile("vibevoice_transformers4_shared")

    assert profile is not None
    assert "av" in profile.pip_packages
    assert not {
        "aiortc",
        "pyee",
        "dnspython",
        "ifaddr",
        "pylibsrtp",
        "pyopenssl",
    }.intersection(profile.pip_packages)


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile_name",
    ("vibevoice_transformers4_shared", "qwen3_tts_transformers4_dedicated"),
)
def test_legacy_transformers_runtime_pins_compatible_kernels(profile_name):
    profile = get_runtime_profile(profile_name)

    assert profile is not None
    assert "kernels>=0.6.1,<=0.9" in profile.pip_packages


@pytest.mark.unit
def test_fish_install_resolves_immutable_sources_without_temporary_provenance(monkeypatch):
    installer = INSTALL_MODULE.TTSAudioInstaller()
    commands = []
    monkeypatch.setattr(installer, "module_available", lambda name: name != "fish_speech.inference_engine")
    monkeypatch.setattr(installer, "check_package_installed", lambda name: False)
    monkeypatch.setattr(installer, "log", lambda *args: None)
    monkeypatch.setattr(installer, "run_pip_command", lambda args, *a, **kw: commands.append(args) or True)
    installer.install_fish_audio_s2()
    assert len(commands) == 1
    command = commands[0]
    assert "--no-deps" not in command
    assert any("Chargeuk/fish-speech.git@fdadb2d13e032393ec2551bff7de8b0cd63d84fe" in arg for arg in command)
    assert any("Chargeuk/audiotools.git@073ab2c1fc7a1e0cadb5d1d58d92f3410080ebff" in arg for arg in command)


@pytest.mark.unit
@pytest.mark.parametrize("method, expected_source", [
    ("install_vibevoice", "git+https://github.com/Chargeuk/VibeVoice.git@02f82d5e24597c694a01fecd09349923a7edfba1"),
    ("install_dots_tts", "git+https://github.com/Chargeuk/dots.tts.git@3abf4e73f45b94afe58e1d602dcb33fdbb4fbc9a"),
])
def test_engine_installer_resolves_compatible_immutable_source(monkeypatch, method, expected_source):
    installer = INSTALL_MODULE.TTSAudioInstaller()
    commands = []
    monkeypatch.setattr(installer, "module_available", lambda name: name != "dots_tts.runtime")
    monkeypatch.setattr(installer, "check_package_installed", lambda name: name == "av")
    monkeypatch.setattr(installer, "log", lambda *args: None)
    monkeypatch.setattr(installer, "run_pip_command", lambda args, *a, **kw: commands.append(args) or True)
    getattr(installer, method)()
    assert len(commands) == 1
    assert expected_source in commands[0]
    assert "--no-deps" not in commands[0]
