import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "utils" / "runtimes" / "workers" / "fish_audio_s2_worker.py"
SPEC = importlib.util.spec_from_file_location("fish_audio_s2_worker_test_module", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER_MODULE)


class FakeReferenceLoader:
    def __init__(self):
        self.ref_by_hash = {}
        self.encoded = []

    def encode_reference(self, reference_audio, enable_reference_audio):
        assert enable_reference_audio is True
        token = f"encoded-{len(self.encoded)}"
        self.encoded.append(reference_audio)
        return token


@pytest.mark.unit
def test_reference_cache_reuses_audio_but_not_previous_speaker_tag():
    loader = FakeReferenceLoader()

    first_tokens, first_texts = WORKER_MODULE._load_references_preserving_current_speaker_tags(
        loader,
        [SimpleNamespace(audio=b"joe", text="<|speaker:0|>Joe reference")],
        "on",
    )
    second_tokens, second_texts = WORKER_MODULE._load_references_preserving_current_speaker_tags(
        loader,
        [
            SimpleNamespace(audio=b"bob", text="<|speaker:0|>Bob reference"),
            SimpleNamespace(audio=b"joe", text="<|speaker:1|>Joe reference"),
        ],
        "on",
    )

    assert first_texts == ["<|speaker:0|>Joe reference"]
    assert second_tokens == ["encoded-1", first_tokens[0]]
    assert second_texts == ["<|speaker:0|>Bob reference", "<|speaker:1|>Joe reference"]


@pytest.mark.unit
def test_removed_torchaudio_backend_probe_is_restored_temporarily(monkeypatch):
    fake_torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    WORKER_MODULE._patch_removed_torchaudio_backend_probe()

    assert fake_torchaudio.list_audio_backends() == []


@pytest.mark.unit
@pytest.mark.parametrize("variant,quantization", [("s2-pro-fp8", "none"), ("s2-pro", "bnb_int8")])
def test_quantized_s2_loaders_accept_fish3_queue_contract(monkeypatch, variant, quantization):
    model = SimpleNamespace(to=lambda **kwargs: model, eval=lambda: model)
    packages = {}
    for name in ("fish_speech", "fish_speech.models", "fish_speech.models.text2semantic"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
        monkeypatch.setitem(sys.modules, name, package)
        parent, _, attr = name.rpartition(".")
        if parent:
            setattr(packages[parent], attr, package)
    llama = types.ModuleType("fish_speech.models.text2semantic.llama")
    original_from_pretrained = lambda *args, **kwargs: model
    llama.DualARTransformer = type("DualARTransformer", (), {"from_pretrained": staticmethod(original_from_pretrained)})
    inference = types.ModuleType("fish_speech.models.text2semantic.inference")
    original_init = lambda *args, **kwargs: None
    inference.init_model = original_init
    inference.decode_one_token_ar = object()
    def launch(**kwargs):
        loaded, _ = inference.init_model(kwargs["checkpoint_path"], kwargs["device"], kwargs["precision"], compile=kwargs["compile"], reference_sampling=None)
        return loaded
    inference.launch_thread_safe_queue = launch
    for module in (llama, inference):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(packages["fish_speech.models.text2semantic"], module.__name__.rsplit(".", 1)[-1], module)
    fp8 = types.ModuleType("utils.runtimes.fish_audio_s2_fp8")
    fp8.load_fp8_model = lambda *args: model
    bnb = types.ModuleType("utils.runtimes.fish_audio_s2_bnb")
    bnb.quantize_linear_layers = lambda source, *args: (source, 1)
    monkeypatch.setitem(sys.modules, fp8.__name__, fp8)
    monkeypatch.setitem(sys.modules, bnb.__name__, bnb)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 9)), tensor=lambda *args, **kwargs: object(), float=object())
    monkeypatch.setattr(WORKER_MODULE, "torch", fake_torch)
    assert WORKER_MODULE._launch_llama_queue("weights", variant, quantization, "cuda", "bf16", 8192, False) is model
    assert inference.init_model is original_init
    assert llama.DualARTransformer.from_pretrained is original_from_pretrained
