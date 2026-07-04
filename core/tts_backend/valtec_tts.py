import os
from pathlib import Path
from core.utils import load_key

_VALTEC_TTS_CLIENT = None
_VALTEC_ZEROSHOT_CLIENT = None


def _clean_optional(value):
    value = "" if value is None else str(value).strip()
    return value or None


def _load_valtec_clients(need_zeroshot=False):
    global _VALTEC_TTS_CLIENT, _VALTEC_ZEROSHOT_CLIENT

    try:
        from valtec_tts import TTS, ZeroShotTTS
    except ImportError as exc:
        raise ImportError(
            "Valtec-TTS is not installed. Install it in this environment with:\n"
            "pip install git+https://github.com/tronghieuit/valtec-tts.git"
        ) from exc

    if need_zeroshot:
        if _VALTEC_ZEROSHOT_CLIENT is None:
            _VALTEC_ZEROSHOT_CLIENT = ZeroShotTTS()
        return _VALTEC_ZEROSHOT_CLIENT
    else:
        if _VALTEC_TTS_CLIENT is None:
            model_path = _get_cached_multispeaker_model_path()
            _VALTEC_TTS_CLIENT = TTS(model_path=model_path) if model_path else TTS()
        return _VALTEC_TTS_CLIENT


def _get_cached_multispeaker_model_path():
    if os.name == "nt":
        cache_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    model_dir = cache_base / "valtec_tts" / "models" / "vits-vietnamese"
    if (model_dir / "config.json").exists() and ((model_dir / "G.pth").exists() or list(model_dir.glob("G_*.pth"))):
        return str(model_dir)
    return None


def valtec_tts(text, save_as, number=None, task_df=None):
    settings = load_key("valtec_tts")
    speaker = _clean_optional(settings.get("speaker", "NF"))
    ref_audio = _clean_optional(settings.get("ref_audio", ""))

    if ref_audio:
        tts = _load_valtec_clients(need_zeroshot=True)
        tts.clone_voice(text=text, reference_audio=ref_audio, output_path=save_as)
    else:
        tts = _load_valtec_clients(need_zeroshot=False)
        tts.speak(text, speaker=speaker, output_path=save_as)
