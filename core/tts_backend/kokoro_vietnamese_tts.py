import os
import threading

from core.utils.config_utils import load_key

_KOKORO_CLIENT = None
_KOKORO_CLIENT_KEY = None
_KOKORO_LOCK = threading.Lock()


KOKORO_VOICES = {
    "diem_trinh": "Diem Trinh - female",
    "hung_thinh": "Hung Thinh - male",
    "mai_linh": "Mai Linh - female",
    "mai_loan": "Mai Loan - female",
    "manh_dung": "Manh Dung - male",
    "my_yen": "My Yen - female",
    "ngoc_huyen": "Ngoc Huyen - female",
    "phat_tai": "Phat Tai - male",
    "thanh_dat": "Thanh Dat - male",
    "thuc_trinh": "Thuc Trinh - female",
    "tuan_ngoc": "Tuan Ngoc - male",
    "storyvert": "Storyvert",
    "duc_an": "Duc An - male",
    "duc_duy": "Duc Duy - male",
}


def _clean_optional(value, default=None):
    value = default if value is None else value
    value = "" if value is None else str(value).strip()
    return value or default


def _resolve_device(device):
    device = _clean_optional(device, "cuda")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _load_kokoro_client():
    global _KOKORO_CLIENT, _KOKORO_CLIENT_KEY

    try:
        from kokoro_vietnamese import KokoroVietnamese
    except ImportError as exc:
        raise ImportError(
            "Kokoro Vietnamese TTS is not installed. Install it in this environment with:\n"
            ".\\.venv\\Scripts\\python.exe -m pip install git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git"
        ) from exc

    settings = load_key("kokoro_vietnamese_tts")
    voice = _clean_optional(settings.get("voice"), "diem_trinh")
    if voice not in KOKORO_VOICES:
        voice = "diem_trinh"
    device = _resolve_device(settings.get("device", "cuda"))

    client_key = (voice, device)
    if _KOKORO_CLIENT is not None and _KOKORO_CLIENT_KEY == client_key:
        return _KOKORO_CLIENT

    _KOKORO_CLIENT = KokoroVietnamese(device=device, voice=voice)
    _KOKORO_CLIENT_KEY = client_key
    return _KOKORO_CLIENT


def kokoro_vietnamese_tts(text, save_as, number=None, task_df=None):
    output_dir = os.path.dirname(save_as)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with _KOKORO_LOCK:
        tts = _load_kokoro_client()
        audio, _ = tts.synthesize(text)

    if len(audio) == 0:
        raise RuntimeError("Kokoro Vietnamese generated empty audio.")

    import soundfile as sf
    sf.write(save_as, audio, 24000)
