import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import load_key

_VIENEU_CLIENT = None
_VIENEU_CLIENT_KEY = None
_VIENEU_REF_CACHE = {}
_VIENEU_VOICE_CACHE = {}
_BATCH_ENGINE = None


def _clean_optional(value):
    value = "" if value is None else str(value).strip()
    return value or None


def _clean_int(value, default, minimum=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, value) if minimum is not None else value


def _clean_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default if value is None else bool(value)


def _is_cuda_oom(error):
    message = str(error).lower()
    return "out of memory" in message or ("cuda error" in message and "memory" in message)


def _load_vieneu_client():
    global _VIENEU_CLIENT, _VIENEU_CLIENT_KEY

    try:
        from vieneu import Vieneu
    except ImportError as exc:
        raise ImportError(
            "VieNeu-TTS is not installed. Install it in this environment with: "
            f'"{sys.executable}" -m pip install vieneu'
        ) from exc

    settings = load_key("vieneu_tts")
    mode = _clean_optional(settings.get("mode", "v3turbo"))
    api_base = _clean_optional(settings.get("api_base", ""))
    model_name = _clean_optional(settings.get("model_name", ""))
    emotion = _clean_optional(settings.get("emotion", "natural"))

    client_key = (mode, api_base, model_name, emotion)
    if _VIENEU_CLIENT is not None and _VIENEU_CLIENT_KEY == client_key:
        return _VIENEU_CLIENT

    kwargs = {}
    if mode:
        kwargs["mode"] = mode
    if api_base:
        kwargs["api_base"] = api_base
    if model_name:
        kwargs["model_name"] = model_name
    if emotion:
        kwargs["emotion"] = emotion

    _VIENEU_CLIENT = Vieneu(**kwargs)
    _VIENEU_CLIENT_KEY = client_key
    return _VIENEU_CLIENT


def _normalize_voice(voice_name, available_voices):
    if not voice_name:
        return voice_name

    for voice in available_voices:
        if voice.lower() == voice_name.lower():
            return voice

    def strip_accents(text):
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )

    stripped_input = strip_accents(voice_name.lower())
    for voice in available_voices:
        if strip_accents(voice.lower()) == stripped_input:
            return voice

    return voice_name


def _get_cached_ref_codes(tts, ref_audio):
    if not ref_audio or not hasattr(tts, "encode_reference"):
        return None

    cache_key = (id(tts), ref_audio)
    if cache_key not in _VIENEU_REF_CACHE:
        _VIENEU_REF_CACHE[cache_key] = tts.encode_reference(ref_audio)
    return _VIENEU_REF_CACHE[cache_key]


def _get_cached_preset_voice(tts, voice):
    if not voice or not hasattr(tts, "get_preset_voice"):
        return voice

    cache_key = (id(tts), voice)
    if cache_key not in _VIENEU_VOICE_CACHE:
        _VIENEU_VOICE_CACHE[cache_key] = tts.get_preset_voice(voice)
    return _VIENEU_VOICE_CACHE[cache_key]


def vieneu_tts(text, save_as, number=None, task_df=None):
    settings = load_key("vieneu_tts")
    voice = _clean_optional(settings.get("voice", "Binh An"))
    ref_audio = _clean_optional(settings.get("ref_audio", ""))
    ref_text = _clean_optional(settings.get("ref_text", ""))
    emotion = _clean_optional(settings.get("emotion", "natural"))

    tts = _load_vieneu_client()
    infer_kwargs = {"text": text}
    if emotion:
        infer_kwargs["emotion"] = emotion
    if ref_audio:
        ref_codes = _get_cached_ref_codes(tts, ref_audio)
        if ref_codes is not None:
            infer_kwargs["ref_codes"] = ref_codes
        else:
            infer_kwargs["ref_audio"] = ref_audio
        if ref_text:
            infer_kwargs["ref_text"] = ref_text
    elif voice:
        if hasattr(tts, "_preset_voices"):
            voice = _normalize_voice(voice, list(tts._preset_voices))
        infer_kwargs["voice"] = _get_cached_preset_voice(tts, voice)
    infer_kwargs["apply_watermark"] = False

    audio = tts.infer(**infer_kwargs)
    tts.save(audio, save_as)


def _get_batch_engine(tts):
    global _BATCH_ENGINE
    if _BATCH_ENGINE is not None:
        return _BATCH_ENGINE

    engine_obj = getattr(tts, "engine", None)
    if getattr(tts, "backend", None) == "pytorch" and engine_obj is not None:
        device = getattr(engine_obj, "device", None)
        if device is not None and "cuda" in str(device).lower():
            try:
                from vieneu.v3_turbo_serve import V3TurboBatchEngine
                _BATCH_ENGINE = V3TurboBatchEngine(engine_obj)
            except Exception as exc:
                import warnings
                warnings.warn(f"Failed to load V3TurboBatchEngine: {exc}")
                _BATCH_ENGINE = False
        else:
            _BATCH_ENGINE = False
    else:
        _BATCH_ENGINE = False

    return _BATCH_ENGINE


def comes_to_batch_viable():
    try:
        settings = load_key("vieneu_tts")
        mode = _clean_optional(settings.get("mode", "v3turbo"))
        if mode != "v3turbo":
            return False

        api_base = _clean_optional(settings.get("api_base", ""))
        if api_base:
            return False

        tts = _load_vieneu_client()
        if getattr(tts, "backend", None) != "pytorch":
            return False

        engine_obj = getattr(tts, "engine", None)
        if engine_obj is None:
            return False
        device = getattr(engine_obj, "device", None)
        if device is None or "cuda" not in str(device).lower():
            return False

        return bool(_get_batch_engine(tts))
    except Exception:
        return False


def vieneu_tts_batch(items, batch_size=4, progress_callback=None):
    try:
        tts = _load_vieneu_client()
        engine = _get_batch_engine(tts)
    except Exception as exc:
        print(f"Failed to initialize batch engine: {exc}. Falling back to single-path.", file=sys.stderr)
        for item in items:
            _fallback_item(item, progress_callback)
        return

    if not engine:
        for item in items:
            _fallback_item(item, progress_callback)
        return

    try:
        settings = load_key("vieneu_tts")
        voice = _clean_optional(settings.get("voice", "Binh An"))
        ref_audio = _clean_optional(settings.get("ref_audio", ""))
        ref_text = _clean_optional(settings.get("ref_text", ""))
        emotion = _clean_optional(settings.get("emotion", "natural"))
        batch_size = _clean_int(settings.get("batch_size", batch_size), batch_size, minimum=1)
        temperature = _clean_float(settings.get("temperature", 0.8), 0.8)
        top_k = _clean_int(settings.get("top_k", 25), 25, minimum=1)
        top_p = _clean_float(settings.get("top_p", 0.95), 0.95)
        repetition_penalty = _clean_float(settings.get("repetition_penalty", 1.0), 1.0)
        max_new_frames = _clean_int(settings.get("max_new_frames", 300), 300, minimum=1)
        use_cudagraph = _clean_bool(settings.get("use_cudagraph", True), True)

        ref_codes = _get_cached_ref_codes(tts, ref_audio) if ref_audio else None

        voice_data = None
        if voice and not ref_audio:
            if hasattr(tts, "_preset_voices"):
                voice = _normalize_voice(voice, list(tts._preset_voices))
            voice_data = _get_cached_preset_voice(tts, voice)

        resolved_ref_codes, voice_token_id = tts._resolve_v3_ref(
            voice=voice_data,
            ref_audio=None if ref_codes is not None else ref_audio,
            ref_codes=ref_codes
        )
    except Exception as exc:
        print(f"Failed to resolve voice reference: {exc}. Falling back to single-path.", file=sys.stderr)
        for item in items:
            _fallback_item(item, progress_callback)
        return

    try:
        from vieneu.v3turbo import normalize_to_chunks_v3, phonemize_text_with_emotions
    except ImportError as exc:
        print(f"Failed to import chunk/phoneme utils: {exc}. Falling back to single-path.", file=sys.stderr)
        for item in items:
            _fallback_item(item, progress_callback)
        return

    try:
        preprocess_workers = _clean_int(load_key("tts_max_workers"), 4, minimum=1)
    except Exception:
        preprocess_workers = 4

    def prepare_item(item):
        if os.path.exists(item["save_as"]):
            return "skip", item, None, None
        try:
            chunks = normalize_to_chunks_v3(item["text"])
        except Exception:
            chunks = []

        if len(chunks) != 1:
            return "fallback", item, None, f"Text normalized to {len(chunks)} chunk(s)"

        try:
            chunk_text = chunks[0]
            phonemes = phonemize_text_with_emotions(chunk_text)
            return "batch", item, {
                "text": chunk_text,
                "phonemes": phonemes,
                "ref_codes": resolved_ref_codes,
                "voice_token_id": voice_token_id,
                "emotion": emotion,
            }, None
        except Exception as exc:
            return "fallback", item, None, f"Phonemization error: {exc}"

    batch_reqs = []
    batch_items = []
    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        futures = [executor.submit(prepare_item, item) for item in items]
        for future in as_completed(futures):
            status, item, req, reason = future.result()
            if status == "skip":
                if progress_callback:
                    progress_callback(1)
            elif status == "batch":
                batch_reqs.append(req)
                batch_items.append(item)
            else:
                _fallback_item(item, progress_callback, reason)

    paired = sorted(
        zip(batch_reqs, batch_items),
        key=lambda pair: len(pair[0].get("phonemes") or pair[0].get("text") or "")
    )
    if not paired:
        return
    batch_reqs, batch_items = map(list, zip(*paired))

    def generate_and_save(req_chunk, item_chunk, allow_graph=True):
        try:
            wavs = engine.generate_batch(
                req_chunk,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_frames=max_new_frames,
                use_cudagraph=use_cudagraph and allow_graph,
            )
            for item, wav in zip(item_chunk, wavs):
                try:
                    tts.save(wav, item["save_as"])
                except Exception as exc:
                    print(f"Error saving wav: {exc}. Re-generating with single path.", file=sys.stderr)
                    vieneu_tts(item["text"], item["save_as"])
            if progress_callback:
                progress_callback(len(item_chunk))
        except Exception as exc:
            if use_cudagraph and allow_graph:
                print(f"CUDA graph batch generation error: {exc}. Retrying without CUDA graph.", file=sys.stderr)
                generate_and_save(req_chunk, item_chunk, allow_graph=False)
                return
            if _is_cuda_oom(exc) and len(req_chunk) > 1:
                midpoint = max(1, len(req_chunk) // 2)
                print(f"CUDA memory pressure at batch size {len(req_chunk)}. Splitting batch.", file=sys.stderr)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                generate_and_save(req_chunk[:midpoint], item_chunk[:midpoint], allow_graph=False)
                generate_and_save(req_chunk[midpoint:], item_chunk[midpoint:], allow_graph=False)
                return
            print(f"Batch generation error: {exc}. Falling back to single path for batch.", file=sys.stderr)
            for item in item_chunk:
                _fallback_item(item, progress_callback)

    for i in range(0, len(batch_reqs), batch_size):
        generate_and_save(batch_reqs[i:i + batch_size], batch_items[i:i + batch_size])


def _fallback_item(item, progress_callback=None, reason=None):
    if reason:
        print(f"{reason}. Falling back to single path for text {item['text']}.", file=sys.stderr)
    try:
        vieneu_tts(item["text"], item["save_as"])
    except Exception as exc:
        print(f"Fallback generation error for text {item['text']}: {exc}", file=sys.stderr)
    if progress_callback:
        progress_callback(1)
