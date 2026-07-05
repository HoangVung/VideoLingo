import streamlit as st
import requests
import os
import json
from pathlib import Path
from translations.translations import translate as t
from core.utils import *
from core.tts_backend.kokoro_vietnamese_tts import KOKORO_VOICES

VIENEU_MODES = {
    "v3turbo": "Local v3 Turbo",
    "remote": "Remote API",
}

VIENEU_EMOTIONS = {
    "natural": "Natural",
    "storytelling": "Storytelling",
}

VALTEC_SPEAKERS = {
    "NF": "NF - Nữ miền Bắc",
    "SF": "SF - Nữ miền Nam",
    "NM1": "NM1 - Nam miền Bắc 1",
    "SM": "SM - Nam miền Nam",
    "NM2": "NM2 - Nam miền Bắc 2",
}

def _save_uploaded_reference_audio(uploaded_file):
    upload_dir = os.path.join("output", "audio", "reference_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(uploaded_file.name).replace(" ", "_")
    save_path = os.path.join(upload_dir, filename)
    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return save_path


def _safe_update_key(key, value):
    try:
        update_key(key, value)
        return True
    except Exception:
        return False


def _glossary_settings():
    from core.glossary_utils import (
        build_custom_terms_from_glossary,
        get_glossary_config,
        validate_glossary,
    )

    cfg = get_glossary_config()
    with st.expander(t("Glossary Settings"), expanded=False):
        enabled = st.toggle(t("Enable glossary"), value=bool(cfg.get("enabled", False)))
        if enabled != cfg.get("enabled", False):
            _safe_update_key("glossary.enabled", enabled)
            st.rerun()

        glossary_path = st.text_input(
            t("Glossary JSON path"),
            value=str(cfg.get("path") or "glossaries/olympiad_math_zh_vi_glossary.json"),
        )
        if glossary_path != cfg.get("path"):
            _safe_update_key("glossary.path", glossary_path)
            st.rerun()

        uploaded = st.file_uploader(t("Upload glossary JSON"), type=["json"])
        if uploaded is not None:
            # Track which file was already processed to avoid infinite rerun loop.
            # st.file_uploader retains the file after st.rerun(), so without this
            # guard the block fires on every rerun → flicker.
            upload_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_glossary_last_upload_id") != upload_id:
                os.makedirs("glossaries", exist_ok=True)
                save_path = os.path.join("glossaries", uploaded.name)
                with open(save_path, "wb") as file:
                    file.write(uploaded.getbuffer())
                _safe_update_key("glossary.path", save_path)
                st.session_state["_glossary_last_upload_id"] = upload_id
                st.success(t("Glossary uploaded."))
                st.rerun()

        auto_normalize = st.toggle(
            t("Auto normalize source transcript"),
            value=bool(cfg.get("auto_normalize_source", True)),
        )
        if auto_normalize != cfg.get("auto_normalize_source", True):
            _safe_update_key("glossary.auto_normalize_source", auto_normalize)
            st.rerun()

        auto_build = st.toggle(
            t("Auto build custom terms"),
            value=bool(cfg.get("auto_build_custom_terms", True)),
        )
        if auto_build != cfg.get("auto_build_custom_terms", True):
            _safe_update_key("glossary.auto_build_custom_terms", auto_build)
            st.rerun()

        max_terms = st.number_input(
            t("Max glossary terms"),
            min_value=20,
            max_value=500,
            value=int(cfg.get("max_terms", 120)),
            step=1,
        )
        if int(max_terms) != int(cfg.get("max_terms", 120)):
            _safe_update_key("glossary.max_terms", int(max_terms))
            st.rerun()

        always_include_asr = st.toggle(
            t("Always include ASR/OCR correction terms"),
            value=bool(cfg.get("always_include_asr", True)),
        )
        if always_include_asr != cfg.get("always_include_asr", True):
            _safe_update_key("glossary.always_include_asr", always_include_asr)
            st.rerun()

        if st.button(t("Validate glossary"), use_container_width=True):
            validation = validate_glossary(glossary_path)
            if validation.get("ok"):
                st.success(t("Glossary is valid."))
                st.write(
                    {
                        "version": validation.get("version", ""),
                        "domains": validation.get("domain_count", 0),
                        "terms": validation.get("total_terms", 0),
                        "has_asr_ocr_corrections": validation.get(
                            "has_asr_ocr_corrections", False
                        ),
                    }
                )
            else:
                st.warning(
                    "\n".join(validation.get("warnings", []))
                    or t("Glossary is invalid.")
                )

        if st.button(t("Build custom terms now"), use_container_width=True):
            if not os.path.exists("output/log/split_by_meaning.txt"):
                st.warning(
                    "Please run b.1-3 until sentence segmentation first, or use always-include ASR terms only."
                )
            result = build_custom_terms_from_glossary(
                glossary_path=glossary_path,
                transcript_path="output/log/split_by_meaning.txt",
                output_path="custom_terms.xlsx",
                max_terms=int(max_terms),
                always_include_asr=bool(always_include_asr),
            )
            if result.get("ok"):
                st.success(
                    f"Exported {result.get('selected_terms', 0)} terms to custom_terms.xlsx."
                )
            else:
                st.warning(
                    "\n".join(result.get("warnings", []))
                    or "No custom terms were exported."
                )

def _get_vieneu_voice_options():
    fallback = {
        "Bình An": "Bình An - nam, giọng điềm đạm",
        "Ngọc Linh": "Ngọc Linh - nữ, giọng tươi sáng",
    }
    try:
        import sys
        from pathlib import Path
        voices_path = Path(sys.prefix) / "Lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json"
        if not voices_path.exists():
            voices_path = Path(sys.prefix) / "lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json"
        if not voices_path.exists():
            voices_path = Path(r"d:\0-vung-apps\VideoLingo\.venv\Lib\site-packages\vieneu\assets\voices_v3_turbo.json")
            
        data = json.loads(voices_path.read_text(encoding="utf-8"))
        presets = data.get("presets", {})
        if not presets:
            return fallback
        return {
            name: f"{name} - {info.get('description', '').strip()}".rstrip(" -")
            for name, info in presets.items()
        }
    except Exception:
        return fallback



def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not api_key or not base_url:
        return []
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted([m["id"] for m in data if "id" in m])
    except Exception:
        return []


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def page_setting():
    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key", placeholder=t("Enter your API key"))
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", use_container_width=True
            ):
                with st.spinner(t("Fetching models...")):
                    models = _fetch_model_list(
                        load_key("api.base_url"), load_key("api.key")
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        # Update searchbox internal state directly so dropdown shows options
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

            if st.button("📡 " + t("Check API"), key="api", use_container_width=True):
                with st.spinner(t("Check API") + "..."):
                    is_valid = check_api()
                st.toast(
                    t("API Key is valid") if is_valid else t("API Key is invalid"),
                    icon="✅" if is_valid else "❌",
                )
        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    help=t("click to check API validity") + " 👉",
                    placeholder=t("Search or enter model name"),
                )
            with c2:
                if st.button("📡", key="api"):
                    is_valid = check_api()
                    st.toast(
                        t("API Key is valid") if is_valid else t("API Key is invalid"),
                        icon="✅" if is_valid else "❌",
                    )
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtime = st.selectbox(
            t("WhisperX Runtime"),
            options=["local", "cloud", "elevenlabs"],
            index=["local", "cloud", "elevenlabs"].index(load_key("whisper.runtime")),
            format_func=lambda x: {
                "local": t("Local"),
                "cloud": t("Cloud"),
                "elevenlabs": t("ElevenLabs"),
            }[x],
            help=t(
                "Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"
            ),
        )
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(t("ElevenLabs API"), "whisper.elevenlabs_api_key")

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            help=t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        reflect_translate = st.toggle(
            t("Reflect Translate (2-step)"),
            value=load_key("reflect_translate"),
            help=t(
                "Use two-step translation (direct + free) for more natural results, but costs double the API tokens."
            ),
        )
        if reflect_translate != load_key("reflect_translate"):
            update_key("reflect_translate", reflect_translate)
            st.rerun()

        from core._1_ytdlp import is_audio_only_input
        audio_only = is_audio_only_input()
        if audio_only:
            st.toggle(
                t("Burn-in Subtitles"),
                value=False,
                disabled=True,
                help=t("Audio-only input produces subtitle files only; no video is generated."),
            )
        else:
            burn_subtitles = st.toggle(
                t("Burn-in Subtitles"),
                value=load_key("burn_subtitles"),
                help=t(
                    "Whether to burn subtitles into the video, will increase processing time"
                ),
            )
            if burn_subtitles != load_key("burn_subtitles"):
                update_key("burn_subtitles", burn_subtitles)
                st.rerun()
    _glossary_settings()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = [
            "azure_tts",
            "openai_tts",
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
            "vieneu_tts",
            "valtec_tts",
            "kokoro_vietnamese_tts",
        ]
        tts_method_labels = {
            "azure_tts": t("Azure TTS"),
            "openai_tts": t("OpenAI TTS"),
            "fish_tts": t("Fish TTS"),
            "sf_fish_tts": t("SiliconFlow Fish TTS"),
            "edge_tts": t("Edge TTS"),
            "gpt_sovits": t("GPT-SoVITS"),
            "custom_tts": t("Custom TTS"),
            "sf_cosyvoice2": t("SiliconFlow CosyVoice2"),
            "f5tts": t("F5-TTS"),
            "vieneu_tts": t("VieNeu TTS"),
            "valtec_tts": t("Valtec TTS"),
            "kokoro_vietnamese_tts": t("Kokoro Vietnamese TTS"),
        }
        select_tts = st.selectbox(
            t("TTS Method"),
            options=tts_methods,
            index=tts_methods.index(load_key("tts_method")),
            format_func=lambda x: tts_method_labels[x],
        )
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input(t("Voice"), "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input(t("302ai API"), "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input(t("302ai API"), "fish_tts.api_key")
            fish_tts_character = st.selectbox(
                t("Fish TTS Character"),
                options=list(load_key("fish_tts.character_id_dict").keys()),
                index=list(load_key("fish_tts.character_id_dict").keys()).index(
                    load_key("fish_tts.character")
                ),
            )
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input(t("302ai API"), "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")

        elif select_tts == "f5tts":
            config_input(t("302ai API"), "f5tts.302_api")

        elif select_tts == "vieneu_tts":
            st.info(t("VieNeu-TTS requires the `vieneu` Python package."))
            current_mode = load_key("vieneu_tts.mode")
            if current_mode not in VIENEU_MODES:
                current_mode = "v3turbo"
            selected_mode = st.selectbox(
                t("VieNeu Mode"),
                options=list(VIENEU_MODES.keys()),
                format_func=lambda x: VIENEU_MODES[x],
                index=list(VIENEU_MODES.keys()).index(current_mode),
                help="Use Local v3 Turbo unless you already run a VieNeu API server."
            )
            if selected_mode != load_key("vieneu_tts.mode"):
                update_key("vieneu_tts.mode", selected_mode)
                st.rerun()

            current_emotion = load_key("vieneu_tts.emotion")
            if current_emotion not in VIENEU_EMOTIONS:
                current_emotion = "natural"
            selected_emotion = st.selectbox(
                t("VieNeu Emotion"),
                options=list(VIENEU_EMOTIONS.keys()),
                format_func=lambda emotion: VIENEU_EMOTIONS[emotion],
                index=list(VIENEU_EMOTIONS.keys()).index(current_emotion),
            )
            if selected_emotion != load_key("vieneu_tts.emotion"):
                update_key("vieneu_tts.emotion", selected_emotion)
                st.rerun()

            current_ref_audio = load_key("vieneu_tts.ref_audio")
            voice_mode_options = {
                "preset": "Use built-in voice",
                "clone": "Clone from reference audio",
            }
            current_voice_mode = "clone" if current_ref_audio else "preset"
            selected_voice_mode = st.selectbox(
                t("VieNeu Voice Mode"),
                options=list(voice_mode_options.keys()),
                format_func=lambda mode: voice_mode_options[mode],
                index=list(voice_mode_options.keys()).index(current_voice_mode),
            )

            if selected_voice_mode == "preset":
                if current_ref_audio or load_key("vieneu_tts.ref_text"):
                    update_key("vieneu_tts.ref_audio", "")
                    update_key("vieneu_tts.ref_text", "")
                    st.rerun()
                voice_options = _get_vieneu_voice_options()
                current_voice = load_key("vieneu_tts.voice")
                if current_voice not in voice_options:
                    current_voice = next(iter(voice_options))
                selected_voice = st.selectbox(
                    t("VieNeu Voice"),
                    options=list(voice_options.keys()),
                    format_func=lambda voice: voice_options[voice],
                    index=list(voice_options.keys()).index(current_voice),
                )
                if selected_voice != load_key("vieneu_tts.voice"):
                    update_key("vieneu_tts.voice", selected_voice)
                    st.rerun()
            else:
                uploaded_audio = st.file_uploader(
                    t("VieNeu Reference Audio"),
                    type=load_key("allowed_audio_formats"),
                    help="Upload a short, clear 3-10 second voice sample for cloning."
                )
                if uploaded_audio is not None:
                    saved_ref_audio = _save_uploaded_reference_audio(uploaded_audio)
                    if saved_ref_audio != load_key("vieneu_tts.ref_audio"):
                        update_key("vieneu_tts.ref_audio", saved_ref_audio)
                        st.rerun()
                config_input(t("VieNeu Reference Audio Path"), "vieneu_tts.ref_audio", help="Optional manual path to the reference audio file.")
                if selected_mode == "remote":
                    config_input(t("VieNeu Reference Text"), "vieneu_tts.ref_text", help="Optional transcript of the reference audio for remote VieNeu.")

            if selected_mode == "remote":
                config_input(t("VieNeu API Base"), "vieneu_tts.api_base")
                config_input(t("VieNeu Model Name"), "vieneu_tts.model_name")

        elif select_tts == "valtec_tts":
            st.info(t("Valtec-TTS requires the `valtec_tts` package. If not installed, run: .\\.venv\\Scripts\\python.exe -m pip install git+https://github.com/tronghieuit/valtec-tts.git"))
            current_speaker = load_key("valtec_tts.speaker")
            if current_speaker not in VALTEC_SPEAKERS:
                current_speaker = "NF"
            selected_speaker = st.selectbox(
                t("Valtec Speaker"),
                options=list(VALTEC_SPEAKERS.keys()),
                format_func=lambda speaker: VALTEC_SPEAKERS[speaker],
                index=list(VALTEC_SPEAKERS.keys()).index(current_speaker),
                help="Built-in Valtec voices. NF/SF are female voices, NM1/SM/NM2 are male voices."
            )
            if selected_speaker != load_key("valtec_tts.speaker"):
                update_key("valtec_tts.speaker", selected_speaker)
                st.rerun()

            current_ref_audio = load_key("valtec_tts.ref_audio")
            mode_options = {
                "speaker": "Use built-in speaker",
                "clone": "Clone from reference audio",
            }
            current_mode = "clone" if current_ref_audio else "speaker"
            selected_mode = st.selectbox(
                t("Valtec Mode"),
                options=list(mode_options.keys()),
                format_func=lambda mode: mode_options[mode],
                index=list(mode_options.keys()).index(current_mode),
                help="Reference audio is only needed when cloning a voice."
            )

            if selected_mode == "speaker":
                if current_ref_audio:
                    update_key("valtec_tts.ref_audio", "")
                    st.rerun()
                st.caption("Reference audio is not used in built-in speaker mode.")
            else:
                uploaded_audio = st.file_uploader(
                    t("Valtec Reference Audio"),
                    type=load_key("allowed_audio_formats"),
                    help="Upload a short, clear 3-10 second voice sample for cloning."
                )
                if uploaded_audio is not None:
                    saved_ref_audio = _save_uploaded_reference_audio(uploaded_audio)
                    if saved_ref_audio != load_key("valtec_tts.ref_audio"):
                        update_key("valtec_tts.ref_audio", saved_ref_audio)
                        st.rerun()
                config_input(t("Valtec Reference Audio Path"), "valtec_tts.ref_audio", help="Optional manual path to the reference audio file.")

        elif select_tts == "kokoro_vietnamese_tts":
            st.info(t("Kokoro Vietnamese TTS requires the `kokoro-vietnamese` package. If not installed, run: .\\.venv\\Scripts\\python.exe -m pip install git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git"))

            current_voice = load_key("kokoro_vietnamese_tts.voice")
            if current_voice not in KOKORO_VOICES:
                current_voice = "diem_trinh"
            selected_voice = st.selectbox(
                t("Kokoro Vietnamese Voice"),
                options=list(KOKORO_VOICES.keys()),
                format_func=lambda voice: KOKORO_VOICES[voice],
                index=list(KOKORO_VOICES.keys()).index(current_voice),
            )
            if selected_voice != load_key("kokoro_vietnamese_tts.voice"):
                update_key("kokoro_vietnamese_tts.voice", selected_voice)
                st.rerun()

            device_options = {
                "cuda": "GPU CUDA if available",
                "cpu": "CPU",
            }
            current_device = load_key("kokoro_vietnamese_tts.device")
            if current_device not in device_options:
                current_device = "cuda"
            selected_device = st.selectbox(
                t("Kokoro Vietnamese Device"),
                options=list(device_options.keys()),
                format_func=lambda device: device_options[device],
                index=list(device_options.keys()).index(current_device),
            )
            if selected_device != load_key("kokoro_vietnamese_tts.device"):
                update_key("kokoro_vietnamese_tts.device", selected_device)
                st.rerun()


def check_api():
    try:
        resp = ask_gpt(
            "This is a test, response 'message':'success' in json format.",
            resp_type="json",
            log_title="None",
        )
        return resp.get("message") == "success"
    except Exception:
        return False


if __name__ == "__main__":
    check_api()
