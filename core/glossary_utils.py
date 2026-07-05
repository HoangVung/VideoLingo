from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

try:
    from rich import print as rprint
except Exception:
    rprint = print


DEFAULT_GLOSSARY_CONFIG = {
    "enabled": False,
    "path": "glossaries/olympiad_math_zh_vi_glossary.json",
    "auto_build_custom_terms": True,
    "auto_normalize_source": True,
    "max_terms": 120,
    "always_include_asr": True,
}


def get_glossary_config() -> dict:
    cfg = DEFAULT_GLOSSARY_CONFIG.copy()
    try:
        from core.utils.config_utils import load_key

        configured = load_key("glossary")
    except Exception:
        return cfg
    if isinstance(configured, dict):
        cfg.update({key: configured[key] for key in cfg if key in configured})
    return cfg


def load_glossary(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as file:
        glossary = json.load(file)
    if not isinstance(glossary, dict):
        raise ValueError("Glossary JSON root must be an object.")
    return glossary


def validate_glossary(path: str) -> dict:
    result = {
        "ok": False,
        "version": "",
        "domain_count": 0,
        "total_terms": 0,
        "has_asr_ocr_corrections": False,
        "warnings": [],
    }
    try:
        glossary = load_glossary(path)
        domains = glossary.get("domains", [])
        if not isinstance(domains, list):
            result["warnings"].append("Glossary field 'domains' must be a list.")
            return result
        terms = flatten_glossary_terms(glossary)
        result.update(
            {
                "ok": True,
                "version": str(glossary.get("version", "") or ""),
                "domain_count": len(domains),
                "total_terms": len(terms),
                "has_asr_ocr_corrections": any(
                    isinstance(domain, dict)
                    and domain.get("id") == "asr_ocr_corrections"
                    for domain in domains
                ),
            }
        )
    except FileNotFoundError:
        result["warnings"].append(f"Glossary file not found: {path}")
    except json.JSONDecodeError as exc:
        result["warnings"].append(f"Invalid glossary JSON: {exc}")
    except Exception as exc:
        result["warnings"].append(f"Failed to validate glossary: {exc}")
    return result


def flatten_glossary_terms(glossary: dict) -> list[dict]:
    flattened = []
    domains = glossary.get("domains", [])
    if not isinstance(domains, list):
        return flattened

    for domain in domains:
        if not isinstance(domain, dict):
            continue
        terms = domain.get("terms", [])
        if not isinstance(terms, list):
            continue
        for term in terms:
            if not isinstance(term, dict):
                continue
            zh = str(term.get("zh", "") or "").strip()
            vi = str(term.get("vi", "") or "").strip()
            if not zh or not vi:
                continue
            try:
                priority = int(term.get("priority", 5))
            except (TypeError, ValueError):
                priority = 5
            flattened.append(
                {
                    "src": zh,
                    "tgt": vi,
                    "note": str(term.get("note", "") or "").strip(),
                    "domain_id": str(domain.get("id", "") or "").strip(),
                    "domain_name": str(domain.get("name", "") or "").strip(),
                    "priority": priority,
                    "canonical_zh": str(term.get("canonical_zh", "") or "").strip(),
                }
            )
    return flattened


def read_transcript(path: str = "output/log/split_by_meaning.txt") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def term_matches_transcript(term: dict, transcript: str) -> bool:
    if not transcript:
        return False
    src = str(term.get("src", "") or "")
    canonical = str(term.get("canonical_zh", "") or "")
    return bool((src and src in transcript) or (canonical and canonical in transcript))


def _note_for_export(term: dict) -> str:
    note = str(term.get("note", "") or "").strip()
    src = str(term.get("src", "") or "").strip()
    canonical = str(term.get("canonical_zh", "") or "").strip()
    if canonical:
        prefix = f"Chuẩn hóa: {src} -> {canonical}."
        return f"{prefix} {note}".strip()
    return note


def build_custom_terms_from_glossary(
    glossary_path: str,
    transcript_path: str = "output/log/split_by_meaning.txt",
    output_path: str = "custom_terms.xlsx",
    max_terms: int = 120,
    always_include_asr: bool = True,
) -> dict:
    result = {
        "ok": False,
        "output_path": None,
        "selected_terms": 0,
        "total_terms": 0,
        "warnings": [],
    }
    try:
        glossary = load_glossary(glossary_path)
        terms = flatten_glossary_terms(glossary)
        result["total_terms"] = len(terms)
        transcript_exists = os.path.exists(transcript_path)
        transcript = read_transcript(transcript_path) if transcript_exists else ""
        if not transcript_exists:
            result["warnings"].append(f"Transcript file not found: {transcript_path}")

        selected = []
        seen = set()
        for term in terms:
            is_asr = term.get("domain_id") == "asr_ocr_corrections"
            if term_matches_transcript(term, transcript) or (always_include_asr and is_asr):
                src = term.get("src", "")
                if src and src not in seen:
                    seen.add(src)
                    selected.append(term)

        selected.sort(
            key=lambda term: (
                0 if term.get("domain_id") == "asr_ocr_corrections" else 1,
                -int(term.get("priority", 5)),
                -len(str(term.get("src", ""))),
            )
        )
        try:
            max_terms = int(max_terms)
        except (TypeError, ValueError):
            max_terms = DEFAULT_GLOSSARY_CONFIG["max_terms"]
        selected = selected[: max(0, max_terms)]

        if not selected:
            result["warnings"].append("No glossary terms matched the transcript.")
            return result

        output_parent = Path(output_path).parent
        if str(output_parent) not in ("", "."):
            output_parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"src": term["src"], "tgt": term["tgt"], "note": _note_for_export(term)}
                for term in selected
            ],
            columns=["src", "tgt", "note"],
        ).to_excel(output_path, index=False)

        result.update(
            {"ok": True, "output_path": output_path, "selected_terms": len(selected)}
        )
    except FileNotFoundError:
        result["warnings"].append(f"Glossary file not found: {glossary_path}")
    except json.JSONDecodeError as exc:
        result["warnings"].append(f"Invalid glossary JSON: {exc}")
    except ImportError as exc:
        result["warnings"].append(
            f"Failed to write Excel file. Please install an Excel writer such as openpyxl: {exc}"
        )
    except Exception as exc:
        result["warnings"].append(f"Failed to build custom terms: {exc}")
    return result


def normalize_text_with_glossary(
    text: str,
    glossary_path: str,
) -> tuple[str, list[dict]]:
    try:
        glossary = load_glossary(glossary_path)
        terms = flatten_glossary_terms(glossary)
    except Exception as exc:
        rprint(f"Warning: failed to load glossary for normalization: {exc}")
        return text, []

    normalized = text
    replacements = []
    candidates = [
        term
        for term in terms
        if term.get("canonical_zh") and term.get("src") != term.get("canonical_zh")
    ]
    candidates.sort(key=lambda term: -len(str(term.get("src", ""))))
    for term in candidates:
        src = str(term.get("src", "") or "")
        canonical = str(term.get("canonical_zh", "") or "")
        count = normalized.count(src)
        if count <= 0:
            continue
        normalized = normalized.replace(src, canonical)
        replacements.append(
            {
                "src": src,
                "canonical_zh": canonical,
                "count": count,
                "tgt": term.get("tgt", ""),
            }
        )
    return normalized, replacements


def normalize_transcript_file_with_glossary(
    input_path: str,
    glossary_path: str,
    output_path: str | None = None,
) -> dict:
    result = {
        "ok": False,
        "input_path": input_path,
        "output_path": output_path or input_path,
        "replacement_count": 0,
        "replacements": [],
        "warnings": [],
    }
    if not os.path.exists(input_path):
        result["warnings"].append(f"Transcript file not found: {input_path}")
        return result
    try:
        original = read_transcript(input_path)
        normalized, replacements = normalize_text_with_glossary(original, glossary_path)
        final_output_path = output_path or input_path
        if output_path is None:
            input_file = Path(input_path)
            backup_path = str(
                input_file.with_name(
                    f"{input_file.stem}.before_glossary{input_file.suffix}"
                )
            )
            shutil.copyfile(input_path, backup_path)
            result["backup_path"] = backup_path
        Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(final_output_path).write_text(normalized, encoding="utf-8")
        result.update(
            {
                "ok": True,
                "output_path": final_output_path,
                "replacement_count": sum(item["count"] for item in replacements),
                "replacements": replacements,
            }
        )
    except Exception as exc:
        result["warnings"].append(f"Failed to normalize transcript: {exc}")
    return result


def prepare_glossary_for_translation() -> dict:
    cfg = get_glossary_config()
    if not cfg.get("enabled", False):
        return {"ok": True, "skipped": True, "reason": "Glossary disabled"}

    glossary_path = cfg.get("path")
    if not glossary_path or not os.path.exists(glossary_path):
        return {
            "ok": False,
            "skipped": True,
            "warnings": [f"Glossary file not found: {glossary_path}"],
        }

    results = {"ok": True, "normalized": None, "custom_terms": None, "warnings": []}
    if cfg.get("auto_normalize_source", True):
        normalized = normalize_transcript_file_with_glossary(
            input_path="output/log/split_by_meaning.txt",
            glossary_path=glossary_path,
            output_path=None,
        )
        results["normalized"] = normalized
        results["warnings"].extend(normalized.get("warnings", []))

    if cfg.get("auto_build_custom_terms", True):
        custom_terms = build_custom_terms_from_glossary(
            glossary_path=glossary_path,
            transcript_path="output/log/split_by_meaning.txt",
            output_path="custom_terms.xlsx",
            max_terms=int(cfg.get("max_terms", 120)),
            always_include_asr=bool(cfg.get("always_include_asr", True)),
        )
        results["custom_terms"] = custom_terms
        results["warnings"].extend(custom_terms.get("warnings", []))

    if results["warnings"]:
        rprint("Glossary preparation warnings:", results["warnings"])
    return results
