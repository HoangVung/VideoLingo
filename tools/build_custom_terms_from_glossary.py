from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core.glossary_utils import (  # noqa: E402
    build_custom_terms_from_glossary,
    normalize_transcript_file_with_glossary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build custom_terms.xlsx from a Chinese-Vietnamese math glossary."
    )
    parser.add_argument(
        "--glossary",
        default="glossaries/olympiad_math_zh_vi_glossary.json",
        help="Path to the glossary JSON file.",
    )
    parser.add_argument(
        "--transcript",
        default="output/log/split_by_meaning.txt",
        help="Path to split_by_meaning.txt.",
    )
    parser.add_argument(
        "--output",
        default="custom_terms.xlsx",
        help="Output Excel path.",
    )
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument(
        "--no-always-include-asr",
        action="store_true",
        help="Do not force include terms from domain asr_ocr_corrections.",
    )
    parser.add_argument(
        "--normalize-source",
        action="store_true",
        help="Normalize the transcript file before building custom terms.",
    )
    args = parser.parse_args()

    if args.normalize_source:
        normalized = normalize_transcript_file_with_glossary(
            input_path=args.transcript,
            glossary_path=args.glossary,
            output_path=None,
        )
        print("Normalization result:")
        print(json.dumps(normalized, ensure_ascii=False, indent=2))

    result = build_custom_terms_from_glossary(
        glossary_path=args.glossary,
        transcript_path=args.transcript,
        output_path=args.output,
        max_terms=args.max_terms,
        always_include_asr=not args.no_always_include_asr,
    )
    print("Custom terms result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
