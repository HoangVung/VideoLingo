import json
import os

import pandas as pd

from core.prompts import get_summary_prompt
from core.utils import *
from core.utils.models import _3_2_SPLIT_BY_MEANING, _4_1_TERMINOLOGY

CUSTOM_TERMS_PATH = "custom_terms.xlsx"


def combine_chunks():
    """Combine the text chunks identified by whisper into a single long text"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = " ".join(cleaned_sentences)
    return combined_text[: load_key("summary_length")]


def search_things_to_note_in_prompt(sentence):
    """Search for terms to note in the given sentence"""
    with open(_4_1_TERMINOLOGY, "r", encoding="utf-8") as file:
        things_to_note = json.load(file)
    things_to_note_list = [
        term["src"]
        for term in things_to_note["terms"]
        if term["src"].lower() in sentence.lower()
    ]
    if things_to_note_list:
        prompt = "\n".join(
            f'{i + 1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note["terms"])
            if term["src"] in things_to_note_list
        )
        return prompt
    return None


def load_custom_terms(path=CUSTOM_TERMS_PATH):
    if not os.path.exists(path):
        return {"terms": []}

    try:
        df = pd.read_excel(path)
    except Exception as e:
        rprint(f"Warning: failed to read custom terms: {e}")
        return {"terms": []}

    if df.empty:
        return {"terms": []}

    terms = []
    if {"src", "tgt", "note"}.issubset(set(df.columns)):
        for _, row in df.iterrows():
            terms.append(
                {
                    "src": str(row["src"]).strip(),
                    "tgt": str(row["tgt"]).strip(),
                    "note": str(row["note"]).strip(),
                }
            )
    else:
        for _, row in df.iterrows():
            if len(row) >= 3:
                terms.append(
                    {
                        "src": str(row.iloc[0]).strip(),
                        "tgt": str(row.iloc[1]).strip(),
                        "note": str(row.iloc[2]).strip(),
                    }
                )

    clean_terms = []
    seen = set()
    for term in terms:
        src = term.get("src", "")
        if not src or src.lower() == "nan" or src in seen:
            continue
        seen.add(src)
        clean_terms.append(term)

    return {"terms": clean_terms}


def get_summary():
    src_content = combine_chunks()
    custom_terms_json = load_custom_terms(CUSTOM_TERMS_PATH)
    if len(custom_terms_json["terms"]) > 0:
        rprint(f"Custom Terms Loaded: {len(custom_terms_json['terms'])} terms")
        rprint(
            "Terms Content:",
            json.dumps(custom_terms_json, indent=2, ensure_ascii=False),
        )
    summary_prompt = get_summary_prompt(src_content, custom_terms_json)
    rprint("Summarizing and extracting terminology ...")

    def valid_summary(response_data):
        required_keys = {"src", "tgt", "note"}
        if "terms" not in response_data:
            return {"status": "error", "message": "Invalid response format"}
        for term in response_data["terms"]:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid response format"}
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(
        summary_prompt, resp_type="json", valid_def=valid_summary, log_title="summary"
    )
    summary["terms"].extend(custom_terms_json["terms"])

    with open(_4_1_TERMINOLOGY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    rprint(f"Summary log saved to -> `{_4_1_TERMINOLOGY}`")


if __name__ == "__main__":
    get_summary()
