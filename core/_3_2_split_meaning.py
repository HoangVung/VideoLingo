import os
import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING
console = Console()

def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    try:
        split_prompt = get_split_prompt(sentence, num_parts, word_limit)
        def valid_split(response_data):
            choice = response_data["choice"]
            if f'split{choice}' not in response_data:
                return {"status": "error", "message": "Missing required key: `split`"}
            split_text = response_data[f"split{choice}"]
            if "[br]" not in split_text:
                return {"status": "error", "message": "Split failed, no [br] found"}
            
            # Explicit blocklist of template strings
            blocklist = [
                "first splitting approach",
                "alternative splitting approach",
                "tags at split positions",
                "brief description of sentence structure",
                "brief description"
            ]
            for phrase in blocklist:
                if phrase in split_text.lower():
                    return {"status": "error", "message": f"Split result contains blocklisted template phrase: '{phrase}'"}
            
            # Guard against small models echoing the placeholder from the prompt template
            clean_split = split_text.replace("[br]", "").replace(" ", "").lower()
            clean_orig = sentence.replace(" ", "").lower()
            # Remove punctuation and quotes
            for char in ",.?!;:-\"'\u3002\uff0c\uff01\uff1f\u3001":
                clean_split = clean_split.replace(char, "")
                clean_orig = clean_orig.replace(char, "")
                
            if SequenceMatcher(None, clean_split, clean_orig).ratio() < 0.5:
                return {"status": "error", "message": "Split result does not resemble original sentence (possible placeholder echo)"}
            return {"status": "success", "message": "Split completed"}
        
        response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json', valid_def=valid_split, log_title='split_by_meaning')
        choice = response_data["choice"]
        best_split = response_data[f"split{choice}"]
        split_points = find_split_positions(sentence, best_split)
        # split the sentence based on the split points
        for i, split_point in enumerate(split_points):
            if i == 0:
                best_split = sentence[:split_point] + '\n' + sentence[split_point:]
            else:
                parts = best_split.split('\n')
                last_part = parts[-1]
                parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
                best_split = '\n'.join(parts)
        if index != -1:
            console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
        table = Table(title="")
        table.add_column("Type", style="cyan")
        table.add_column("Sentence")
        table.add_row("Original", sentence, style="yellow")
        table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
        console.print(table)
        
        return best_split
    except Exception as e:
        console.print(f"[red]Error splitting sentence {index} '{sentence}': {e}. Falling back to original sentence.[/red]")
        return sentence

def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            # print("Tokenization result:", tokens)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # read input sentences
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=load_key("max_split_length"), max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')
    
    # Auto-repair any placeholders that might have bypassed logic
    repair_split_file()

def repair_split_file(split_by_meaning_path=_3_2_SPLIT_BY_MEANING, split_by_nlp_path=_3_1_SPLIT_BY_NLP):
    if not os.path.exists(split_by_meaning_path) or not os.path.exists(split_by_nlp_path):
        return {"ok": False, "msg": "Files not found"}
        
    with open(split_by_nlp_path, 'r', encoding='utf-8') as f:
        nlp_sentences = [line.strip() for line in f if line.strip()]
        
    with open(split_by_meaning_path, 'r', encoding='utf-8') as f:
        meaning_lines = [line.strip() for line in f if line.strip()]
        
    blocklist = [
        "splitting approach",
        "tags at split positions",
        "brief description"
    ]
    
    has_bad = False
    for line in meaning_lines:
        if any(phrase in line.lower() for phrase in blocklist):
            has_bad = True
            break
            
    if not has_bad:
        return {"ok": True, "repaired": False, "msg": "No placeholder lines detected"}
        
    repaired_lines = []
    meaning_idx = 0
    repaired_count = 0
    
    for nlp_sentence in nlp_sentences:
        clean_nlp = nlp_sentence.replace(" ", "").replace("　", "").lower()
        if not clean_nlp:
            continue
            
        current_combined = ""
        consumed_lines = []
        
        while meaning_idx < len(meaning_lines):
            next_line = meaning_lines[meaning_idx]
            is_placeholder = any(phrase in next_line.lower() for phrase in blocklist)
            clean_next = next_line.replace(" ", "").replace("　", "").lower()
            
            if is_placeholder:
                meaning_idx += 1
                consumed_lines = [nlp_sentence]
                repaired_count += 1
                break
                
            if len(current_combined) + len(clean_next) <= len(clean_nlp) + 5:
                current_combined += clean_next
                consumed_lines.append(next_line)
                meaning_idx += 1
                
                ratio = SequenceMatcher(None, current_combined, clean_nlp).ratio()
                if ratio > 0.85 or len(current_combined) >= len(clean_nlp) - 2:
                    break
            else:
                if not consumed_lines:
                    consumed_lines.append(next_line)
                    meaning_idx += 1
                break
                
        contains_placeholder = any(any(phrase in line.lower() for phrase in blocklist) for line in consumed_lines)
        if contains_placeholder or not consumed_lines:
            repaired_lines.append(nlp_sentence)
            repaired_count += 1
        else:
            repaired_lines.extend(consumed_lines)
            
    # Write repaired lines back
    with open(split_by_meaning_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(repaired_lines))
        
    msg = f"Repaired split_by_meaning.txt successfully: fixed {repaired_count} corrupted lines."
    console.print(f"[green]Success: {msg}[/green]")
    return {"ok": True, "repaired": True, "msg": msg}

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()