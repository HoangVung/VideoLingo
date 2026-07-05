from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
console = Console()

def valid_translate_result(result, required_keys: list, required_sub_keys: list):
    if not isinstance(result, dict):
        return {"status": "error", "message": f"Response is not a dictionary, got {type(result).__name__}"}
    
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not isinstance(result[key], dict):
            return {"status": "error", "message": f"Item {key} is not a dictionary, got {type(result[key]).__name__}"}
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    check_cancel()
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['direct'])
        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['free'])
        last_result = None
        for retry in range(3):
            try:
                if step_name == 'faithfulness':
                    result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_faith, log_title=f'translate_{step_name}')
                elif step_name == 'expressiveness':
                    result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_express, log_title=f'translate_{step_name}')
                last_result = result
                if len(lines.split('\n')) == len(result):
                    return result
            except Exception:
                pass
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        # Fallback: fill missing keys with empty translation instead of crashing
        console.print(f'[yellow]⚠️ Block {index}: filling missing keys with empty strings as fallback[/yellow]')
        line_list = lines.split('\n')
        fallback = last_result if isinstance(last_result, dict) else {}
        for i, line in enumerate(line_list, 1):
            key = str(i)
            if key not in fallback or not isinstance(fallback[key], dict):
                if step_name == 'faithfulness':
                    fallback[key] = {'origin': line, 'direct': line}
                else:
                    fallback[key] = {'free': fallback.get(key, {}).get('direct', line) if isinstance(fallback.get(key), dict) else line}
        return fallback

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')

    n_lines = len(lines.split('\n'))
    ordered_keys = [str(i) for i in range(1, n_lines + 1)]
    for k in ordered_keys:
        if k in faith_result and isinstance(faith_result[k], dict):
            faith_result[k]["direct"] = faith_result[k]["direct"].replace('\n', ' ')

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # Use explicit ordered keys to avoid extra/out-of-order keys from LLM
        translate_result = "\n".join([faith_result[k]["direct"].strip() for k in ordered_keys if k in faith_result])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for idx, k in enumerate(ordered_keys):
            if k not in faith_result:
                continue
            table.add_row(f"[cyan]Origin:  {faith_result[k].get('origin', '')}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[k]['direct']}[/magenta]")
            if idx < n_lines - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness')

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    express_ordered = [str(i) for i in range(1, n_lines + 1)]
    translate_result = "\n".join([express_result[k]["free"].replace('\n', ' ').strip() for k in express_ordered if k in express_result])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)