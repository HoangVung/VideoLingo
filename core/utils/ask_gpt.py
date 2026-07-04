import os
import json
import time
import hashlib
import re
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _load_log_file(file):
    if not os.path.exists(file):
        return []
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        corrupt_file = f"{file}.corrupt-{int(time.time())}"
        os.replace(file, corrupt_file)
        rprint(f"[yellow]Corrupt GPT cache moved to {corrupt_file}: {e}[/yellow]")
        return []

def _write_log_file(file, logs):
    tmp_file = f"{file}.tmp"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    os.replace(tmp_file, file)

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        logs = _load_log_file(file)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        _write_log_file(file, logs)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        for item in _load_log_file(file):
            if item["prompt"] == prompt and item["resp_type"] == resp_type:
                return item["resp"]
        return False

def _load_optional_key(key, default=None):
    try:
        return load_key(key)
    except KeyError:
        return default

def _normalize_base_url(base_url):
    if 'ark' in base_url:
        return "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    if 'v1' not in base_url:
        return base_url.strip('/') + '/v1'
    return base_url

def _sanitize_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        sanitized = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and item_type:
                sanitized.append(item)
        return sanitized
    return "" if content is None else str(content)

def _sanitize_messages(messages):
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not role:
            continue
        sanitized.append({
            "role": role,
            "content": _sanitize_content(message.get("content")),
        })
    return sanitized

def _without_none_values(params):
    return {key: value for key, value in params.items() if value is not None}

def _coerce_json_response(resp):
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, list):
        dict_items = [item for item in resp if isinstance(item, dict)]
        if len(dict_items) == 1:
            return dict_items[0]
        for item in dict_items:
            if any(key in item for key in ("choice", "split1", "split2", "align", "theme", "terms", "result")):
                return item
    return resp

def _repair_common_json_typos(content):
    if not isinstance(content, str):
        return content
    # Small local models sometimes emit `"direct："text"` or `"free："text"`;
    # fix those field separators before json_repair drops the value.
    fixed = re.sub(r'"(direct|free|origin|reflect|result|text)\s*[\uff1a:]\s*"', r'"\1": "', content)
    fixed = re.sub(r'"(direct|free|origin|reflect|result|text)\s*[\uff1a:]\s*([^"\s{}\[\],][^,\n\r}]*)', r'"\1": "\2"', fixed)
    return fixed

def _preview_payload(params, api_settings):
    def preview_content(content):
        if isinstance(content, str):
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
            if len(content) <= 500:
                preview = content
            else:
                preview = f"{content[:350]}\n...\n{content[-150:]}"
            return {
                "sha1": digest,
                "chars": len(content),
                "preview": preview,
            }
        return content

    safe_messages = [
        {**message, "content": preview_content(message.get("content"))}
        for message in params.get("messages", [])
    ]
    safe_payload = {**params, "messages": safe_messages}
    safe_payload["base_url"] = api_settings["base_url"]
    safe_payload["api_key"] = "***" if api_settings["api_key"] else ""
    rprint("[cyan]GPT request payload preview:[/cyan]", json.dumps(safe_payload, ensure_ascii=False, indent=2))

def _get_api_settings():
    provider = _load_optional_key("api.provider", "openai")
    base_url = _normalize_base_url(load_key("api.base_url"))
    api_key = load_key("api.key")

    if provider in ("local_openai", "local_openai_compatible"):
        api_key = api_key or "lm-studio"
        base_url = load_key("api.base_url") or "http://localhost:1234/v1"

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": load_key("api.model"),
        "temperature": _load_optional_key("api.temperature", None),
        "timeout": _load_optional_key("api.timeout", 300),
        "max_tokens": _load_optional_key("api.max_tokens", None),
    }

def test_gpt_connection():
    api_settings = _get_api_settings()
    try:
        client = OpenAI(api_key=api_settings["api_key"], base_url=api_settings["base_url"])
        params = dict(
            model=api_settings["model"],
            messages=[{"role": "user", "content": "Reply only OK."}],
            timeout=api_settings["timeout"],
        )
        if api_settings["temperature"] is not None:
            params["temperature"] = api_settings["temperature"]
        resp_raw = client.chat.completions.create(**params)
        resp_content = resp_raw.choices[0].message.content.strip()
        return resp_content.upper() == "OK", resp_content
    except Exception as e:
        if api_settings["provider"] in ("local_openai", "local_openai_compatible"):
            return False, "Cannot connect to Local OpenAI-compatible backend at http://localhost:1234/v1. Please start LM Studio Local Server."
        return False, str(e)

def fetch_gpt_models():
    api_settings = _get_api_settings()
    if not api_settings["api_key"]:
        raise ValueError("API key is not set")
    client = OpenAI(api_key=api_settings["api_key"], base_url=api_settings["base_url"])
    models = client.models.list()
    model_ids = sorted(
        model.id for model in models.data
        if getattr(model, "id", None)
    )
    if not model_ids:
        raise ValueError("No models returned by the API")
    return model_ids

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default", max_tokens=None, temperature=None):
    api_settings = _get_api_settings()
    if not api_settings["api_key"]:
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        if resp_type == "json":
            cached = _coerce_json_response(cached)
        return cached

    model = api_settings["model"]
    client = OpenAI(api_key=api_settings["api_key"], base_url=api_settings["base_url"])
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = _sanitize_messages([{"role": "user", "content": prompt}])

    params = _without_none_values(dict(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=api_settings["timeout"],
        max_tokens=max_tokens if max_tokens is not None else api_settings["max_tokens"],
    ))
    temp_to_use = temperature if temperature is not None else api_settings["temperature"]
    if temp_to_use is not None:
        params["temperature"] = temp_to_use
    _preview_payload(params, api_settings)
    resp_raw = client.chat.completions.create(**params)

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = _coerce_json_response(json_repair.loads(_repair_common_json_typos(resp_content)))
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
