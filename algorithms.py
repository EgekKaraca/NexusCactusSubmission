#Alpro
import sys, os
sys.path.insert(0, os.path.expanduser("~/cactus/python/src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
functiongemma_path = os.path.expanduser("~/cactus/weights/functiongemma-270m-it")

import json, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# ADDED: A powerful, context-aware system prompt
SYS_PROMPT = (
    "You are a strict function-calling assistant. "
    "Extract argument values EXACTLY as they appear in the user text. Do not invent emails or append symbols. "
    "Example response:\n"
    '{"function_calls": [{"name": "tool_name", "arguments": {"arg1": "value1"}}]}'
)

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": SYS_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.0,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=(
                "You are an automation agent. The user will ask for multiple actions at once. "
                "You MUST output ALL required function calls in parallel, immediately, in a single response. "
                "DO NOT wait for the result of one tool before calling the next. Output them all at once."
            ),
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    if gemini_response.candidates:
        for candidate in gemini_response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                        })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# --------------- synonym dictionary ---------------

SYNONYM_MAP = {
    "alarm": ["wake", "clock", "wakeup", "alert", "morning"],
    "message": ["text", "sms", "email", "msg", "tell", "saying", "send"],
    "weather": ["forecast", "temperature", "rain", "sunny", "climate", "outside"],
    "music": ["song", "podcast", "play", "playlist", "tune", "beats", "listen", "audio"],
    "navigate": ["map", "directions", "route", "drive", "gps"],
    "reminder": ["remind", "calendar", "schedule", "appointment", "note", "meeting", "groceries"],
    "timer": ["countdown", "stopwatch", "minute", "minutes"],
    "contacts": ["contact", "lookup", "find", "search", "look"],
}

_REVERSE_SYNONYMS = {}
for _canonical, _syns in SYNONYM_MAP.items():
    for _s in _syns:
        _REVERSE_SYNONYMS.setdefault(_s, set()).add(_canonical)
    _REVERSE_SYNONYMS.setdefault(_canonical, set()).add(_canonical)


# --------------- helpers ---------------

STOP_WORDS = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "about", 
              "some", "my", "me", "i", "get", "set", "is", "what", "how", "find", 
              "look", "saying", "and", "of", "like", "up"}

def _get_clean_words(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - STOP_WORDS

def _get_tool_keywords(tool):
    keywords = set()
    for part in tool["name"].split("_"):
        if part.lower() not in STOP_WORDS:
            keywords.add(part.lower())
    
    keywords |= _get_clean_words(tool.get("description", ""))
    
    for prop_info in tool["parameters"].get("properties", {}).values():
        keywords |= _get_clean_words(prop_info.get("description", ""))
    return keywords

def _expand_with_synonyms(words):
    expanded = set(words)
    for w in words:
        if w in _REVERSE_SYNONYMS:
            expanded |= _REVERSE_SYNONYMS[w]
        if w in SYNONYM_MAP:
            expanded |= set(SYNONYM_MAP[w])
    return expanded

def _has_tool_vocabulary_overlap(prompt, tools):
    prompt_words = _get_clean_words(prompt)
    expanded_prompt = _expand_with_synonyms(prompt_words)

    for tool in tools:
        tool_keywords = _get_tool_keywords(tool)
        expanded_tool = _expand_with_synonyms(tool_keywords)
        if expanded_prompt & expanded_tool:
            return True
    return False

def _tool_matches_subprompt(tool, sub_prompt):
    sub_words = _get_clean_words(sub_prompt)
    expanded_sub = _expand_with_synonyms(sub_words)

    tool_keywords = _get_tool_keywords(tool)
    expanded_tool = _expand_with_synonyms(tool_keywords)

    return bool(expanded_sub & expanded_tool)

def _fix_tool_name(call_name, tools):
    tool_names = {t["name"] for t in tools}
    if call_name in tool_names:
        return call_name
    fixed = call_name.replace(" ", "_")
    if fixed in tool_names:
        return fixed
    fixed_lower = fixed.lower()
    for tn in tool_names:
        if tn.lower() == fixed_lower:
            return tn
    return call_name

def _clean_string_arguments(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            continue
        props = tool["parameters"].get("properties", {})
        for key, val in list(call.get("arguments", {}).items()):
            expected_type = props.get(key, {}).get("type", "string")
            if expected_type == "string" and isinstance(val, str):
                cleaned = val.replace("_", " ")
                cleaned = cleaned.rstrip('.,!?')
                if key in ("title", "message"):
                    cleaned = re.sub(r'^(about\s+|to\s+|the\s+|a\s+)', '', cleaned, flags=re.IGNORECASE)
                
                # ADDED: Aggressive sanitization for names to stop hallucinated emails/tokens
                if key == "recipient":
                    if "@" in cleaned:
                        cleaned = cleaned.split("@")[0].capitalize()
                    cleaned = re.sub(r"[^a-zA-Z\s]", "", cleaned)
                
                call["arguments"][key] = cleaned

def _coerce_arguments(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            continue
        props = tool["parameters"].get("properties", {})
        for key, val in list(call.get("arguments", {}).items()):
            expected_type = props.get(key, {}).get("type", "string")
            if expected_type == "integer" and not isinstance(val, int):
                nums = re.findall(r'\d+', str(val))
                if nums:
                    call["arguments"][key] = int(nums[0])
            elif expected_type == "number" and not isinstance(val, (int, float)):
                nums = re.findall(r'[\d.]+', str(val))
                if nums:
                    call["arguments"][key] = float(nums[0])

def _fix_integer_args_from_prompt(call, prompt, tools):
    tool_map = {t["name"]: t for t in tools}
    tool = tool_map.get(call["name"])
    if not tool:
        return

    props = tool["parameters"].get("properties", {})
    for param_name, param_info in props.items():
        if param_info.get("type") != "integer":
            continue

        if param_name in ("hour", "minute"):
            m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', prompt)
            if m:
                if param_name == "hour":
                    h = int(m.group(1))
                    ampm = m.group(3).upper().replace('.', '')
                    if ampm == "PM" and h != 12:
                        h += 12
                    elif ampm == "AM" and h == 12:
                        h = 0
                    call["arguments"][param_name] = h
                elif param_name == "minute":
                    call["arguments"][param_name] = int(m.group(2)) if m.group(2) else 0
            continue

        m = re.search(rf'(\d+)\s*{re.escape(param_name)}', prompt, re.IGNORECASE)
        if m:
            call["arguments"][param_name] = int(m.group(1))
            continue

        m = re.search(rf'(\d+)-{re.escape(param_name)}', prompt, re.IGNORECASE)
        if m:
            call["arguments"][param_name] = int(m.group(1))
            continue

        desc = param_info.get("description", "").lower()
        desc_keywords = [w for w in desc.split() if len(w) > 3]
        for kw in desc_keywords:
            m = re.search(rf'(\d+)\s*{re.escape(kw)}', prompt, re.IGNORECASE)
            if m:
                call["arguments"][param_name] = int(m.group(1))
                break

def _validate_local(result, tools):
    calls = result.get("function_calls", [])
    if not calls:
        print("  [VALIDATE] Error: No function calls found in local output.")
        return False
        
    tool_map = {t["name"]: t for t in tools}
    
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            print(f"  [VALIDATE] Error: Hallucinated tool '{call['name']}' not found in available tools.")
            return False
            
        required = tool["parameters"].get("required", [])
        props = tool["parameters"].get("properties", {})
        
        # Check for missing requirements
        for req in required:
            if req not in call.get("arguments", {}):
                print(f"  [VALIDATE] Error: Missing required argument '{req}' for tool '{call['name']}'.")
                return False
        
        # ADDED: Strict Key Validation to block hallucinated parameters like "location."
        for arg_key in call.get("arguments", {}).keys():
            if arg_key not in props:
                print(f"  [VALIDATE] Error: Hallucinated argument key '{arg_key}' for tool '{call['name']}'.")
                return False
                
    return True

def _is_multi_action(prompt):
    connectors = [', and ', ' and ', ' then ', ' also ', ' plus ']
    return any(c in prompt.lower() for c in connectors)

def _split_actions(prompt):
    parts = re.split(r',?\s+and\s+|,\s+(?=[a-z])', prompt, flags=re.IGNORECASE)
    parts = [p.strip().rstrip('.,!?') for p in parts if p.strip()]
    if len(parts) <= 1:
        return [prompt.strip().rstrip('.,!?')]

    non_names = {"Set", "Send", "Text", "Get", "Check", "Find", "Look", "Play",
                 "Remind", "Search", "What", "How", "When", "Where", "The", "I",
                 "AM", "PM", "No", "Yes", "OK", "Hi", "Hello", "Let", "New"}
    resolved = []
    last_name = None
    for part in parts:
        names = re.findall(r'\b([A-Z][a-z]+)\b', part)
        real_names = [n for n in names if n not in non_names]
        if real_names:
            last_name = real_names[-1]
        elif last_name:
            part = re.sub(r'\bhim\b', last_name, part, flags=re.IGNORECASE)
            part = re.sub(r'\bher\b', last_name, part, flags=re.IGNORECASE)
            part = re.sub(r'\bthem\b', last_name, part, flags=re.IGNORECASE)
        resolved.append(part)

    return resolved

def _postprocess_calls(calls, tools):
    for call in calls:
        call["name"] = _fix_tool_name(call["name"], tools)
    
    _clean_string_arguments(calls, tools)
    _coerce_arguments(calls, tools)


# --------------- main hybrid logic ---------------

def generate_hybrid(messages, tools, confidence_threshold=0.50):
    prompt = " ".join(m["content"] for m in messages if m["role"] == "user")
    
    print(f"\n[HYBRID LOGIC] Analyzing prompt: '{prompt}'")
    print(f"[HYBRID LOGIC] Total available tools: {[t['name'] for t in tools]}")

    has_overlap = _has_tool_vocabulary_overlap(prompt, tools)
    print(f"[HYBRID LOGIC] Vocabulary overlap detected: {has_overlap}")

    multi = _is_multi_action(prompt) and len(tools) > 1
    print(f"[HYBRID LOGIC] Multi-action detected: {multi}")

    if not multi:
        print("\n[HYBRID LOGIC] Attempting single-action on-device (FunctionGemma)...")
        
        pruned_tools = [t for t in tools if _tool_matches_subprompt(t, prompt)]
        if not pruned_tools:
            pruned_tools = tools
        print(f"[HYBRID LOGIC] Tools passed to local model: {[t['name'] for t in pruned_tools]}")
        
        local = generate_cactus(messages, pruned_tools)
        calls = local.get("function_calls", [])
        print(f"[HYBRID LOGIC] Raw local calls generated: {json.dumps(calls)}")

        _postprocess_calls(calls, pruned_tools)

        for call in calls:
            _fix_integer_args_from_prompt(call, prompt, pruned_tools)

        is_valid = _validate_local(local, pruned_tools)
        
        if is_valid:
            print("[HYBRID LOGIC] Local validation PASSED. Trusting on-device.")
            local["source"] = "on-device"
            return local

        print("[HYBRID LOGIC] Local validation FAILED. Falling back to Gemini Cloud.")
        cloud = generate_cloud(messages, tools)
        _postprocess_calls(cloud.get("function_calls", []), tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud

    print("\n[HYBRID LOGIC] Attempting multi-action on-device (FunctionGemma)...")
    sub_prompts = _split_actions(prompt)
    print(f"[HYBRID LOGIC] Split prompt into {len(sub_prompts)} sub-prompts: {sub_prompts}")
    
    model = cactus_init(functiongemma_path)
    sys_msg = {"role": "system", "content": SYS_PROMPT}

    all_calls = []
    total_time = 0
    used_tools = set()

    for i, sub in enumerate(sub_prompts):
        print(f"\n  [HYBRID LOGIC] Processing Sub-prompt {i+1}/{len(sub_prompts)}: '{sub}'")
        remaining_tools = [t for t in tools if t["name"] not in used_tools]
        if not remaining_tools:
            print("  [HYBRID LOGIC] All tools exhausted. Stopping multi-action loop.")
            break

        pruned_tools = [t for t in remaining_tools if _tool_matches_subprompt(t, sub)]
        if not pruned_tools:
            pruned_tools = remaining_tools
            
        print(f"  [HYBRID LOGIC] Tools passed for this sub-prompt: {[t['name'] for t in pruned_tools]}")

        cactus_tools = [{"type": "function", "function": t} for t in pruned_tools]

        raw_str = cactus_complete(
            model,
            [sys_msg, {"role": "user", "content": sub}],
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            confidence_threshold=0.0, 
        )
        cactus_reset(model)

        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            print(f"  [HYBRID LOGIC] Error: JSONDecodeError from local model.")
            continue

        calls = raw.get("function_calls", [])
        print(f"  [HYBRID LOGIC] Raw sub-prompt calls generated: {json.dumps(calls)}")
        
        total_time += raw.get("total_time_ms", 0)

        _postprocess_calls(calls, pruned_tools)

        for call in calls:
            _fix_integer_args_from_prompt(call, sub, pruned_tools)
            if call["name"] in {t["name"] for t in pruned_tools}:
                all_calls.append(call)
                used_tools.add(call["name"])
                print(f"  [HYBRID LOGIC] Appended valid call: {call['name']}")

    cactus_destroy(model)

    if len(all_calls) == len(sub_prompts) and all_calls:
        print("\n[HYBRID LOGIC] Perfectly captured all multi-action calls. Trusting on-device.")
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "source": "on-device",
        }

    print("\n[HYBRID LOGIC] Multi-action partially failed or dropped calls. Falling back to Gemini Cloud.")
    cloud = generate_cloud(messages, tools)
    _postprocess_calls(cloud.get("function_calls", []), tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_time
    return cloud


def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result.get("function_calls", []):
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)

'''
Penaldo
import sys, os
sys.path.insert(0, os.path.expanduser("~/cactus/python/src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
functiongemma_path = os.path.expanduser("~/cactus/weights/functiongemma-270m-it")

import json, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# Pre-initialize Gemini client at module load to avoid cold-start latency
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ADDED: A powerful, context-aware system prompt
SYS_PROMPT = (
    "You are a strict function-calling assistant. "
    "Extract argument values EXACTLY as they appear in the user text. Do not invent emails or append symbols. "
    "Example response:\n"
    '{"function_calls": [{"name": "tool_name", "arguments": {"arg1": "value1"}}]}'
)

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": SYS_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.0,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _gemini_client

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=(
                "You are an automation agent. The user will ask for multiple actions at once. "
                "You MUST output ALL required function calls in parallel, immediately, in a single response. "
                "DO NOT wait for the result of one tool before calling the next. Output them all at once."
            ),
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    if gemini_response.candidates:
        for candidate in gemini_response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                        })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# --------------- synonym dictionary ---------------

SYNONYM_MAP = {
    "alarm": ["wake", "clock", "wakeup", "alert", "morning"],
    "message": ["text", "sms", "email", "msg", "tell", "saying", "send"],
    "weather": ["forecast", "temperature", "rain", "sunny", "climate", "outside"],
    "music": ["song", "podcast", "play", "playlist", "tune", "beats", "listen", "audio"],
    "navigate": ["map", "directions", "route", "drive", "gps"],
    "reminder": ["remind", "calendar", "schedule", "appointment", "note", "meeting", "groceries"],
    "timer": ["countdown", "stopwatch", "minute", "minutes"],
    "contacts": ["contact", "lookup", "find", "search", "look"],
}

_REVERSE_SYNONYMS = {}
for _canonical, _syns in SYNONYM_MAP.items():
    for _s in _syns:
        _REVERSE_SYNONYMS.setdefault(_s, set()).add(_canonical)
    _REVERSE_SYNONYMS.setdefault(_canonical, set()).add(_canonical)


# --------------- helpers ---------------

STOP_WORDS = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "about", 
              "some", "my", "me", "i", "get", "set", "is", "what", "how", "find", 
              "look", "saying", "and", "of", "like", "up"}

def _get_clean_words(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - STOP_WORDS

def _get_tool_keywords(tool):
    keywords = set()
    for part in tool["name"].split("_"):
        if part.lower() not in STOP_WORDS:
            keywords.add(part.lower())
    
    keywords |= _get_clean_words(tool.get("description", ""))
    
    for prop_info in tool["parameters"].get("properties", {}).values():
        keywords |= _get_clean_words(prop_info.get("description", ""))
    return keywords

def _expand_with_synonyms(words):
    expanded = set(words)
    for w in words:
        if w in _REVERSE_SYNONYMS:
            expanded |= _REVERSE_SYNONYMS[w]
        if w in SYNONYM_MAP:
            expanded |= set(SYNONYM_MAP[w])
    return expanded

def _tool_matches_subprompt(tool, sub_prompt):
    sub_words = _get_clean_words(sub_prompt)
    expanded_sub = _expand_with_synonyms(sub_words)

    tool_keywords = _get_tool_keywords(tool)
    expanded_tool = _expand_with_synonyms(tool_keywords)

    return bool(expanded_sub & expanded_tool)

def _fix_tool_name(call_name, tools):
    tool_names = {t["name"] for t in tools}
    if call_name in tool_names:
        return call_name
    fixed = call_name.replace(" ", "_")
    if fixed in tool_names:
        return fixed
    fixed_lower = fixed.lower()
    for tn in tool_names:
        if tn.lower() == fixed_lower:
            return tn
    return call_name

def _clean_string_arguments(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            continue
        props = tool["parameters"].get("properties", {})
        for key, val in list(call.get("arguments", {}).items()):
            expected_type = props.get(key, {}).get("type", "string")
            if expected_type == "string" and isinstance(val, str):
                cleaned = val.replace("_", " ")
                cleaned = cleaned.rstrip('.,!?')
                if key == "title":
                    cleaned = re.sub(r'^(?:about\s+|to\s+|the\s+|a\s+)', '', cleaned, flags=re.IGNORECASE)
                
                # ADDED: Aggressive sanitization for names to stop hallucinated emails/tokens
                if key == "recipient":
                    if "@" in cleaned:
                        cleaned = cleaned.split("@")[0].capitalize()
                    cleaned = re.sub(r"[^a-zA-Z\s]", "", cleaned)
                
                call["arguments"][key] = cleaned

def _coerce_arguments(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            continue
        props = tool["parameters"].get("properties", {})
        for key, val in list(call.get("arguments", {}).items()):
            expected_type = props.get(key, {}).get("type", "string")
            if expected_type == "integer" and not isinstance(val, int):
                nums = re.findall(r'\d+', str(val))
                if nums:
                    call["arguments"][key] = int(nums[0])
            elif expected_type == "number" and not isinstance(val, (int, float)):
                nums = re.findall(r'[\d.]+', str(val))
                if nums:
                    call["arguments"][key] = float(nums[0])

def _fix_integer_args_from_prompt(call, prompt, tools):
    tool_map = {t["name"]: t for t in tools}
    tool = tool_map.get(call["name"])
    if not tool:
        return

    props = tool["parameters"].get("properties", {})
    for param_name, param_info in props.items():
        if param_info.get("type") != "integer":
            continue

        if param_name in ("hour", "minute"):
            m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', prompt)
            if m:
                if param_name == "hour":
                    h = int(m.group(1))
                    ampm = m.group(3).upper().replace('.', '')
                    if ampm == "PM" and h != 12:
                        h += 12
                    elif ampm == "AM" and h == 12:
                        h = 0
                    call["arguments"][param_name] = h
                elif param_name == "minute":
                    call["arguments"][param_name] = int(m.group(2)) if m.group(2) else 0
            continue

        m = re.search(rf'(\d+)\s*{re.escape(param_name)}', prompt, re.IGNORECASE)
        if m:
            call["arguments"][param_name] = int(m.group(1))
            continue

        m = re.search(rf'(\d+)-{re.escape(param_name)}', prompt, re.IGNORECASE)
        if m:
            call["arguments"][param_name] = int(m.group(1))
            continue

        desc = param_info.get("description", "").lower()
        desc_keywords = [w for w in desc.split() if len(w) > 3]
        for kw in desc_keywords:
            m = re.search(rf'(\d+)\s*{re.escape(kw)}', prompt, re.IGNORECASE)
            if m:
                call["arguments"][param_name] = int(m.group(1))
                break

def _validate_local(result, tools):
    calls = result.get("function_calls", [])
    if not calls:
        return False

    tool_map = {t["name"]: t for t in tools}

    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            return False

        required = tool["parameters"].get("required", [])
        props = tool["parameters"].get("properties", {})

        for req in required:
            if req not in call.get("arguments", {}):
                return False

        for arg_key in call.get("arguments", {}).keys():
            if arg_key not in props:
                return False

    return True

def _is_multi_action(prompt):
    connectors = [', and ', ' and ', ' then ', ' also ', ' plus ']
    return any(c in prompt.lower() for c in connectors)

def _split_actions(prompt):
    parts = re.split(r',?\s+and\s+|,\s+(?=[a-z])', prompt, flags=re.IGNORECASE)
    parts = [p.strip().rstrip('.,!?') for p in parts if p.strip()]
    if len(parts) <= 1:
        return [prompt.strip().rstrip('.,!?')]

    non_names = {"Set", "Send", "Text", "Get", "Check", "Find", "Look", "Play",
                 "Remind", "Search", "What", "How", "When", "Where", "The", "I",
                 "AM", "PM", "No", "Yes", "OK", "Hi", "Hello", "Let", "New"}
    resolved = []
    last_name = None
    for part in parts:
        names = re.findall(r'\b([A-Z][a-z]+)\b', part)
        real_names = [n for n in names if n not in non_names]
        if real_names:
            last_name = real_names[-1]
        elif last_name:
            part = re.sub(r'\bhim\b', last_name, part, flags=re.IGNORECASE)
            part = re.sub(r'\bher\b', last_name, part, flags=re.IGNORECASE)
            part = re.sub(r'\bthem\b', last_name, part, flags=re.IGNORECASE)
        resolved.append(part)

    return resolved

def _fix_reminder_args_from_prompt(call, prompt):
    """Extract title and time for create_reminder directly from the prompt."""
    if call.get("name") != "create_reminder":
        return

    # Extract time: look for patterns like "at 3:00 PM", "at 5 PM"
    time_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?)\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', prompt)
    if time_match:
        t = time_match.group(1)
        ampm = time_match.group(2).upper().replace('.', '')
        if ':' not in t:
            t = t + ":00"
        call["arguments"]["time"] = f"{t} {ampm}"

    # Extract title: text between "remind me to/about" and "at <time>"
    title_match = re.search(
        r'(?:remind\s+me\s+(?:to|about)\s+)(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)',
        prompt, re.IGNORECASE
    )
    if title_match:
        title = title_match.group(1).strip().rstrip('.,!?')
        # Strip leading articles ("the", "a", "an") to match expected format
        title = re.sub(r'^(?:the|a|an)\s+', '', title, flags=re.IGNORECASE)
        call["arguments"]["title"] = title


def _postprocess_calls(calls, tools):
    for call in calls:
        call["name"] = _fix_tool_name(call["name"], tools)

    _clean_string_arguments(calls, tools)
    _coerce_arguments(calls, tools)


# --------------- regex-based fallback extractor ---------------

def _extract_string_arg(prompt, param_name, param_info):
    """Try to extract a string argument from the prompt using heuristics."""
    desc = param_info.get("description", "").lower()

    # --- location / city ---
    if param_name == "location" or "city" in desc or "location" in desc:
        # "weather in <Location>" / "weather for <Location>"
        m = re.search(r'(?:in|for|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?:\?|$|,|\.|!)', prompt)
        if m:
            return m.group(1).strip()
        # Case-insensitive fallback: "in <location>"
        m = re.search(r'(?:in|for|at)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\?|$|,|\.|!)', prompt, re.IGNORECASE)
        if m:
            # Title-case the result
            return m.group(1).strip().title()
        # Capitalized words at end
        m = re.search(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*\??$', prompt.strip())
        if m:
            return m.group(1).strip()

    # --- recipient / contact name ---
    if param_name in ("recipient", "contact", "name") or "recipient" in desc or "contact" in desc or "person" in desc:
        # "send/text a message to <Name>"
        m = re.search(r'(?:message|text)\s+to\s+([A-Z][a-z]+)', prompt)
        if m:
            return m.group(1)
        # "send/text/message/tell <Name>"
        m = re.search(r'(?:send|text|message|tell|contact)\s+([A-Z][a-z]+)', prompt)
        if m:
            return m.group(1)
        # "to <Name>"
        m = re.search(r'\bto\s+([A-Z][a-z]+)', prompt)
        if m:
            return m.group(1)
        # Case-insensitive fallback for sub-prompts
        m = re.search(r'(?:send|text|message|tell|contact)\s+(?:a\s+(?:message|text)\s+to\s+)?([a-zA-Z]+)', prompt, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        m = re.search(r'\bto\s+([a-z]+)', prompt, re.IGNORECASE)
        if m:
            val = m.group(1)
            # Skip common non-name words
            if val.lower() not in {"the", "a", "an", "my", "say", "play", "set", "get", "be", "do", "go", "him", "her"}:
                return val.capitalize()

    # --- message body ---
    if param_name == "message" or ("message" in desc and "recipient" not in desc and "contact" not in desc):
        # "saying/asking/about <message>"
        m = re.search(r'(?:saying|say|that\s+says|asking|asked)\s+["\']?(.+?)["\']?\s*$', prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.,!?')
        # "tell Name <message>"
        m = re.search(r'(?:tell)\s+[A-Z][a-z]+\s+(.+?)$', prompt, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.,!?')
            val = re.sub(r'^(?:that|to)\s+', '', val, flags=re.IGNORECASE)
            if val:
                return val
        # "to Name saying/that <message>"
        m = re.search(r'to\s+[A-Z][a-z]+\s+(?:saying|that\s+says?|to\s+say|asking)\s+(.+?)$', prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.,!?')
        # "text/message/send to Name <message>" — everything after the name
        m = re.search(r'(?:text|message|send\s+(?:a\s+)?(?:message|text)\s+to)\s+[A-Z][a-z]+\s+(.+?)$', prompt, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.,!?')
            val = re.sub(r'^(?:saying|that\s+says?|to\s+say|asking)\s+', '', val, flags=re.IGNORECASE)
            if val:
                return val
        # Generic fallback: everything after a capitalized name
        m = re.search(r'[A-Z][a-z]+\s+(?:saying|that|asking|about)\s+(.+?)$', prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.,!?')

    # --- time (string, for reminders) ---
    if param_name == "time" or "time" in desc:
        m = re.search(r'(?:at\s+)?(\d{1,2}(?::\d{2})?)\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', prompt)
        if m:
            t = m.group(1)
            ampm = m.group(2).upper().replace('.', '')
            if ':' not in t:
                t = t + ":00"
            return f"{t} {ampm}"

    # --- title (for reminders) ---
    if param_name == "title" or ("title" in desc and ("remind" in desc or "task" in desc or "event" in desc)):
        m = re.search(
            r'(?:remind\s+me\s+(?:to|about)\s+)(.+?)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)',
            prompt, re.IGNORECASE
        )
        if m:
            title = m.group(1).strip().rstrip('.,!?')
            title = re.sub(r'^(?:the|a|an)\s+', '', title, flags=re.IGNORECASE)
            return title
        # "reminder to/about <title>"
        m = re.search(r'(?:remind(?:er)?\s+(?:me\s+)?(?:to|about)\s+)(.+?)(?:\s+at\s+|$)', prompt, re.IGNORECASE)
        if m:
            title = m.group(1).strip().rstrip('.,!?')
            title = re.sub(r'^(?:the|a|an)\s+', '', title, flags=re.IGNORECASE)
            return title

    # --- song / music ---
    is_music = ("song" in desc or "music" in desc or "play" in desc or "track" in desc
                or param_name in ("song", "track"))
    if is_music:
        m = re.search(r'(?:play|listen\s+to|put\s+on)\s+(.+?)(?:\s+on\s+|\s+by\s+|$)', prompt, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.,!?')
            # Only remove "the song" prefix, keep other text as-is
            val = re.sub(r'^(?:the\s+song\s+)', '', val, flags=re.IGNORECASE)
            if val:
                return val

    # --- search query ---
    if param_name == "query" or "search" in desc or "query" in desc or "lookup" in desc:
        # "search for <query>" / "find <query>" / "look up <query>"
        m = re.search(r'(?:search\s+(?:for\s+)?|find|look\s+up|look\s+for)\s+(.+?)(?:\s+in\s+|\s*$)', prompt, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.,!?')
            val = re.sub(r'^(?:the|a|an|my)\s+', '', val, flags=re.IGNORECASE)
            val = re.sub(r'\s+(?:in\s+)?(?:my\s+)?contacts?\s*$', '', val, flags=re.IGNORECASE)
            if val:
                return val

    # --- artist ---
    if param_name == "artist" or "artist" in desc:
        m = re.search(r'\bby\s+(.+?)(?:\s+on\s+|$)', prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.,!?')

    return None


def _extract_integer_arg(prompt, param_name, param_info):
    """Try to extract an integer argument from the prompt."""
    if param_name in ("hour", "minute"):
        m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', prompt)
        if m:
            if param_name == "hour":
                h = int(m.group(1))
                ampm = m.group(3).upper().replace('.', '')
                if ampm == "PM" and h != 12:
                    h += 12
                elif ampm == "AM" and h == 12:
                    h = 0
                return h
            elif param_name == "minute":
                return int(m.group(2)) if m.group(2) else 0

    # "N minutes" / "N minute"
    m = re.search(rf'(\d+)\s*{re.escape(param_name)}', prompt, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Try description keywords
    desc = param_info.get("description", "").lower()
    for kw in [w for w in desc.split() if len(w) > 3]:
        m = re.search(rf'(\d+)\s*{re.escape(kw)}', prompt, re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def _regex_extract_call(prompt, tools):
    """Build a function call purely from regex extraction against the best-matching tool."""
    # Find the best matching tool
    matched_tools = [t for t in tools if _tool_matches_subprompt(t, prompt)]
    if not matched_tools:
        return None

    # Try each matched tool, pick the one where we can fill the most required args
    best_call = None
    best_filled = -1

    for tool in matched_tools:
        props = tool["parameters"].get("properties", {})
        required = tool["parameters"].get("required", [])
        args = {}

        for param_name, param_info in props.items():
            ptype = param_info.get("type", "string")
            if ptype == "string":
                val = _extract_string_arg(prompt, param_name, param_info)
                if val:
                    args[param_name] = val
            elif ptype == "integer":
                val = _extract_integer_arg(prompt, param_name, param_info)
                if val is not None:
                    args[param_name] = val

        # Check if all required args are filled
        filled_required = sum(1 for r in required if r in args)
        if filled_required == len(required) and filled_required > best_filled:
            best_filled = filled_required
            best_call = {"name": tool["name"], "arguments": args}

    return best_call


# --------------- main hybrid logic ---------------

def generate_hybrid(messages, tools, confidence_threshold=0.50):
    prompt = " ".join(m["content"] for m in messages if m["role"] == "user")

    multi = _is_multi_action(prompt) and len(tools) > 1

    if not multi:
        pruned_tools = [t for t in tools if _tool_matches_subprompt(t, prompt)]
        if not pruned_tools:
            pruned_tools = tools

        local = generate_cactus(messages, pruned_tools)
        calls = local.get("function_calls", [])

        _postprocess_calls(calls, pruned_tools)

        for call in calls:
            _fix_integer_args_from_prompt(call, prompt, pruned_tools)
            _fix_reminder_args_from_prompt(call, prompt)

        is_valid = _validate_local(local, pruned_tools)

        if is_valid:
            local["source"] = "on-device"
            return local

        # Regex fallback before cloud
        regex_call = _regex_extract_call(prompt, pruned_tools)
        if regex_call:
            _fix_reminder_args_from_prompt(regex_call, prompt)
            result = {
                "function_calls": [regex_call],
                "total_time_ms": local.get("total_time_ms", 0),
                "source": "on-device",
            }
            if _validate_local(result, pruned_tools):
                return result

        cloud = generate_cloud(messages, tools)
        _postprocess_calls(cloud.get("function_calls", []), tools)
        for call in cloud.get("function_calls", []):
            _fix_reminder_args_from_prompt(call, prompt)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud

    sub_prompts = _split_actions(prompt)

    model = cactus_init(functiongemma_path)
    sys_msg = {"role": "system", "content": SYS_PROMPT}

    all_calls = []
    total_time = 0
    used_tools = set()

    for sub in sub_prompts:
        remaining_tools = [t for t in tools if t["name"] not in used_tools]
        if not remaining_tools:
            break

        pruned_tools = [t for t in remaining_tools if _tool_matches_subprompt(t, sub)]
        if not pruned_tools:
            pruned_tools = remaining_tools

        cactus_tools = [{"type": "function", "function": t} for t in pruned_tools]

        raw_str = cactus_complete(
            model,
            [sys_msg, {"role": "user", "content": sub}],
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            confidence_threshold=0.0,
        )
        cactus_reset(model)

        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            raw = {"function_calls": [], "total_time_ms": 0}

        calls = raw.get("function_calls", [])
        total_time += raw.get("total_time_ms", 0)

        _postprocess_calls(calls, pruned_tools)

        got_valid = False
        for call in calls:
            _fix_integer_args_from_prompt(call, sub, pruned_tools)
            _fix_reminder_args_from_prompt(call, sub)
            if call["name"] in {t["name"] for t in pruned_tools}:
                all_calls.append(call)
                used_tools.add(call["name"])
                got_valid = True

        # Regex fallback for this sub-prompt if FunctionGemma failed
        if not got_valid:
            regex_call = _regex_extract_call(sub, pruned_tools)
            if regex_call:
                _fix_reminder_args_from_prompt(regex_call, sub)
                all_calls.append(regex_call)
                used_tools.add(regex_call["name"])

    cactus_destroy(model)

    if len(all_calls) == len(sub_prompts) and all_calls:
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "source": "on-device",
        }

    cloud = generate_cloud(messages, tools)
    _postprocess_calls(cloud.get("function_calls", []), tools)
    for call in cloud.get("function_calls", []):
        _fix_reminder_args_from_prompt(call, prompt)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_time
    return cloud


def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result.get("function_calls", []):
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)

PENALDO
'''
