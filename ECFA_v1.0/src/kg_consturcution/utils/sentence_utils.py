from typing import Optional, Any
import pandas as pd
import json
import re
import unicodedata

def extract_content_between_plus(text: str) -> Optional[str]:


    start_marker = "+++"
    end_marker = "+++"

    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None

    start_idx += len(start_marker)

    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None

    return text[start_idx:end_idx].strip()


def extract_triples_from_df(df: pd.DataFrame) -> pd.DataFrame:


    result = []


    for _, row in df.iterrows():
        source_index = row['index']
        triples_str = row['triples']
        raw_input = row['raw_input']


        if not isinstance(triples_str, str):
            continue


        triple_list = triples_str.split('###')

        for t in triple_list:
            parts = t.split('￨')
            if len(parts) == 5:
                subject, subject_type, relation, obj, object_type = parts
                if '实体' in subject_type:
                    subject_type = subject_type.replace('实体', '')
                if '实体' in object_type:
                    object_type = object_type.replace('实体', '')
                result.append({
                    'subject': subject,
                    'subject_type': subject_type,
                    'relation': relation,
                    'object': obj,
                    'object_type': object_type,
                    'source_row_index': source_index,
                    'raw_input': raw_input
                })


    return pd.DataFrame(result)

def clean_llm_output(text: str) -> str:


    text = text.strip()


    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:

        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()


    start_brace = text.find('{')
    start_bracket = text.find('[')


    start_idx = -1
    is_array = False

    if start_brace != -1 and start_bracket != -1:
        if start_brace < start_bracket:
            start_idx = start_brace
        else:
            start_idx = start_bracket
            is_array = True
    elif start_brace != -1:
        start_idx = start_brace
    elif start_bracket != -1:
        start_idx = start_bracket
        is_array = True

    if start_idx != -1:
        if is_array:
            end_idx = text.rfind(']')
        else:
            end_idx = text.rfind('}')

        if end_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx+1]


            json_text = re.sub(r',\s*}', '}', json_text)
            json_text = re.sub(r',\s*]', ']', json_text)

            return json_text

    return text

def _normalize_quotes_commas(s: str) -> str:
    s = s.replace('“', '"').replace('”', '"').replace('‘', '"').replace('’', '"')
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s

def _to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace("\ufeff", "")

def _python_to_json_literals(s: str) -> str:
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    return s

def _quote_unquoted_keys(s: str) -> str:
    pattern = re.compile(r'([\{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:', re.UNICODE)
    return pattern.sub(lambda m: f'{m.group(1)}"{m.group(2)}":', s)

def _balance_brackets(s: str) -> str:
    obj_diff = s.count('{') - s.count('}')
    arr_diff = s.count('[') - s.count(']')
    if obj_diff > 0:
        s = s + ('}' * obj_diff)
    if arr_diff > 0:
        s = s + (']' * arr_diff)
    return s

def robust_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    s = _to_halfwidth(text.strip())
    s = _normalize_quotes_commas(s)
    s = _python_to_json_literals(s)
    s = _quote_unquoted_keys(s)
    s = _balance_brackets(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    if "```" in s:
        if "```json" in s:
            st = s.find("```json") + 7
            ed = s.find("```", st)
            if ed != -1:
                s2 = _to_halfwidth(s[st:ed].strip())
                s2 = _normalize_quotes_commas(s2)
                s2 = _python_to_json_literals(s2)
                s2 = _quote_unquoted_keys(s2)
                s2 = _balance_brackets(s2)
                try:
                    return json.loads(s2)
                except Exception:
                    pass
        st = s.find("```") + 3
        ed = s.find("```", st)
        if ed != -1:
            s2 = _to_halfwidth(s[st:ed].strip())
            s2 = _normalize_quotes_commas(s2)
            s2 = _python_to_json_literals(s2)
            s2 = _quote_unquoted_keys(s2)
            s2 = _balance_brackets(s2)
            try:
                return json.loads(s2)
            except Exception:
                pass
    try:
        cleaned = _to_halfwidth(clean_llm_output(s))
        cleaned = _normalize_quotes_commas(cleaned)
        cleaned = _python_to_json_literals(cleaned)
        cleaned = _quote_unquoted_keys(cleaned)
        cleaned = _balance_brackets(cleaned)
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        m = re.search(r'\{[\s\S]*\}', s)
        if m:
            s2 = _to_halfwidth(m.group(0))
            s2 = _normalize_quotes_commas(s2)
            s2 = _python_to_json_literals(s2)
            s2 = _quote_unquoted_keys(s2)
            s2 = _balance_brackets(s2)
            return json.loads(s2)
    except Exception:
        pass
    try:
        recovered = _recover_clusters_best_effort(s)
        if recovered:
            return {"clusters": recovered}
    except Exception:
        pass
    raise ValueError("无法解析JSON")

def try_parse_json(text: str) -> Any:
    return robust_parse_json(text)

def _recover_clusters_best_effort(text: str):
    s = text
    idx = 0
    out = []
    while True:
        start = s.find('{"standard_name', idx)
        if start == -1:
            break
        brace = 0
        end = start
        while end < len(s):
            if s[end] == '{':
                brace += 1
            elif s[end] == '}':
                brace -= 1
                if brace == 0:
                    end += 1
                    break
            end += 1
        chunk = s[start:end]
        idx = end
        try:
            obj = json.loads(clean_llm_output(chunk))
            out.append(obj)
        except Exception:
            continue
    return out if out else None
