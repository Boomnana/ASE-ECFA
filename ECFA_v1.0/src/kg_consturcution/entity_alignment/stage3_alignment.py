

import json
import logging
from typing import List, Dict, Any, Tuple
from httpx import TimeoutException, ConnectError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llm_client import LLMClient, get_token_tracker
from utils.sentence_utils import clean_llm_output
from utils.llm_conversation_logger import log_llm_conversation
from prompt.entity_alignment_prompt import (
    build_pairs_messages,
    build_batch_validator_messages,
)
from entity_alignment.common import normalize_text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutException, ConnectError, ValueError)),
)
async def _llm_pairs_for_bucket(
    concept: str,
    entities: List[str],
    llm: LLMClient,
    semaphore,
    type_hint: str,
    fewshot: List[Dict[str, Any]],
) -> List[List[Any]]:


    messages = build_pairs_messages(concept, entities, type_hint=type_hint, fewshot=fewshot)

    async with semaphore:
        response = await llm.agenerate([messages])

    try:
        get_token_tracker().report(logging.getLogger("TokenUsage"))
    except Exception:
        pass

    text = response.generations[0][0].text
    cleaned = clean_llm_output(text)

    try:

        data = json.loads(cleaned)
        result = data if isinstance(data, list) else []


        log_llm_conversation(
            conversation_id="stage2_generation",
            function_name="_llm_pairs_for_bucket",
            system_prompt=messages[0].content if messages else "",
            human_prompt=messages[1].content if len(messages) > 1 else "",
            response_text=cleaned,
            success=True,
            metadata={"concept": concept, "entity_count": len(entities)}
        )
        return result
    except Exception as e:
        logging.getLogger("EntityAlignment").warning(f"Failed to parse pairs for concept {concept}: {e}")

        log_llm_conversation(
            conversation_id="stage2_generation",
            function_name="_llm_pairs_for_bucket",
            system_prompt=messages[0].content if messages else "",
            human_prompt=messages[1].content if len(messages) > 1 else "",
            response_text=text,
            success=False,
            error_message=str(e),
            metadata={"concept": concept, "entity_count": len(entities)}
        )
        return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutException, ConnectError, ValueError)),
)
async def _llm_validate_batch(
    batch_items: List[Dict[str, Any]],
    llm: LLMClient,
    semaphore,
) -> Dict[str, str]:


    if not batch_items:
        return {}

    messages = build_batch_validator_messages(batch_items)

    async with semaphore:
        response = await llm.agenerate([messages])

    try:
        get_token_tracker().report(logging.getLogger("TokenUsage"))
    except Exception:
        pass

    text = response.generations[0][0].text
    cleaned = clean_llm_output(text)

    results = {}
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):

            for k, v in data.items():
                val = str(v).strip()
                if "通过" in val:
                    results[str(k)] = "通过"
                else:
                    results[str(k)] = "否决"


        log_llm_conversation(
            conversation_id="stage2_validation",
            function_name="_llm_validate_batch",
            system_prompt=messages[0].content if messages else "",
            human_prompt=messages[1].content if len(messages) > 1 else "",
            response_text=cleaned,
            success=True,
            metadata={"batch_size": len(batch_items)}
        )
    except Exception as e:
        logging.getLogger("EntityAlignment").error(f"Batch validation failed (parsing error): {e}. Text: {text[:100]}...")

        log_llm_conversation(
            conversation_id="stage2_validation",
            function_name="_llm_validate_batch",
            system_prompt=messages[0].content if messages else "",
            human_prompt=messages[1].content if len(messages) > 1 else "",
            response_text=text,
            success=False,
            error_message=str(e),
            metadata={"batch_size": len(batch_items)}
        )

        pass

    return results

async def intra_bucket_alignment(
    concept: str,
    entities: List[str],
    limit: int,
    llm: LLMClient,
    semaphore,
    type_hint: str,
    fewshot_pairs: List[Dict[str, Any]],
    batch_validation_size: int = 20,
) -> List[Tuple[str, str]]:


    if len(entities) == 0:
        return []
    if len(entities) == 1:

        return []
    pairs_raw: List[List[Any]] = []
    try:

        if len(entities) <= limit:
            pairs_raw = await _llm_pairs_for_bucket(concept, entities, llm, semaphore, type_hint, fewshot_pairs)
        else:

            idx = 0
            while idx < len(entities):
                sub = entities[idx: idx + limit]
                if not sub:
                    break
                try:
                    part = await _llm_pairs_for_bucket(concept, sub, llm, semaphore, type_hint, fewshot_pairs)
                except Exception:
                    part = []
                if part:
                    pairs_raw.extend(part)
                idx += limit
    except Exception:

        pairs_raw = []


    candidates: List[Dict[str, Any]] = []
    for i, item in enumerate(pairs_raw):

        if not isinstance(item, list) or len(item) < 4:
            continue
        a, b, score, reason = item[0], item[1], item[2], item[3]
        try:

            s = float(score)
        except Exception:
            continue

        if s >= 0.5:
            candidates.append({
                "id": str(i),
                "pair": (str(a), str(b)),
                "reason": str(reason),
                "score": s,
                "raw_item": item
            })

    validated: List[Tuple[str, str]] = []


    batches = [candidates[i:i + batch_validation_size] for i in range(0, len(candidates), batch_validation_size)]

    for batch in batches:

        batch_req = [{"id": c["id"], "pair": c["pair"], "reason": c["reason"]} for c in batch]
        try:
            batch_results = await _llm_validate_batch(batch_req, llm, semaphore)
        except Exception:
            batch_results = {}


        for c in batch:
            cid = c["id"]
            s = c["score"]

            verdict = batch_results.get(cid, "否决")


            if verdict == "通过" and s >= 0.6:
                validated.append((normalize_text(c["pair"][0]), normalize_text(c["pair"][1])))

    return validated
