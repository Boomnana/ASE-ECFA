

import json
import logging
import datetime
from typing import List, Dict, Any, Tuple
from httpx import TimeoutException, ConnectError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from tqdm.asyncio import tqdm_asyncio
from llm_client import LLMClient
from utils.sentence_utils import clean_llm_output
from utils.llm_conversation_logger import log_llm_conversation, get_llm_log_dir
from prompt.entity_alignment_prompt import (
    get_system_prompt_text,
    build_concept_messages_batch,
)
from entity_alignment.common import normalize_text, coarse_chunks, coarse_chunks_jaccard_merged
from llm_client import get_token_tracker
from pathlib import Path
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutException, ConnectError, ValueError)),
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutException, ConnectError, ValueError)),
)
async def _llm_concepts_for_chunks_batch(
    items: List[Dict[str, Any]],
    llm: LLMClient,
    semaphore,
    type_hint: str,
    fewshot: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    messages = build_concept_messages_batch(items, type_hint=type_hint, fewshot=fewshot)
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
        if isinstance(data, dict):
            log_llm_conversation(
                conversation_id="stage1_batch",
                function_name="semantic_bucketing_batch",
                system_prompt=get_system_prompt_text(),
                human_prompt=json.dumps(items, ensure_ascii=False),
                response_text=json.dumps(data, ensure_ascii=False),
                success=True,
                metadata={"batch_size": len(items)},
            )
            return data
    except Exception as e:
        log_llm_conversation(
            conversation_id="stage1_batch",
            function_name="semantic_bucketing_batch",
            system_prompt=get_system_prompt_text(),
            human_prompt=json.dumps(items, ensure_ascii=False),
            response_text=text,
            success=False,
            error_message=str(e),
            metadata={"batch_size": len(items)},
        )
    return {}

async def semantic_bucketing(
    entities: List[str],
    coarse_size: int,
    prefix_len: int,
    llm: LLMClient,
    semaphore,
    type_hint: str,
    fewshot_concept: List[Dict[str, Any]],
    logger: logging.Logger,
    id_prefix: str = "",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Any]]:


    chunks = coarse_chunks_jaccard_merged(entities, coarse_size, ngram_size=2, threshold=0.4, merge_threshold=0.6)
    concept_to_entities: Dict[str, List[str]] = {}
    entity_to_concepts: Dict[str, List[str]] = {}

    items = [{"id": f"{id_prefix}{i}", "entities": ch} for i, ch in enumerate(chunks)]
    batch_size = 10
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    tasks = [
        _llm_concepts_for_chunks_batch(batch, llm, semaphore, type_hint, fewshot_concept)
        for batch in batches
    ]
    results = await tqdm_asyncio.gather(*tasks)
    all_batch_map: Dict[str, Dict[str, Any]] = {}
    rid_to_entities: Dict[str, List[str]] = {}
    concept_to_ids: Dict[str, List[str]] = {}
    for batch, result_map in zip(batches, results):

        if not isinstance(result_map, dict):
            result_map = {}


        for k, v in list(result_map.items()):
            if not isinstance(v, dict):
                if isinstance(v, str):
                    result_map[k] = {"concept": v}
                else:
                    result_map[k] = {}

        try:
            all_batch_map.update(result_map or {})
        except Exception:
            pass
        for it in batch:
            rid = it["id"]
            ch = it["entities"]
            rid_to_entities[rid] = ch
            result = result_map.get(rid) or {}
            concept = normalize_text(result.get("concept", "未归类"))
            concept_to_entities.setdefault(concept, []).extend(ch)
            concept_to_ids.setdefault(concept, []).append(rid)
            for e in ch:
                entity_to_concepts.setdefault(e, []).append(concept)

    for k in concept_to_entities:
        concept_to_entities[k] = sorted(list(set(concept_to_entities[k])), key=lambda s: normalize_text(s).lower())
    ambiguous: Dict[str, List[str]] = {}
    for e, cs in entity_to_concepts.items():

        if len(set(cs)) > 1:
            ambiguous[e] = sorted(list(set(cs)))

    concept_to_entities_final: Dict[str, List[str]] = {k: v[:] for k, v in concept_to_entities.items()}
    ambiguous_final: Dict[str, List[str]] = {k: v[:] for k, v in ambiguous.items()}

    buckets_by_id: Dict[str, Dict[str, Any]] = {}
    for rid, desc in (all_batch_map or {}).items():
        concept = normalize_text((desc or {}).get("concept", "未归类"))
        buckets_by_id[rid] = {
            "concept": concept,
            "entities": rid_to_entities.get(rid, []),
            "definition": (desc or {}).get("definition", ""),
            "scope_keywords": (desc or {}).get("scope_keywords", []),
            "examples": (desc or {}).get("examples", []),
        }

    aux = {
        "buckets_by_id": buckets_by_id,
        "concept_to_ids": concept_to_ids,
    }
    return concept_to_entities_final, ambiguous_final, aux
