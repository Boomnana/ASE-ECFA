import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from llm_client import LLMClient
from utils.sentence_utils import clean_llm_output
from utils.llm_conversation_logger import log_llm_conversation
from entity_alignment.bucket_signature import find_semantic_cross_groups


try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    logging.warning("未安装 tenacity，将使用简易重试逻辑。")

logger = logging.getLogger("EntityAlignment")


def _estimate_tokens(text: str) -> int:

    if not text:
        return 0


    return len(text)


def _dynamic_chunk(
    bucket_ids: List[str],
    buckets_by_id: Dict[str, Dict[str, Any]],
    keyword: str,
    max_tokens: int = 6000,
) -> List[List[str]]:


    if len(bucket_ids) <= 1:
        return [bucket_ids] if bucket_ids else []

    system_text = "你是一位实体对齐与概念合并领域的专家。"
    base_prompt_template = (
        f'以下概念都共享作用域关键词：“{keyword}”。\n'
        '请判断其中哪些概念在语义上是等价的，应当合并。\n\n'
        '概念列表：\n{items_json}\n\n'
        '请返回一个 JSON 对象，包含键 "merges"，其值为一个列表...'
    )

    system_tokens = _estimate_tokens(system_text)
    base_tokens = _estimate_tokens(base_prompt_template.replace("{items_json}", ""))

    chunks = []
    current_chunk = []
    current_item_tokens = 0

    for bid in bucket_ids:
        b = buckets_by_id[bid]
        item_repr = {
            "id": bid,
            "concept": b.get("concept", ""),
            "definition": b.get("definition", ""),
            "examples": b.get("examples", [])
        }
        item_str = json.dumps(item_repr, ensure_ascii=False, indent=0)
        item_tokens = _estimate_tokens(item_str)


        estimated_total = (
            system_tokens +
            base_tokens +
            current_item_tokens +
            item_tokens +
            500
        )

        if estimated_total > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [bid]
            current_item_tokens = item_tokens
        else:
            current_chunk.append(bid)
            current_item_tokens += item_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


if HAS_TENACITY:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, json.JSONDecodeError, ValueError))
    )
    async def _call_llm_with_retry(llm, messages, semaphore):
        async with semaphore:
            response = await llm.agenerate([messages])
        text = response.generations[0][0].text
        cleaned = clean_llm_output(text)
        result = json.loads(cleaned)
        return result, text
else:

    async def _call_llm_with_retry(llm, messages, semaphore):
        last_error = None
        for attempt in range(3):
            try:
                async with semaphore:
                    response = await llm.agenerate([messages])
                text = response.generations[0][0].text
                cleaned = clean_llm_output(text)
                result = json.loads(cleaned)
                return result, text
            except (asyncio.TimeoutError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                continue
        raise last_error


async def _merge_buckets_single_round(
    buckets_by_id: Dict[str, Dict[str, Any]],
    llm: LLMClient,
    semaphore: asyncio.Semaphore,
    max_context_tokens: int = 8000,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:


    if not buckets_by_id:
        logger.info("输入桶为空，跳过合并。")
        return {}, {}

    total_buckets = len(buckets_by_id)
    logger.info(f"【阶段2 - 单轮合并】开始处理 {total_buckets} 个原始桶...")


    keyword_to_ids: Dict[str, Set[str]] = defaultdict(set)
    for bid, info in buckets_by_id.items():
        keywords = info.get("scope_keywords", [])
        if not keywords:
            continue
        for kw in keywords:
            if kw and isinstance(kw, str):
                keyword_to_ids[kw].add(bid)

    candidate_keywords = {k: v for k, v in keyword_to_ids.items() if len(v) > 1}


    semantic_groups = find_semantic_cross_groups(
        buckets_by_id=buckets_by_id,
        similarity_threshold=0.85,
        min_group_size=2,
    )
    all_candidate_groups: Dict[str, Set[str]] = {}
    all_candidate_groups.update(candidate_keywords)
    all_candidate_groups.update(semantic_groups)


    if not all_candidate_groups:
        logger.info("未发现可合并的组（关键词组或语义组），跳过合并。")
        buckets = {info["concept"]: info["entities"] for info in buckets_by_id.values()}
        return buckets, buckets_by_id

    nonunique_count = sum(len(ids) for ids in all_candidate_groups.values())
    unique_ids: Set[str] = set()
    for ids in all_candidate_groups.values():
        unique_ids.update(ids)
    unique_count = len(unique_ids)
    logger.info(
        f"发现 {len(candidate_keywords)} 个关键词组 和 {len(semantic_groups)} 个语义组 "
        f"可尝试合并（组成员计数={nonunique_count}，唯一桶数={unique_count}）。"
    )


    parent = {bid: bid for bid in buckets_by_id}
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            return True
        return False

    total_merge_groups = 0


    async def process_keyword_group(keyword: str, bucket_ids: Set[str]):
        nonlocal total_merge_groups
        ids_list = list(bucket_ids)
        if len(ids_list) < 2:
            return


        is_semantic_group = keyword.startswith("__SEMANTIC_GROUP__")

        chunks = _dynamic_chunk(ids_list, buckets_by_id, keyword, max_tokens=max_context_tokens - 1000)
        logger.debug(f"{'语义组' if is_semantic_group else '关键词'} '{keyword}' 拆分为 {len(chunks)} 个提示块进行处理。")

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < 2:
                continue

            items = []
            for bid in chunk:
                b = buckets_by_id[bid]
                items.append({
                    "id": bid,
                    "concept": b.get("concept"),
                    "definition": b.get("definition"),
                    "examples": b.get("examples")
                })

            system_prompt = "你是一位实体对齐与概念合并领域的专家。"

            if is_semantic_group:
                user_prompt = f"""
以下是一组语义相似的概念（通过向量模型发现），请判断它们是否在语义上等价，应当合并。

概念列表：
{json.dumps(items, ensure_ascii=False, indent=2)}

请返回一个 JSON 对象，包含键 "merges"，其值为一个列表。
每个子列表包含应合并的概念 "id"。
示例：{{ "merges": [ ["id1", "id2"] ] }}
若无需合并，请返回：{{ "merges": [] }}
"""
            else:
                user_prompt = f"""
以下概念都共享作用域关键词：“{keyword}”。
请判断其中哪些概念在语义上是等价的，应当合并。

概念列表：
{json.dumps(items, ensure_ascii=False, indent=2)}

请返回一个 JSON 对象，包含键 "merges"，其值为一个列表。
每个子列表包含应合并的概念 "id"。
示例：{{ "merges": [ ["id1", "id2"], ["id3", "id4", "id5"] ] }}
若无需合并，请返回：{{ "merges": [] }}
"""

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            merges_found = 0

            try:
                result, raw_text = await _call_llm_with_retry(llm, messages, semaphore)
                merges = result.get("merges", [])

                log_llm_conversation(
                    conversation_id="stage2_merge",
                    function_name="process_keyword_group",
                    system_prompt=system_prompt,
                    human_prompt=user_prompt,
                    response_text=raw_text,
                    success=True,
                    metadata={"keyword": keyword, "chunk": chunk_idx, "merges": merges}
                )

                for group in merges:
                    if len(group) >= 2:
                        first = group[0]
                        for other in group[1:]:
                            if first in parent and other in parent:
                                if union(first, other):
                                    merges_found += 1

                if merges_found > 0:
                    logger.info(f"✅ {'语义组' if is_semantic_group else '关键词'} '{keyword}'（块 {chunk_idx+1}）成功合并 {merges_found} 组。")
                else:
                    logger.debug(f"🔹 {'语义组' if is_semantic_group else '关键词'} '{keyword}'（块 {chunk_idx+1}）无有效合并。")

                total_merge_groups += merges_found

            except Exception as e:
                logger.warning(f"❌ 处理 {'语义组' if is_semantic_group else '关键词'} '{keyword}'（块 {chunk_idx+1}）失败：{e}")
                log_llm_conversation(
                    conversation_id="stage2_merge",
                    function_name="process_keyword_group",
                    system_prompt=system_prompt,
                    human_prompt=user_prompt,
                    response_text=str(e),
                    success=False,
                    error_message=str(e)
                )


    tasks = [process_keyword_group(kw, ids) for kw, ids in all_candidate_groups.items()]
    await asyncio.gather(*tasks)


    groups: Dict[str, List[str]] = defaultdict(list)
    for bid in buckets_by_id:
        groups[find(bid)].append(bid)

    new_buckets_by_id = {}
    new_buckets = {}

    for root, members in groups.items():
        if len(members) == 1:
            bid = members[0]
            b = buckets_by_id[bid]
            new_buckets_by_id[bid] = b
            new_buckets[b["concept"]] = b["entities"]
        else:
            base = buckets_by_id[root].copy()
            all_entities = set(base.get("entities", []))
            all_examples = set(base.get("examples", []))
            all_keywords = set(base.get("scope_keywords", []))

            for m in members:
                if m == root:
                    continue
                b = buckets_by_id[m]
                all_entities.update(b.get("entities", []))
                all_examples.update(b.get("examples", []))
                all_keywords.update(b.get("scope_keywords", []))

            base["entities"] = sorted(all_entities)
            base["examples"] = sorted(all_examples)
            base["scope_keywords"] = sorted(all_keywords)

            new_buckets_by_id[root] = base
            new_buckets[base["concept"]] = base["entities"]

    final_count = len(new_buckets)
    reduced = total_buckets - final_count
    logger.info(f"【阶段2 - 单轮合并完成】桶数量：{total_buckets} → {final_count}（减少 {reduced} 个），共执行 {total_merge_groups} 次合并操作。")
    return new_buckets, new_buckets_by_id


async def merge_buckets(
    buckets_by_id: Dict[str, Dict[str, Any]],
    llm: LLMClient,
    max_concurrent: int = 5,
    semaphore: Optional[asyncio.Semaphore] = None,
    iterative: bool = False,
    max_rounds: int = 3,
    max_context_tokens: int = 8000,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:


    sem = semaphore if semaphore is not None else asyncio.Semaphore(max_concurrent)
    if not iterative:
        return await _merge_buckets_single_round(
            buckets_by_id=buckets_by_id,
            llm=llm,
            semaphore=sem,
            max_context_tokens=max_context_tokens,
        )
    current_buckets_by_id = buckets_by_id
    last_count = len(current_buckets_by_id) if current_buckets_by_id else 0
    for round_idx in range(max_rounds):
        new_buckets, new_buckets_by_id = await _merge_buckets_single_round(
            buckets_by_id=current_buckets_by_id,
            llm=llm,
            semaphore=sem,
            max_context_tokens=max_context_tokens,
        )
        new_count = len(new_buckets_by_id)
        if new_count == last_count:
            logger.info(f"【阶段2 - 迭代合并提前停止】第 {round_idx+1} 轮无桶数量变化。")
            current_buckets_by_id = new_buckets_by_id
            break
        current_buckets_by_id = new_buckets_by_id
        last_count = new_count
        logger.info(f"【阶段2 - 迭代合并】第 {round_idx+1} 轮合并后桶数: {new_count}")
    final_buckets_by_id = current_buckets_by_id
    final_buckets = {info['concept']: info['entities'] for info in final_buckets_by_id.values()}
    return final_buckets, final_buckets_by_id
