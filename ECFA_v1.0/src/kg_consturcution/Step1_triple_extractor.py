import asyncio
import logging
import traceback
import argparse
import json
from pathlib import Path
from httpx import TimeoutException, ConnectError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from utils.sentence_utils import clean_llm_output
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from llm_client import LLMClient, LLMSettings
from prompt.triple_extraction_prompt import build_messages

import datetime


timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
output_dir_with_time = BASE_DIR / "output" / f"batch_{timestamp_str}"
output_dir_with_time.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", default=str("input/RQ1根数据集.xlsx"))
parser.add_argument("--output-file", default=str(output_dir_with_time / "kg_extraction_results.xlsx"))
parser.add_argument("--triple-path", default=str(output_dir_with_time / "triples.xlsx"))
parser.add_argument("--log-file", default=str(output_dir_with_time / "Step1_log_extractor.txt"))
parser.add_argument("--sheet-name", default="途牛")
parser.add_argument("--llm-config", default=str(REPO_ROOT / "LLM-config.yaml"))
parser.add_argument("--batch-size", type=int, default=3, help="Batch size for LLM extraction")
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("zai").setLevel(logging.WARNING)
logger = logging.getLogger("ZhipuExtractor")
before_sleep = before_sleep_log(logger, logging.WARNING)

settings = LLMSettings.from_yaml(args.llm_config)
llm = LLMClient.from_yaml(args.llm_config).build()


semaphore = asyncio.Semaphore(settings.concurrency)

@retry(
    stop=stop_after_attempt(1),
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((TimeoutException, ConnectError, ValueError)),
    before_sleep=before_sleep
)
async def _generate_and_extract_batch(batch_items):


    messages = build_messages(batch_items)

    response = await llm.agenerate([messages])
    raw_result = response.generations[0][0].text.strip()


    try:
        input_content = "\n".join([m.content for m in messages])
        llm.log_interaction(input_content, raw_result, log_file=args.log_file)
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")


    try:
        cleaned_json = clean_llm_output(raw_result)
        results_list = json.loads(cleaned_json)
        if not isinstance(results_list, list):
            raise ValueError("Parsed JSON is not a list")
        return results_list
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")

async def process_batch_task(batch_items):


    await asyncio.sleep(2.0)
    async with semaphore:
        try:
            results_list = await _generate_and_extract_batch(batch_items)


            results_map = {str(item.get("report_id")): item.get("triples", []) for item in results_list}

            output_rows = []
            for item in batch_items:
                rid = str(item["report_id"])
                triples = results_map.get(rid, [])


                status = 'success'
                if not triples:
                    status = 'success_no_triples'

                output_rows.append({
                    'index': item['original_index'],
                    'id': rid,
                    'raw_input': item['text'],
                    'triples': json.dumps(triples, ensure_ascii=False),
                    'status': status
                })
            return output_rows

        except Exception as e:

            output_rows = []
            for item in batch_items:
                 output_rows.append({
                    'index': item['original_index'],
                    'id': item['report_id'],
                    'raw_input': item['text'],
                    'triples': "[]",
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                })
            return output_rows

input_path = Path(args.input_file)
output_path = Path(args.output_file)
triple_path = Path(args.triple_path)
df = pd.read_excel(input_path, sheet_name=args.sheet_name)

async def run_batch_execution(items, batch_size, desc="Processing"):
    tasks = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        tasks.append(process_batch_task(batch))

    results_nested = await tqdm_asyncio.gather(*tasks, desc=desc, total=len(tasks), leave=True)
    return [item for sublist in results_nested for item in sublist]


async def run_extraction_pipeline():
    BATCH_SIZE = args.batch_size


    all_items = []
    for idx, row in df.iterrows():
        description = str(row.get("description", ""))
        description = description.strip()
        if not description:
            logger.warning(f"跳过空描述，ID: {row.get('id', '')}")
            continue

        report_id = str(row.get("id", idx))
        all_items.append({
            "report_id": report_id,
            "text": description,
            "original_index": idx
        })


    print(f"🚀 Starting extraction for {len(all_items)} items (Batch Size: {BATCH_SIZE})...")
    flat_results = await run_batch_execution(all_items, BATCH_SIZE, desc="Initial Extraction")


    MAX_RETRIES = 3
    results_map = {str(r['id']): r for r in flat_results}

    for attempt in range(1, MAX_RETRIES + 1):
        failed_ids = [rid for rid, res in results_map.items() if res.get('status') == 'failed']
        if not failed_ids:
            break

        print(f"\n🔄 Retry Attempt {attempt}/{MAX_RETRIES}: Found {len(failed_ids)} failed items.")


        retry_items = []
        for rid in failed_ids:
            failed_res = results_map[rid]
            retry_items.append({
                "report_id": rid,
                "text": failed_res['raw_input'],
                "original_index": failed_res['index']
            })


        retry_results = await run_batch_execution(retry_items, BATCH_SIZE, desc=f"Retry {attempt}")


        recovered_count = 0
        for res in retry_results:
            if res.get('status') != 'failed':
                results_map[str(res['id'])] = res
                recovered_count += 1

        print(f"   -> Recovered {recovered_count}/{len(failed_ids)} items.")


        current_results = list(results_map.values())
        result_df = pd.DataFrame(current_results).sort_values('index')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(output_path, index=False)


    final_results = list(results_map.values())
    result_df = pd.DataFrame(final_results).sort_values('index')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(output_path, index=False)


    shared_output_dir = REPO_ROOT / "shared_output"
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(shared_output_dir / "kg_extraction_results.xlsx", index=False)
    print(f"✅ Results also saved to {shared_output_dir / 'kg_extraction_results.xlsx'}")

    fail_count = len([r for r in final_results if r.get('status') == 'failed'])
    if fail_count == 0:
        print(f"\n✅ All items processed successfully! Results saved to {str(output_path)}")
    else:
        print(f"\n⚠️ Completed with {fail_count} failures. Results saved to {str(output_path)}")

    return result_df

def extract_triples_from_json_list(df):
    all_triples = []
    for _, row in df.iterrows():
        try:
            triples_str = row.get('triples', '[]')
            if not isinstance(triples_str, str):
                triples_str = '[]'
            triples = json.loads(triples_str)

            for t in triples:
                all_triples.append({
                    'subject': t.get('subject'),
                    'subject_type': t.get('subject_type'),
                    'relation': t.get('relation'),
                    'object': t.get('object'),
                    'object_type': t.get('object_type'),
                    'source_row_index': row.get('id'),
                    'raw_input': row.get('raw_input')
                })
        except Exception:
            continue
    return pd.DataFrame(all_triples)


if __name__ == "__main__":
    result_df = asyncio.run(run_extraction_pipeline())
    triples_df = extract_triples_from_json_list(result_df)
    triple_path.parent.mkdir(parents=True, exist_ok=True)
    triples_df.to_excel(triple_path, index=False)


    shared_output_dir = REPO_ROOT / "shared_output"
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    triples_df.to_excel(shared_output_dir / "triples.xlsx", index=False)
    print(f"✅ Triples also saved to {shared_output_dir / 'triples.xlsx'}")

    print(f"✅ 三元组已成功提取并保存至 {str(triple_path)}")
