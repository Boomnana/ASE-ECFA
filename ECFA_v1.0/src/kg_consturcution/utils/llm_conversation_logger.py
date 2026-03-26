import json
import logging
import os
import datetime
from typing import Any, Dict, Optional


_LOG_FILE_PATH: Optional[str] = None

def setup_llm_logger(log_dir: str, file_name: str = "llm_interaction_logs.txt"):


    global _LOG_FILE_PATH
    os.makedirs(log_dir, exist_ok=True)
    _LOG_FILE_PATH = os.path.join(log_dir, file_name)


    with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== LLM Interaction Log Started at {datetime.datetime.now().isoformat()} ===\n\n")

def log_llm_conversation(
    conversation_id: str,
    function_name: str,
    system_prompt: str,
    human_prompt: str,
    response_text: str,
    success: bool,
    error_message: str = "",
    metadata: Dict[str, Any] = None
):


    if _LOG_FILE_PATH is None:

        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "FAILED"

    log_entry = (
        f"[{timestamp}] [ID: {conversation_id}] [Func: {function_name}] [{status}]\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False) if metadata else '{}'}\n"
        f"--------------------------------------------------\n"
        f"System Prompt:\n{system_prompt}\n"
        f"--------------------------------------------------\n"
        f"Human Prompt:\n{human_prompt}\n"
        f"--------------------------------------------------\n"
        f"Response:\n{response_text}\n"
    )

    if not success:
        log_entry += f"--------------------------------------------------\nError: {error_message}\n"

    log_entry += "==================================================\n\n"

    try:
        with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write to LLM log: {e}")

def get_llm_log_dir() -> Optional[str]:
    if _LOG_FILE_PATH:
        return os.path.dirname(_LOG_FILE_PATH)
    return None
