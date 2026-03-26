import os
import asyncio
from dataclasses import dataclass
import yaml
from langchain_community.chat_models import ChatZhipuAI
from langchain.schema import LLMResult, Generation
import logging
import re
from pathlib import Path

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

@dataclass
class LLMSettings:
    api_key: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 5000
    concurrency: int = 5
    timeout: int = 30
    max_retries: int = 2

    @classmethod
    def from_yaml(cls, path: str) -> "LLMSettings":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise ValueError(f"配置文件未找到: {path}") from e
        except Exception as e:
            raise ValueError(f"读取配置文件失败: {e}") from e

        try:
            model = data.get("model", {})
            settings = cls(
                api_key=str(model.get("key", "")),
                model_name=str(model.get("name", "")),
                temperature=float(model.get("temperature", 0.2)),
                max_tokens=int(model.get("max_tokens", 5000)),
                concurrency=int(model.get("concurrency", 5)),
                timeout=int(model.get("timeout", 30)),
                max_retries=int(model.get("max_retries", 2)),
            )
        except Exception as e:
            raise ValueError(f"解析配置字段失败: {e}") from e

        if not settings.api_key or not settings.model_name:
            raise ValueError("配置缺失: api_key 与 model_name 不能为空")

        return settings

class ChatLLMAdapter:
    def __init__(self, inner: ChatZhipuAI, timeout: int, max_retries: int, concurrency: int):
        self.inner = inner
        self.timeout = timeout
        self.max_retries = max_retries if max_retries and max_retries > 0 else 1
        self.concurrency = concurrency if concurrency and concurrency > 0 else 1
        self._token_logger = logging.getLogger("TokenUsage")

    def log_interaction(self, input_text: str, output_text: str, log_file: str = "llm_interaction.log"):

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=== {timestamp} ===\n")
                f.write(f"--- INPUT ---\n{input_text}\n")
                f.write(f"--- OUTPUT ---\n{output_text}\n")
                f.write("="*50 + "\n\n")
        except Exception as e:
            logging.getLogger("LLMClient").error(f"Failed to write interaction log: {e}")

    async def agenerate(self, messages_list, **kwargs):
        last_err = None
        model_name = getattr(self.inner, "model_name", getattr(self.inner, "model", ""))
        is_glm47 = "glm-4.7" in str(model_name)
        total_attempts = self.max_retries if self.max_retries > 0 else 1

        for attempt in range(total_attempts):
            try:
                is_zai = isinstance(self.inner, ZaiGLMAdapter) or hasattr(self.inner, "client")

                if is_zai:
                    result = await self.inner.agenerate(messages_list, **kwargs)
                else:
                     result = await asyncio.wait_for(
                        self.inner.agenerate(messages_list, **kwargs),
                        timeout=self.timeout,
                    )

                try:
                    prompt_tokens, completion_tokens = _extract_usage(result, messages_list)
                    tracker = get_token_tracker()
                    tracker.record(prompt_tokens, completion_tokens)
                    tracker.report(self._token_logger)
                except Exception:
                    pass
                return result
            except Exception as e:
                last_err = e
                if attempt < total_attempts - 1:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "1302" in err_str

                    if is_glm47:
                        wait_time = 1
                        log_level = logging.INFO
                    else:
                        if is_rate_limit:
                            wait_time = 5 * (2 ** attempt)
                            log_level = logging.INFO
                        else:
                            wait_time = 2 * (2 ** attempt)
                            log_level = logging.WARNING

                    wait_time = min(wait_time, 60)

                    logging.getLogger("LLMClient").log(
                        log_level,
                        f"LLM request failed (attempt {attempt+1}/{total_attempts}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

        raise last_err

    def __getattr__(self, name):
        return getattr(self.inner, name)

class ZaiGLMAdapter:

    def __init__(self, api_key: str, model_name: str, temperature: float):
        if not ZhipuAiClient:
            raise ImportError("zai-sdk not installed. Run `pip install zai-sdk`")


        try:
            self.client = ZhipuAiClient(api_key=api_key, max_retries=0)
        except TypeError:

            self.client = ZhipuAiClient(api_key=api_key)

        self.model_name = model_name
        self.temperature = temperature

    async def agenerate(self, messages_list, **kwargs):


        if not messages_list:
            return LLMResult(generations=[])

        messages = messages_list[0]
        zai_messages = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            elif m.type == "human":
                role = "user"
            zai_messages.append({"role": role, "content": m.content})


        loop = asyncio.get_running_loop()

        def _call():

            extra_params = {}
            if "glm-4.7" in self.model_name:

                extra_params["thinking"] = {"type": "disabled"}


            valid_kwargs = {k: v for k, v in kwargs.items() if k not in ["stop"]}

            return self.client.chat.completions.create(
                model=self.model_name,
                messages=zai_messages,
                temperature=self.temperature,
                **extra_params,
                **valid_kwargs
            )

        try:
            response = await loop.run_in_executor(None, _call)

            text = response.choices[0].message.content
            usage = response.usage

            generation = Generation(text=text)

            llm_output = {
                "token_usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }
            return LLMResult(generations=[[generation]], llm_output=llm_output)
        except Exception as e:
            raise e

class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        if self.settings.api_key:
            os.environ["ZHIPUAI_API_KEY"] = self.settings.api_key
        self._client = None

    @classmethod
    def from_yaml(cls, path: str) -> "LLMClient":
        try:
            settings = LLMSettings.from_yaml(path)
            return cls(settings)
        except Exception as e:
            raise ValueError(f"初始化 LLMClient 失败: {e}") from e

    def build(self):
        if self._client is not None:
            return self._client

        if not self.settings.api_key or not self.settings.model_name:
            raise ValueError("配置缺失: api_key 与 model_name 不能为空")

        try:


            use_zai = "glm-4.7" in self.settings.model_name and ZhipuAiClient is not None

            if use_zai:
                inner = ZaiGLMAdapter(
                    api_key=self.settings.api_key,
                    model_name=self.settings.model_name,
                    temperature=self.settings.temperature,
                )
            else:
                inner = ChatZhipuAI(
                    model=self.settings.model_name,
                    temperature=self.settings.temperature,
                )

            self._client = ChatLLMAdapter(
                inner=inner,
                timeout=self.settings.timeout,
                max_retries=self.settings.max_retries,
                concurrency=self.settings.concurrency,
            )
            return self._client
        except Exception as e:
            raise RuntimeError(f"LLM 构建失败: {e}") from e

class TokenTracker:
    def __init__(self):
        self.prompt_total = 0
        self.completion_total = 0
        self.calls = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.last_prompt_tokens = int(prompt_tokens or 0)
        self.last_completion_tokens = int(completion_tokens or 0)
        self.prompt_total += self.last_prompt_tokens
        self.completion_total += self.last_completion_tokens
        self.calls += 1

    def report(self, logger: logging.Logger = None):
        msg = (
            f"tokens 本次: prompt={self.last_prompt_tokens}, completion={self.last_completion_tokens}; "
            f"累计: prompt={self.prompt_total}, completion={self.completion_total}, 合计={self.prompt_total + self.completion_total}, 次数={self.calls}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

_global_token_tracker = TokenTracker()

def get_token_tracker() -> TokenTracker:
    return _global_token_tracker

def _extract_usage(result, messages_list) -> tuple:
    try:
        llm_output = getattr(result, "llm_output", None)
        if isinstance(llm_output, dict):

            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}


            prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            if prompt or completion:
                return int(prompt), int(completion)
    except Exception:
        pass


    prompt_text = "\n".join([m.content for m in (messages_list[0] if messages_list else []) if hasattr(m, "content")])
    completion_text = ""
    try:
        completion_text = result.generations[0][0].text or ""
    except Exception:
        completion_text = ""
    return _estimate_tokens(prompt_text), _estimate_tokens(completion_text)

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_words = len(re.findall(r"[A-Za-z_0-9]+", text))
    non_ascii = sum(1 for ch in text if ord(ch) > 127 and not ch.isspace())
    return ascii_words + non_ascii
