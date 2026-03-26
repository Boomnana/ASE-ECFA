from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from .common import BoundaryArtifacts
from .methods import rank_text_baselines
from .representations import RepresentationConfig, RepresentationStore
from .utils import read_clusters_csv


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MAX_PER_ITERATION = 50
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_SOFT_LIMIT = 90000

SYSTEM_PROMPT = """You are a professional mobile app developer and reviewer of mobile app crowdsourced test reports.
Your task is to cluster crowdsourced test reports about ONE mobile app by identifying those that point to the same bug.

The rules for clustering are as follows:
1. Reports with similar triggering operations or app behavior point to the same bug.
2. Reports with different problems with the same functionality point to different bugs.
3. One report should be included in ONLY ONE cluster.

When receiving new reports, you should think step by step to cluster them according to the following guide:
(i) Analyze the newly input reports without repeating their content.
(ii) Assign a report to an existing suitable cluster if one exists.
(iii) Otherwise, create a new cluster for it.
(iv) Prefer adding reports to existing clusters when appropriate, to avoid overly fine-grained clustering.

Your response must ALWAYS end with EXACTLY ONE line in the following format:
CURRENT CLUSTERS: Cluster 1 {1, 4, 5}, Cluster 2 {2, 3, 6}
The numbers in braces are the report numbers. Do not omit any cluster in that final line.
"""

SUMMARY_PROMPT = """You are a professional mobile app developer and reviewer of mobile app crowdsourced test reports.
Given several reports that have already been grouped into the same bug cluster, write one concise one-sentence summary of the common issue.
Requirements:
1. Focus on the shared bug, not on noisy details like devices unless essential.
2. Keep it to one sentence.
3. Output only the sentence, without bullets, labels, or quotation marks.
"""


@dataclass
class ReportItem:
    report_no: int
    report_id: Any
    app: str
    text: str
    row_data: Dict[str, Any]


@dataclass
class ClusterResult:
    cluster_id: int
    members: List[int]
    member_report_ids: List[Any]
    summary: str = ""


class ChatBackend:
    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        raise NotImplementedError


class ZhipuChatBackend(ChatBackend):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        api_url: str = API_URL,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retries: int = 5,
        thinking_enabled: bool = False,
        verify_ssl: bool = True,
        use_env_proxy: bool = False,
        verbose: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.thinking_enabled = thinking_enabled
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.trust_env = bool(use_env_proxy)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        print(str(msg), file=sys.stderr, flush=True)

    def _payload(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking_enabled and self.model.startswith("glm-4.7"):
            payload["thinking"] = {"type": "enabled"}
        return payload

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._payload(messages)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                t0 = time.time()
                self._log(f"[llm] request attempt {attempt}/{self.retries} model={self.model}")
                resp = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                dt = time.time() - t0
                if resp.status_code >= 400:
                    msg = resp.text[:1000]
                    self._log(f"[llm] http_error status={resp.status_code} elapsed={dt:.2f}s body_preview={msg!r}")
                    retryable = int(resp.status_code) in {429, 500, 502, 503, 504}
                    if not retryable:
                        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
                    if attempt >= self.retries:
                        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and str(retry_after).strip().isdigit():
                        sleep_s = int(str(retry_after).strip())
                    else:
                        sleep_s = min(2 ** (attempt - 1), 60)
                        if int(resp.status_code) == 429:
                            sleep_s = max(sleep_s, 15)
                    self._log(f"[llm] retryable_http status={resp.status_code} retry_in={sleep_s}s")
                    time.sleep(float(sleep_s))
                    continue
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("No choices in response")
                message = choices[0].get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if text:
                                parts.append(str(text))
                        elif part:
                            parts.append(str(part))
                    content = "\n".join(parts)
                content = str(content).strip()
                if not content:
                    raise RuntimeError("Empty content in response")
                self._log(f"[llm] ok chars={len(content)} elapsed={dt:.2f}s")
                return content
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                if isinstance(exc, RuntimeError) and msg.startswith("HTTP "):
                    m = re.match(r"HTTP\s+(\d+)\s*:", msg)
                    status = int(m.group(1)) if m else None
                    if status is not None and 400 <= status < 500 and status != 429:
                        break
                if attempt >= self.retries:
                    break
                sleep_s = min(2 ** (attempt - 1), 8)
                self._log(f"[llm] failed attempt {attempt}/{self.retries}: {type(exc).__name__}: {exc}. retry_in={sleep_s}s")
                time.sleep(sleep_s)
        suffix = ""
        if isinstance(last_err, requests.exceptions.ProxyError):
            suffix = " (proxy error: consider unsetting HTTP(S)_PROXY or set llm_use_env_proxy=0)"
        raise RuntimeError(f"Zhipu API call failed after {self.retries} attempts: {last_err}{suffix}")


class MockChatBackend(ChatBackend):
    def __init__(self, report_lookup: Dict[int, ReportItem]) -> None:
        self.report_lookup = report_lookup

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "Write one concise one-sentence summary" in last_user or "Cluster member reports" in last_user:
            texts = re.findall(r"Report \d+: (.+)", last_user)
            text = texts[0] if texts else ""
            short = re.sub(r"\s+", " ", text)[:40]
            return f"Reports in this cluster describe the same issue: {short}"

        current_clusters = parse_clusters_from_text(last_user)
        batch = _extract_report_texts_from_prompt(last_user)
        by_key: Dict[str, List[int]] = {}
        for report_no, text in batch.items():
            key = self.report_lookup[report_no].row_data.get("issue_key")
            if not key:
                key = _simple_key(text)
            by_key.setdefault(str(key), []).append(report_no)

        next_cluster_id = max(current_clusters.keys(), default=0) + 1
        key_to_cluster: Dict[str, int] = {}
        existing_key_to_cluster: Dict[str, int] = {}
        for cid, members in current_clusters.items():
            for m in members:
                issue_key = self.report_lookup[m].row_data.get("issue_key")
                if issue_key:
                    existing_key_to_cluster[str(issue_key)] = cid

        for key, members in sorted(by_key.items()):
            if key in existing_key_to_cluster:
                cid = existing_key_to_cluster[key]
            else:
                cid = key_to_cluster.get(key)
                if cid is None:
                    cid = next_cluster_id
                    next_cluster_id += 1
                    key_to_cluster[key] = cid
            current_clusters.setdefault(cid, [])
            current_clusters[cid].extend(members)

        cluster_line = format_clusters_line(current_clusters)
        return "Mock analysis completed.\n" + cluster_line


class LLMClusterRunner:
    def __init__(
        self,
        backend: ChatBackend,
        *,
        max_per_iteration: int = DEFAULT_MAX_PER_ITERATION,
        seed: int = 42,
        text_column: str = "auto",
        summary_sample_size: int = 8,
        context_soft_limit: int = DEFAULT_CONTEXT_SOFT_LIMIT,
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.max_per_iteration = max_per_iteration
        self.seed = seed
        self.text_column = text_column
        self.summary_sample_size = summary_sample_size
        self.context_soft_limit = context_soft_limit
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        print(str(msg), file=sys.stderr, flush=True)

    def run_app(self, app_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        reports = self._prepare_reports(app_name, df)
        if not reports:
            return {
                "app": app_name,
                "clusters": [],
                "assignments": pd.DataFrame(),
                "iteration_logs": [],
                "n_reports": 0,
            }

        self._log(f"[llmcluster] app={app_name} n_reports={len(reports)} start")
        current_clusters, logs = self._iterative_clustering(app_name, reports)
        corrected = apply_corrections(current_clusters, [r.report_no for r in reports])
        summaries = self._generate_summaries(reports, corrected)
        cluster_results = build_cluster_results(reports, corrected, summaries)
        assignments = build_assignment_frame(reports, cluster_results)
        self._log(f"[llmcluster] app={app_name} done n_clusters={len(corrected)}")
        return {
            "app": app_name,
            "clusters": cluster_results,
            "assignments": assignments,
            "iteration_logs": logs,
            "n_reports": len(reports),
        }

    def _prepare_reports(self, app_name: str, df: pd.DataFrame) -> List[ReportItem]:
        text_col = choose_text_column(df, self.text_column)
        reports: List[ReportItem] = []
        for idx, row in df.reset_index(drop=True).iterrows():
            text = safe_str(row.get(text_col))
            if not text:
                text = safe_str(row.get("description"))
            if text is None:
                text = ""
            item = ReportItem(
                report_no=idx + 1,
                report_id=row.get("id", idx + 1),
                app=app_name,
                text=text,
                row_data={k: normalize_scalar(v) for k, v in row.to_dict().items()},
            )
            reports.append(item)
        return reports

    def _iterative_clustering(self, app_name: str, reports: Sequence[ReportItem]) -> Tuple[Dict[int, List[int]], List[Dict[str, Any]]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        rng = random.Random(self.seed)
        remaining = [r.report_no for r in reports]
        rng.shuffle(remaining)
        report_lookup = {r.report_no: r for r in reports}
        current_clusters: Dict[int, List[int]] = {}
        iteration_logs: List[Dict[str, Any]] = []
        total = int(len(remaining))
        done = 0
        it = 0

        while remaining:
            it += 1
            batch_size = len(remaining) if len(remaining) < 1.1 * self.max_per_iteration else self.max_per_iteration
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]
            pct_next = 100.0 * float(done + int(len(batch))) / float(max(total, 1))
            self._log(f"[llmcluster] app={app_name} iter={it} queued={done + int(len(batch))}/{total} ({pct_next:.1f}%) batch={len(batch)} clusters={len(current_clusters)}")
            prompt = build_iteration_prompt(report_lookup, batch, current_clusters)
            try:
                response = self.backend.complete(messages + [{"role": "user", "content": prompt}])
            except Exception as exc:
                msg = str(exc).lower()
                looks_oversize = any(k in msg for k in ["context", "token", "too long", "length", "max_tokens", "exceed"])
                if looks_oversize and len(batch) > 1:
                    new_size = max(1, int(len(batch) // 2))
                    self._log(f"[llmcluster] app={app_name} iter={it} oversize_error -> shrink_batch {len(batch)}->{new_size}")
                    remaining = batch[new_size:] + remaining
                    remaining = batch[:new_size] + remaining
                    continue
                raise
            parsed = parse_clusters_from_text(response)
            if not parsed:
                repair_prompt = (
                    "Please reformat your previous answer. Output only the final cluster line in this exact format:\n"
                    "CURRENT CLUSTERS: Cluster 1 {1, 4, 5}, Cluster 2 {2, 3, 6}"
                )
                self._log(f"[llmcluster] app={app_name} iter={it} parse_failed -> repair")
                repaired = self.backend.complete(
                    messages
                    + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}, {"role": "user", "content": repair_prompt}]
                )
                parsed = parse_clusters_from_text(repaired)
                response = response + "\n\n[REFORMATTED]\n" + repaired
            if not parsed:
                raise RuntimeError("Failed to parse clusters from model response.")
            current_clusters = canonicalize_clusters(parsed)
            done += int(len(batch))
            pct = 100.0 * float(done) / float(max(total, 1))
            self._log(f"[llmcluster] app={app_name} iter={it} ok clusters={len(current_clusters)}")
            messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ])
            messages = compact_messages(messages, current_clusters, self.context_soft_limit)
            iteration_logs.append(
                {
                    "batch_report_nos": batch,
                    "batch_report_ids": [report_lookup[n].report_id for n in batch],
                    "n_clusters_after_iteration": len(current_clusters),
                    "response_preview": response[:2000],
                }
            )
        return current_clusters, iteration_logs

    def _generate_summaries(self, reports: Sequence[ReportItem], clusters: Dict[int, List[int]]) -> Dict[int, str]:
        report_lookup = {r.report_no: r for r in reports}
        summaries: Dict[int, str] = {}
        for cid, members in clusters.items():
            sample_members = members[: self.summary_sample_size]
            lines = [f"Cluster {cid} member reports:"]
            for report_no in sample_members:
                lines.append(f"Report {report_no}: {report_lookup[report_no].text}")
            if len(members) > len(sample_members):
                lines.append(f"(Only the first {len(sample_members)} of {len(members)} reports are shown for brevity.)")
            prompt = "\n".join(lines)
            response = self.backend.complete([
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": prompt},
            ])
            summaries[cid] = normalize_summary(response)
        return summaries


def choose_text_column(df: pd.DataFrame, requested: str) -> str:
    requested = requested or "auto"
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested text column not found: {requested}")
        return requested
    for candidate in ["primary_clue", "description", "text", "content"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not infer a text column.")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _simple_key(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text[:24]


def _extract_report_texts_from_prompt(prompt: str) -> Dict[int, str]:
    pairs: Dict[int, str] = {}
    for match in re.finditer(r"Report\s+(\d+)\s*:\s*(.+)", prompt):
        pairs[int(match.group(1))] = match.group(2).strip()
    return pairs


def build_iteration_prompt(report_lookup: Dict[int, ReportItem], batch: Sequence[int], current_clusters: Dict[int, List[int]]) -> str:
    if current_clusters:
        cluster_text = format_clusters_line(current_clusters)
    else:
        cluster_text = "CURRENT CLUSTERS: (none yet)"
    lines = [
        "Existing clusters from previous iterations:",
        cluster_text,
        "",
        "New reports to process:",
    ]
    for report_no in batch:
        lines.append(f"Report {report_no}: {report_lookup[report_no].text}")
    lines.append("")
    lines.append("Please analyze the new reports, merge them into existing clusters when suitable, create new clusters when needed, and end with the exact CURRENT CLUSTERS line.")
    return "\n".join(lines)


def parse_clusters_from_text(text: str) -> Dict[int, List[int]]:
    if not text:
        return {}
    normalized = text.replace("；", ";").replace("，", ",")
    normalized = normalized.replace("（", "(").replace("）", ")")
    m = re.search(r"CURRENT\s*CLUSTERS\s*:\s*(.*)$", normalized, flags=re.IGNORECASE | re.DOTALL)
    cluster_section = m.group(1).strip() if m else normalized
    clusters: Dict[int, List[int]] = {}
    pattern = re.compile(r"Cluster\s*(\d+)\s*\{([^}]*)\}", flags=re.IGNORECASE)
    for match in pattern.finditer(cluster_section):
        cid = int(match.group(1))
        nums = [int(x) for x in re.findall(r"\d+", match.group(2))]
        clusters[cid] = nums
    return canonicalize_clusters(clusters)


def canonicalize_clusters(clusters: Dict[int, Iterable[int]]) -> Dict[int, List[int]]:
    canon: Dict[int, List[int]] = {}
    for cid in sorted(clusters):
        members = sorted({int(x) for x in clusters[cid] if int(x) > 0})
        if members:
            canon[int(cid)] = members
    return canon


def format_clusters_line(clusters: Dict[int, Iterable[int]]) -> str:
    parts: List[str] = []
    for new_id, cid in enumerate(sorted(clusters), start=1):
        members = sorted({int(x) for x in clusters[cid] if int(x) > 0})
        if not members:
            continue
        member_text = ", ".join(str(x) for x in members)
        parts.append(f"Cluster {new_id} {{{member_text}}}")
    return "CURRENT CLUSTERS: " + ", ".join(parts) if parts else "CURRENT CLUSTERS: "


def compact_messages(messages: List[Dict[str, str]], current_clusters: Dict[int, List[int]], soft_limit: int) -> List[Dict[str, str]]:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars <= soft_limit:
        return messages
    system_message = messages[0]
    compacted = [
        system_message,
        {
            "role": "assistant",
            "content": (
                "Compressed iterative state to control context length. The latest authoritative clustering state is:\n"
                + format_clusters_line(current_clusters)
            ),
        },
    ]
    return compacted


def apply_corrections(clusters: Dict[int, List[int]], all_reports: Sequence[int]) -> Dict[int, List[int]]:
    completed = completeness_correction(clusters, all_reports)
    validated = validity_correction(completed, all_reports)
    return renumber_clusters(validated)


def completeness_correction(clusters: Dict[int, List[int]], all_reports: Sequence[int]) -> Dict[int, List[int]]:
    present = {r for members in clusters.values() for r in members}
    missing = [r for r in all_reports if r not in present]
    fixed = {cid: list(members) for cid, members in clusters.items()}
    if missing:
        next_cid = max(fixed.keys(), default=0) + 1
        fixed[next_cid] = missing
    return fixed


def validity_correction(clusters: Dict[int, List[int]], all_reports: Sequence[int]) -> Dict[int, List[int]]:
    fixed: Dict[int, List[int]] = {cid: [] for cid in clusters}
    membership: Dict[int, List[int]] = {r: [] for r in all_reports}
    for cid, members in clusters.items():
        for report_no in members:
            if report_no in membership:
                membership[report_no].append(cid)
    for report_no in all_reports:
        owners = membership.get(report_no, [])
        if len(owners) == 1:
            fixed[owners[0]].append(report_no)
        elif len(owners) > 1:
            largest_owner = max(owners, key=lambda c: len(clusters.get(c, [])))
            fixed[largest_owner].append(report_no)
    return fixed


def renumber_clusters(clusters: Dict[int, List[int]]) -> Dict[int, List[int]]:
    renumbered: Dict[int, List[int]] = {}
    next_id = 1
    for cid in sorted(clusters):
        members = sorted({int(x) for x in clusters[cid] if int(x) > 0})
        if members:
            renumbered[next_id] = members
            next_id += 1
    return renumbered


def normalize_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^[\-\*\d\.\s]+", "", text)
    return text[:300]


def build_cluster_results(
    reports: Sequence[ReportItem],
    clusters: Dict[int, List[int]],
    summaries: Dict[int, str],
) -> List[ClusterResult]:
    report_lookup = {r.report_no: r for r in reports}
    results: List[ClusterResult] = []
    for cid, members in clusters.items():
        results.append(
            ClusterResult(
                cluster_id=cid,
                members=members,
                member_report_ids=[report_lookup[m].report_id for m in members],
                summary=summaries.get(cid, ""),
            )
        )
    return results


def build_assignment_frame(reports: Sequence[ReportItem], cluster_results: Sequence[ClusterResult]) -> pd.DataFrame:
    cluster_by_report: Dict[int, ClusterResult] = {}
    for cluster in cluster_results:
        for report_no in cluster.members:
            cluster_by_report[report_no] = cluster

    rows: List[Dict[str, Any]] = []
    for report in reports:
        cluster = cluster_by_report.get(report.report_no)
        row = dict(report.row_data)
        row.update(
            {
                "report_no": report.report_no,
                "pred_cluster": cluster.cluster_id if cluster else None,
                "pred_cluster_size": len(cluster.members) if cluster else 0,
                "cluster_summary": cluster.summary if cluster else "",
                "cluster_member_report_ids": ", ".join(str(x) for x in cluster.member_report_ids) if cluster else "",
                "cluster_member_report_nos": ", ".join(str(x) for x in cluster.members) if cluster else "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _sanitize_boundary_token(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "llmcluster"
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:64].strip("_") or "llmcluster"


def build_llmcluster_clusters_df(corpus: pd.DataFrame, cfg: "LlmClusterMethodConfig") -> tuple[pd.DataFrame, str]:
    model = str(cfg.llm_model or "").strip() or "glm-4.7"
    boundary_name = f"llmcluster_{_sanitize_boundary_token(model)}"
    api_key = str(cfg.llm_api_key or "").strip()
    if not api_key:
        api_key = str(os.environ.get(str(cfg.llm_api_key_env or "ZHIPUAI_API_KEY"), "")).strip()

    if bool(cfg.llm_mock):
        backend: ChatBackend = MockChatBackend({})
    else:
        if not api_key:
            raise ValueError("Missing LLM API key. Pass llm_api_key or set environment variable.")
        backend = ZhipuChatBackend(
            api_key=api_key,
            model=model,
            api_url=str(cfg.llm_api_url or API_URL),
            timeout=int(cfg.llm_timeout),
            temperature=float(cfg.llm_temperature),
            max_tokens=int(cfg.llm_max_tokens),
            retries=int(cfg.llm_retries),
            thinking_enabled=bool(cfg.llm_thinking_enabled),
            verify_ssl=bool(cfg.llm_verify_ssl),
            use_env_proxy=bool(cfg.llm_use_env_proxy),
            verbose=bool(cfg.llm_verbose),
        )

    rows: list[dict] = []
    groups = list(corpus.groupby("app", sort=True))
    for i_app, (app, g) in enumerate(groups, start=1):
        g = g.reset_index(drop=True)
        runner_backend = backend
        if bool(cfg.llm_mock):
            report_lookup: Dict[int, ReportItem] = {}
            text_col = choose_text_column(g, str(cfg.llm_text_column or "auto"))
            for idx, row in g.reset_index(drop=True).iterrows():
                report_no = idx + 1
                report_lookup[report_no] = ReportItem(
                    report_no=report_no,
                    report_id=row.get("id", report_no),
                    app=str(app),
                    text=safe_str(row.get(text_col)) or safe_str(row.get("description")),
                    row_data={k: normalize_scalar(v) for k, v in row.to_dict().items()},
                )
            runner_backend = MockChatBackend(report_lookup)

        runner = LLMClusterRunner(
            backend=runner_backend,
            max_per_iteration=int(cfg.llm_max_per_iteration),
            seed=int(cfg.llm_seed),
            text_column=str(cfg.llm_text_column or "auto"),
            summary_sample_size=int(cfg.llm_summary_sample_size),
            context_soft_limit=int(cfg.llm_context_soft_limit),
            verbose=bool(cfg.llm_verbose),
        )
        if bool(cfg.llm_verbose):
            print(f"[llmcluster] app_index={i_app}/{len(groups)} app={app}", file=sys.stderr, flush=True)
        result = runner.run_app(str(app), g)
        assigns = result["assignments"]
        if assigns is None or len(assigns) == 0:
            continue
        for r in assigns.itertuples(index=False):
            rid = getattr(r, "id", "")
            issue_key = getattr(r, "issue_key", "")
            pred_cluster = getattr(r, "pred_cluster", None)
            if pred_cluster is None or (isinstance(pred_cluster, float) and pd.isna(pred_cluster)):
                pred_cluster = 0
            rows.append(
                {
                    "app": str(app),
                    "id": str(rid),
                    "cluster_id": f"{str(app)}__C{int(pred_cluster)}",
                    "issue_key": str(issue_key),
                }
            )

    clusters_df = pd.DataFrame(rows)
    if len(clusters_df) == 0:
        raise ValueError("LLMCluster produced empty clusters_df.")
    clusters_df = clusters_df.sort_values(["app", "cluster_id", "id"]).reset_index(drop=True)
    return clusters_df, boundary_name


@dataclass(frozen=True)
class LlmClusterMethodConfig:
    clusters_csv: str = ''
    repr_kind: str = 'tfidf_char'
    tfidf_ngram_min: int = 2
    tfidf_ngram_max: int = 4
    sbert_model: str = ''
    sbert_device: str = 'cpu'
    sbert_batch_size: int = 64
    llm_model: str = ''
    llm_api_url: str = API_URL
    llm_api_key: str = ''
    llm_api_key_env: str = 'ZHIPUAI_API_KEY'
    llm_timeout: int = DEFAULT_TIMEOUT
    llm_temperature: float = 0.0
    llm_max_tokens: int = DEFAULT_MAX_TOKENS
    llm_retries: int = 5
    llm_thinking_enabled: bool = False
    llm_verify_ssl: bool = True
    llm_use_env_proxy: bool = False
    llm_verbose: bool = True
    llm_text_column: str = 'auto'
    llm_max_per_iteration: int = DEFAULT_MAX_PER_ITERATION
    llm_seed: int = 42
    llm_summary_sample_size: int = 8
    llm_context_soft_limit: int = DEFAULT_CONTEXT_SOFT_LIMIT
    llm_mock: bool = False
    multi_Ls: tuple[int, ...] = (2, 3, 5)
    max_rank_cap: int = 200


def run_llmcluster_method(
    corpus: pd.DataFrame,
    q_keys: list[tuple[str, str]],
    cfg: LlmClusterMethodConfig,
) -> BoundaryArtifacts:
    boundary_name = ''
    clusters_df: pd.DataFrame
    if str(cfg.clusters_csv or '').strip():
        p = Path(str(cfg.clusters_csv or '')).resolve()
        if not p.exists():
            raise FileNotFoundError(f'LLMCluster clusters csv not found: {p}')
        clusters_df = read_clusters_csv(p)
        boundary_name = p.stem.replace('clusters_', '')
    else:
        clusters_df, boundary_name = build_llmcluster_clusters_df(corpus, cfg)
    if str(cfg.repr_kind) == 'sbert':
        rep_cfg = RepresentationConfig(
            kind='sbert',
            sbert_model=str(cfg.sbert_model),
            sbert_device=str(cfg.sbert_device),
            sbert_batch_size=int(cfg.sbert_batch_size),
        )
    else:
        rep_cfg = RepresentationConfig(
            kind='tfidf_char',
            tfidf_analyzer='char',
            tfidf_ngram_min=int(cfg.tfidf_ngram_min),
            tfidf_ngram_max=int(cfg.tfidf_ngram_max),
        )
    rep_store = RepresentationStore(corpus, rep_cfg)
    rep_store.attach_clusters(clusters_df)
    rankings = rank_text_baselines(
        df_reports=corpus,
        clusters_df=clusters_df,
        rep_store=rep_store,
        q_keys=q_keys,
        multi_Ls=list(cfg.multi_Ls),
        max_rank_cap=int(cfg.max_rank_cap),
    )
    return BoundaryArtifacts(
        boundary_name=str(boundary_name),
        clusters_df=clusters_df,
        rep_store=rep_store,
        rankings=rankings,
        method_family='llmcluster',
    )
