from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

APP_SHEETS_BLACKLIST = {"QA_Summary", "Label_Guide"}
_SPLIT_RE = re.compile(r"[\|,;\s]+")
_NON_WORD_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def safe_str(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_report_id(x: object) -> str:
    s = safe_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+?\d+", s):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    if re.fullmatch(r"\d+\.\d+", s):
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return s


def split_report_ids(s: object) -> List[str]:
    raw = safe_str(s)
    if not raw:
        return []
    parts = [p for p in _SPLIT_RE.split(raw) if p]
    out: List[str] = []
    for p in parts:
        rid = normalize_report_id(p)
        if rid:
            out.append(rid)
    return out


def normalize_text_key(s: object) -> str:
    text = safe_str(s).lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def text_len_info(s: str) -> Tuple[int, int]:
    raw = safe_str(s)
    key = normalize_text_key(raw)
    return len(raw), len(key.replace(" ", ""))


def pick_text_col(df: pd.DataFrame) -> str:

    if "primary_clue" in df.columns:
        return "primary_clue"
    if "description" in df.columns:
        return "description"
    for c in ["text", "content", "desc"]:
        if c in df.columns:
            return c
    raise ValueError("No text column found (expected primary_clue/description/text).")


def encode_unicode_path_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append(f"#U{ord(ch):x}")
    return "".join(out)


def resolve_app_dir(input_root: Path, app: str) -> Path:
    app = str(app)
    direct = input_root / app
    if direct.exists():
        return direct
    enc = input_root / encode_unicode_path_name(app)
    if enc.exists():
        return enc

    for p in input_root.iterdir():
        if p.is_dir() and (p.name == app or p.name == encode_unicode_path_name(app)):
            return p
    raise FileNotFoundError(f"Cannot resolve app dir for app={app!r} under {input_root}")


def build_anchor_inventory(input_root: Optional[str], apps: List[str]) -> Dict[Tuple[str, str], int]:

    if not input_root:
        return {}
    root = Path(input_root)
    inv: Dict[Tuple[str, str], int] = {}
    for app in sorted(set(map(str, apps))):
        try:
            app_dir = resolve_app_dir(root, app)
        except Exception:
            continue
        for rel in ["unify/nodes.xlsx", "unify/edges.xlsx"]:
            p = app_dir / rel
            if not p.exists():
                continue
            try:
                df = pd.read_excel(p)
            except Exception:
                continue
            if "source_row_index" not in df.columns:
                continue
            for ids in df["source_row_index"].fillna(""):
                for rid in split_report_ids(ids):
                    inv[(app, rid)] = 1
    return inv


def load_raw_excel(xlsx_path: str) -> pd.DataFrame:
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    rows = []
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        rows.append(_coerce_sheet(df, default_app=p.stem, sheet_name=p.stem))
    else:
        xls = pd.ExcelFile(p)
        for sh in xls.sheet_names:
            if sh in APP_SHEETS_BLACKLIST:
                continue
            df = pd.read_excel(p, sheet_name=sh)
            rows.append(_coerce_sheet(df, default_app=sh, sheet_name=sh))
    out = pd.concat(rows, ignore_index=True)
    return out


def _coerce_sheet(df: pd.DataFrame, default_app: str, sheet_name: str) -> pd.DataFrame:
    df = df.copy()
    text_col = pick_text_col(df)
    if "app" not in df.columns:
        df["app"] = default_app
    else:
        app_series = df["app"].fillna("").astype(str).str.strip()
        df.loc[app_series.str.len() == 0, "app"] = default_app
    if "id" not in df.columns:
        raise ValueError(f"Sheet {sheet_name} missing required column: id")
    if "issue_key" not in df.columns:
        raise ValueError(f"Sheet {sheet_name} missing required column: issue_key")

    df["__sheet__"] = sheet_name
    df["id_norm"] = df["id"].map(normalize_report_id)
    df["issue_key_norm"] = df["issue_key"].fillna("").astype(str).str.strip()
    df["text_used"] = df[text_col].map(safe_str)
    df["text_col_used"] = text_col
    return df


def add_report_features(df: pd.DataFrame, anchor_inv: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    df = df.copy()
    raw_lens, key_lens, text_keys = [], [], []
    for t in df["text_used"].fillna(""):
        raw_len, key_len = text_len_info(t)
        raw_lens.append(raw_len)
        key_lens.append(key_len)
        text_keys.append(normalize_text_key(t))
    df["raw_text_len"] = raw_lens
    df["norm_text_len"] = key_lens
    df["text_key"] = text_keys
    df["has_anchor"] = [int(anchor_inv.get((str(a), str(i)), 0)) for a, i in zip(df["app"], df["id_norm"])]
    return df


def build_issue_audit(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["app", "issue_key_norm"], dropna=False)
    rows = []
    for (app, issue_key), g in grp:
        texts = g["text_key"].fillna("")
        uniq_text_ratio = texts.nunique() / max(len(g), 1)
        other_token_count = sum(1 for part in str(issue_key).split(".") if part.lower() == "other")
        rows.append({
            "app": app,
            "issue_key": issue_key,
            "issue_size": int(len(g)),
            "n_unique_text_key": int(texts.nunique()),
            "unique_text_ratio": float(uniq_text_ratio),
            "other_token_count": int(other_token_count),
            "has_any_anchor_rate": float(g["has_anchor"].mean()) if len(g) else 0.0,
            "median_raw_text_len": float(g["raw_text_len"].median()) if len(g) else 0.0,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["issue_size", "other_token_count", "unique_text_ratio"], ascending=[False, False, True])


def apply_filter_protocol(
    df: pd.DataFrame,
    *,
    min_text_len: int,
    require_anchor: bool,
    dedup_exact: bool,
    drop_broad_issue: bool,
    broad_issue_size: int,
    broad_other_tokens: int,
    max_issue_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    work = df.copy()
    work["drop_reason"] = ""


    mask_bad_id = work["id_norm"].astype(str).str.len() == 0
    work.loc[mask_bad_id & (work["drop_reason"] == ""), "drop_reason"] = "empty_id"

    mask_bad_issue = work["issue_key_norm"].astype(str).str.len() == 0
    work.loc[mask_bad_issue & (work["drop_reason"] == ""), "drop_reason"] = "empty_issue_key"

    mask_bad_text = work["text_used"].astype(str).str.len() == 0
    work.loc[mask_bad_text & (work["drop_reason"] == ""), "drop_reason"] = "empty_text"


    mask_short = work["norm_text_len"] < int(min_text_len)
    work.loc[mask_short & (work["drop_reason"] == ""), "drop_reason"] = f"short_text_lt_{min_text_len}"


    if require_anchor:
        mask_no_anchor = work["has_anchor"].fillna(0).astype(int) <= 0
        work.loc[mask_no_anchor & (work["drop_reason"] == ""), "drop_reason"] = "no_graph_anchor"


    issue_audit = build_issue_audit(work[work["drop_reason"] == ""].copy())
    issue_audit["broad_issue_flag"] = 0
    if drop_broad_issue:
        broad_mask = (
            (issue_audit["issue_size"] > int(max_issue_size))
            | ((issue_audit["issue_size"] >= int(broad_issue_size)) & (issue_audit["other_token_count"] >= int(broad_other_tokens)))
        )
        issue_audit.loc[broad_mask, "broad_issue_flag"] = 1
        broad_set = set(zip(issue_audit.loc[broad_mask, "app"], issue_audit.loc[broad_mask, "issue_key"]))
        broad_row_mask = work.apply(lambda r: (r["app"], r["issue_key_norm"]) in broad_set, axis=1)
        work.loc[broad_row_mask & (work["drop_reason"] == ""), "drop_reason"] = "broad_or_noisy_issue_group"


    if dedup_exact:
        key_cols = ["app", "issue_key_norm", "text_key"]
        dup_mask = work[work["drop_reason"] == ""].duplicated(subset=key_cols, keep="first")
        work.loc[dup_mask[dup_mask].index, "drop_reason"] = "exact_text_dup_in_issue"

    kept = work[work["drop_reason"] == ""].copy().reset_index(drop=True)
    dropped = work[work["drop_reason"] != ""].copy().reset_index(drop=True)


    kept_issue_sizes = kept.groupby(["app", "issue_key_norm"]).size().rename("kept_issue_size")
    issue_audit = issue_audit.merge(kept_issue_sizes.reset_index(), how="left", left_on=["app", "issue_key"], right_on=["app", "issue_key_norm"])
    if "issue_key_norm" in issue_audit.columns:
        issue_audit = issue_audit.drop(columns=["issue_key_norm"])
    issue_audit["kept_issue_size"] = issue_audit["kept_issue_size"].fillna(0).astype(int)
    return kept, dropped, issue_audit


def write_filtered_excel(original_df: pd.DataFrame, kept_df: pd.DataFrame, out_xlsx: Path) -> None:
    keep_ids = set(zip(kept_df["__sheet__"], kept_df["id_norm"].astype(str)))
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sh, g in original_df.groupby("__sheet__", sort=False):
            g = g.copy()
            mask = [((sh, str(rid)) in keep_ids) for rid in g["id_norm"]]
            out = g.loc[mask].copy()
            drop_cols = [
                "__sheet__", "id_norm", "issue_key_norm", "text_used", "text_col_used",
                "raw_text_len", "norm_text_len", "text_key", "has_anchor", "drop_reason"
            ]
            drop_cols = [c for c in drop_cols if c in out.columns]
            out = out.drop(columns=drop_cols)
            out.to_excel(writer, sheet_name=str(sh)[:31], index=False)


def summarize(original_df: pd.DataFrame, kept_df: pd.DataFrame, dropped_df: pd.DataFrame, issue_audit: pd.DataFrame) -> Dict[str, object]:
    def ratio(a, b):
        return float(a) / float(b) if b else 0.0
    out = {
        "n_rows_raw": int(len(original_df)),
        "n_rows_kept": int(len(kept_df)),
        "keep_rate": ratio(len(kept_df), len(original_df)),
        "n_rows_dropped": int(len(dropped_df)),
        "n_apps": int(original_df["app"].astype(str).nunique()),
        "n_issue_groups_raw": int(original_df.groupby(["app", "issue_key_norm"]).ngroups),
        "n_issue_groups_kept": int(kept_df.groupby(["app", "issue_key_norm"]).ngroups),
        "dropped_by_reason": dropped_df["drop_reason"].value_counts().to_dict(),
        "raw_issue_size": {
            "median": float(original_df.groupby(["app", "issue_key_norm"]).size().median()),
            "p90": float(original_df.groupby(["app", "issue_key_norm"]).size().quantile(0.9)),
            "max": int(original_df.groupby(["app", "issue_key_norm"]).size().max()),
        },
        "kept_issue_size": {
            "median": float(kept_df.groupby(["app", "issue_key_norm"]).size().median()) if len(kept_df) else 0.0,
            "p90": float(kept_df.groupby(["app", "issue_key_norm"]).size().quantile(0.9)) if len(kept_df) else 0.0,
            "max": int(kept_df.groupby(["app", "issue_key_norm"]).size().max()) if len(kept_df) else 0,
        },
    }
    return out


def add_presets(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--preset", default="ecfa_friendly", choices=["conservative", "ecfa_friendly"])
    ap.add_argument("--min_text_len", type=int, default=None, help="Override preset")
    ap.add_argument("--require_anchor", type=int, default=None, help="1/0; override preset")
    ap.add_argument("--dedup_exact", type=int, default=None, help="1/0; override preset")
    ap.add_argument("--drop_broad_issue", type=int, default=None, help="1/0; override preset")
    ap.add_argument("--broad_issue_size", type=int, default=None, help="Issue size threshold paired with other-token rule")
    ap.add_argument("--broad_other_tokens", type=int, default=None, help="Min count of 'other' tokens in issue_key")
    ap.add_argument("--max_issue_size", type=int, default=None, help="Hard max issue size; overly broad groups dropped")


def resolve_cfg(args: argparse.Namespace) -> Dict[str, object]:

    base = {
        "min_text_len": 16,
        "require_anchor": True,
        "dedup_exact": True,
        "drop_broad_issue": False,
        "broad_issue_size": 12,
        "broad_other_tokens": 2,
        "max_issue_size": 999999,
    }

    if args.preset == "ecfa_friendly":
        base.update({
            "min_text_len": 16,
            "require_anchor": True,
            "dedup_exact": True,
            "drop_broad_issue": True,
            "broad_issue_size": 12,
            "broad_other_tokens": 2,
            "max_issue_size": 25,
        })
    for k in ["min_text_len", "broad_issue_size", "broad_other_tokens", "max_issue_size"]:
        v = getattr(args, k)
        if v is not None:
            base[k] = v
    for k in ["require_anchor", "dedup_exact", "drop_broad_issue"]:
        v = getattr(args, k)
        if v is not None:
            base[k] = bool(int(v))
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter raw RQ3 annotated data into a cleaner ECFA-friendly corpus.")
    ap.add_argument("--xlsx", required=True, help="Original annotated Excel (single-sheet or multi-sheet).")
    ap.add_argument("--input_root", default="input/ROOT_glm4.7", help="ROOT_glm4.7 for graph-anchor coverage. Can be empty to skip anchor filtering.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    add_presets(ap)
    args = ap.parse_args()

    cfg = resolve_cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_excel(args.xlsx)
    anchor_inv = build_anchor_inventory(args.input_root, apps=raw["app"].astype(str).tolist()) if args.input_root else {}
    feat = add_report_features(raw, anchor_inv)
    kept, dropped, issue_audit = apply_filter_protocol(feat, **cfg)


    kept_csv = out_dir / "filtered_reports.csv"
    dropped_csv = out_dir / "dropped_reports.csv"
    issue_csv = out_dir / "issue_audit.csv"
    report_flags_csv = out_dir / "report_flags.csv"
    kept_xlsx = out_dir / "filtered_reports.xlsx"
    summary_json = out_dir / "filter_summary.json"

    kept.to_csv(kept_csv, index=False, encoding="utf-8-sig")
    dropped.to_csv(dropped_csv, index=False, encoding="utf-8-sig")
    issue_audit.to_csv(issue_csv, index=False, encoding="utf-8-sig")
    feat[["__sheet__", "app", "id", "id_norm", "issue_key", "issue_key_norm", "text_used", "raw_text_len", "norm_text_len", "has_anchor"]].to_csv(report_flags_csv, index=False, encoding="utf-8-sig")
    write_filtered_excel(feat, kept, kept_xlsx)
    summary = summarize(feat, kept, dropped, issue_audit)
    summary["config"] = cfg
    summary["xlsx"] = str(Path(args.xlsx).resolve())
    summary["input_root"] = str(args.input_root)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] wrote:")
    for p in [kept_xlsx, kept_csv, dropped_csv, issue_csv, report_flags_csv, summary_json]:
        print(f"  - {p}")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
