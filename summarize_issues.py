from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_INPUT_LIMIT = 4000

SYSTEM_PROMPT = (
    "あなたは優秀なPMです。GitHub Issue/PRのやり取りを読み、"
    "顧客向けの週次進捗報告を作成します。専門用語は必要に応じて噛み砕き、"
    "現在の状況、決定事項、完了・保留・議論中などの状態を汲み取ってください。"
    "PRについては過剰な技術詳細を避け、解決する課題や顧客に関係する進捗を優先してください。"
)

USER_PROMPT_TEMPLATE = """以下のIssue/PRのやり取りを読み、顧客向けの週次報告を日本語1文で簡潔にまとめてください。

制約:
- 出力は要約文のみ
- 1文で書く
- 技術的な経緯は必要な範囲に絞る
- コメントやタイムラインから現在の状況や決定事項を反映する

{issue_context}
"""


def read_issue_json_files(input_dir: Path) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        record["_source_file"] = str(path)
        records.append(record)
    return records


def join_values(values: list[Any] | None) -> str:
    if not values:
        return ""
    return ", ".join(str(value) for value in values if value is not None)


def number_label(number: Any) -> str:
    return f"#{number}" if number is not None else ""


def clean_one_sentence(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = " ".join(text.strip().split())
    text = text.strip("\"'`「」")
    return text


def trim_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "（前半は長いため省略。以下は直近の内容です）\n" + text[-max_chars:]


def format_comments(comments: list[dict[str, Any]]) -> str:
    if not comments:
        return "なし"

    lines: list[str] = []
    for comment in comments:
        lines.append(
            "\n".join(
                [
                    f"- author: {comment.get('author') or ''}",
                    f"  created_at: {comment.get('created_at') or ''}",
                    f"  body: {comment.get('body') or ''}",
                ]
            )
        )
    return "\n".join(lines)


def format_timeline(timeline: list[dict[str, Any]]) -> str:
    if not timeline:
        return "なし"

    lines: list[str] = []
    for item in timeline:
        event = item.get("event") or ""
        actor = item.get("actor") or ""
        created_at = item.get("created_at") or ""
        commit_id = item.get("commit_id") or ""
        commit_text = f", commit_id: {commit_id}" if commit_id else ""
        lines.append(f"- {created_at}: {event} by {actor}{commit_text}")
    return "\n".join(lines)


def build_issue_context(record: dict[str, Any], max_chars: int) -> str:
    period = record.get("period") or {}
    context = "\n".join(
        [
            "Issue/PR基本情報:",
            f"- 種別: {record.get('kind') or ''}",
            f"- 番号: {number_label(record.get('number'))}",
            f"- タイトル: {record.get('title') or ''}",
            f"- ステータス: {record.get('state') or ''}",
            f"- 担当者: {join_values(record.get('assignees'))}",
            f"- ラベル: {join_values(record.get('labels'))}",
            f"- URL: {record.get('url') or ''}",
            f"- 作成日時: {record.get('created_at') or ''}",
            f"- 更新日時: {record.get('updated_at') or ''}",
            f"- クローズ日時: {record.get('closed_at') or ''}",
            f"- 取得期間: {period.get('start_at') or ''} から {period.get('end_at') or ''}",
            "",
            "本文:",
            record.get("body") or "なし",
            "",
            "期間内コメント:",
            format_comments(record.get("comments") or []),
            "",
            "期間内タイムライン:",
            format_timeline(record.get("timeline") or []),
        ]
    )
    return trim_context(context, max_chars)


def create_openai_client() -> OpenAI:
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"
    return OpenAI(base_url=base_url, api_key=api_key)


def summarize_record(client: OpenAI, record: dict[str, Any], model: str, max_chars: int) -> str:
    issue_context = build_issue_context(record, max_chars)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(issue_context=issue_context)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return clean_one_sentence(content)


def build_row(record: dict[str, Any], summary: str) -> dict[str, str]:
    return {
        "種別": record.get("kind") or "",
        "番号": number_label(record.get("number")),
        "タイトル": record.get("title") or "",
        "ステータス": record.get("state") or "",
        "担当者": join_values(record.get("assignees")),
        "ラベル": join_values(record.get("labels")),
        "LLMによる進捗要約": summary,
        "URL": record.get("url") or "",
    }


def summarize_issues(input_dir: Path, output_csv: Path, model: str, max_chars: int, limit: int | None) -> None:
    records = read_issue_json_files(input_dir)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    client = create_openai_client()
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        number = number_label(record.get("number"))
        title = record.get("title") or ""
        print(f"[{index}/{len(records)}] Summarizing {number} {title}")
        summary = summarize_record(client, record, model, max_chars)
        rows.append(build_row(record, summary))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} rows to {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Summarize GitHub Issue/PR JSON files with an OpenAI-compatible Ollama endpoint and save CSV."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="output/issues",
        help="Directory containing JSON files created by github_issue_fetcher.py. Default: output/issues",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default="output/weekly_report.csv",
        help="CSV output path. Default: output/weekly_report.csv",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL,
        help=f"Model name served by Ollama. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_INPUT_LIMIT,
        help=f"Maximum characters of context per issue/PR. Default: {DEFAULT_INPUT_LIMIT}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only summarize the first N JSON files. Useful for connection tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize_issues(Path(args.input_dir), Path(args.output_csv), args.model, args.max_chars, args.limit)


if __name__ == "__main__":
    main()
