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
DEFAULT_INPUT_LIMIT = 2000
DEFAULT_MAX_TOKENS = 160
DEFAULT_NUM_CTX = 4096

SYSTEM_PROMPT = (
    "あなたは優秀なPMです。GitHub Issue/PRのやり取りを読み、"
    "顧客向けの週次進捗報告を作成します。専門用語は必要に応じて噛み砕き、"
    "現在の状況、決定事項、完了・保留・議論中などの状態を汲み取ってください。"
    "PRについては過剰な技術詳細を避け、解決する課題や顧客に関係する進捗を優先してください。"
)

USER_PROMPT_TEMPLATE = """以下のIssue/PRのやり取りを読み、顧客向けの週次報告を日本語1文で簡潔にまとめてください。
/no_think

制約:
- 出力は要約文のみ
- 1文で書く
- 80文字から120文字程度に収める
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
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = " ".join(text.strip().split())
    text = text.strip("\"'`「」")
    return text


def response_content(response: Any) -> str:
    message = response.choices[0].message
    return message.content or ""


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


def create_openai_client(timeout: float) -> OpenAI:
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def summarize_record(
    client: OpenAI,
    record: dict[str, Any],
    model: str,
    max_chars: int,
    max_tokens: int,
    num_ctx: int,
) -> str:
    issue_context = build_issue_context(record, max_chars)
    user_prompt = USER_PROMPT_TEMPLATE.format(issue_context=issue_context)
    for attempt, attempt_max_tokens in enumerate((max_tokens, max(max_tokens * 3, 320)), start=1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=attempt_max_tokens,
            extra_body={
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": attempt_max_tokens,
                    "temperature": 0.2,
                },
                "keep_alive": "10m",
            },
        )
        raw_content = response_content(response)
        summary = clean_one_sentence(raw_content)
        if summary:
            return summary
        if attempt == 1:
            print("  Empty summary returned; retrying once with a larger output limit.")

    raise RuntimeError(
        f"LLM returned an empty summary for {number_label(record.get('number'))}. "
        "Try increasing --max-tokens or using a non-thinking model/tag."
    )


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


def load_existing_rows(output_csv: Path) -> list[dict[str, Any]]:
    if not output_csv.exists():
        return []
    return pd.read_csv(output_csv, dtype=str).fillna("").to_dict("records")


def save_rows(records: list[dict[str, Any]], rows_by_number: dict[str, dict[str, Any]], output_csv: Path) -> None:
    rows = []
    for record in records:
        number = number_label(record.get("number"))
        if number in rows_by_number:
            rows.append(rows_by_number[number])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


def summarize_issues(
    input_dir: Path,
    output_csv: Path,
    model: str,
    max_chars: int,
    max_tokens: int,
    num_ctx: int,
    limit: int | None,
    timeout: float,
    resume: bool,
) -> None:
    records = read_issue_json_files(input_dir)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    client = create_openai_client(timeout)
    existing_rows = load_existing_rows(output_csv) if resume else []
    rows_by_number = {row.get("番号"): row for row in existing_rows if row.get("番号")}
    done_numbers = {number for number, row in rows_by_number.items() if row.get("LLMによる進捗要約")}
    for index, record in enumerate(records, start=1):
        number = number_label(record.get("number"))
        if resume and number in done_numbers:
            print(f"[{index}/{len(records)}] Skipping {number} (already summarized)")
            continue

        title = record.get("title") or ""
        print(f"[{index}/{len(records)}] Summarizing {number} {title}")
        summary = summarize_record(client, record, model, max_chars, max_tokens, num_ctx)
        rows_by_number[number] = build_row(record, summary)
        save_rows(records, rows_by_number, output_csv)

    save_rows(records, rows_by_number, output_csv)
    print(f"Saved {len(rows_by_number)} rows to {output_csv}")


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
        "--max-tokens",
        type=int,
        default=int(os.getenv("OPENAI_MAX_TOKENS") or DEFAULT_MAX_TOKENS),
        help=f"Maximum output tokens per issue/PR. Default: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(os.getenv("OLLAMA_NUM_CTX") or DEFAULT_NUM_CTX),
        help=f"Ollama context window option. Default: {DEFAULT_NUM_CTX}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only summarize the first N JSON files. Useful for connection tests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OPENAI_TIMEOUT") or 600),
        help="OpenAI client timeout seconds. Default: 600",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing rows in the output CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize_issues(
        Path(args.input_dir),
        Path(args.output_csv),
        args.model,
        args.max_chars,
        args.max_tokens,
        args.num_ctx,
        args.limit,
        args.timeout,
        not args.no_resume,
    )


if __name__ == "__main__":
    main()
