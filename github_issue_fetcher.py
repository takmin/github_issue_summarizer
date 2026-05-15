from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import GithubException


DEFAULT_DAYS = 7


@dataclass
class CommentRecord:
    id: int | str | None
    type: str
    author: str | None
    body: str | None
    created_at: str | None
    updated_at: str | None
    url: str | None


@dataclass
class TimelineRecord:
    id: int | str | None
    event: str | None
    actor: str | None
    created_at: str | None
    commit_id: str | None
    url: str | None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    normalized = as_utc(value)
    if normalized is None:
        return None
    return normalized.isoformat()


def in_period(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = as_utc(value)
    return normalized is not None and start_at <= normalized <= end_at


def login_of(user: Any) -> str | None:
    return getattr(user, "login", None) if user else None


def safe_filename(kind: str, number: int, title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
    slug = slug[:80] if slug else "untitled"
    return f"{number:06d}_{kind.lower()}_{slug}.json"


def issue_touched_in_period(issue: Any, start_at: datetime, end_at: datetime) -> bool:
    dates = [
        getattr(issue, "created_at", None),
        getattr(issue, "updated_at", None),
        getattr(issue, "closed_at", None),
    ]
    return any(in_period(value, start_at, end_at) for value in dates)


def collect_period_comments(issue: Any, start_at: datetime, end_at: datetime) -> list[CommentRecord]:
    records: list[CommentRecord] = []
    for comment in issue.get_comments():
        created_at = getattr(comment, "created_at", None)
        updated_at = getattr(comment, "updated_at", None)
        if not (in_period(created_at, start_at, end_at) or in_period(updated_at, start_at, end_at)):
            continue
        records.append(
            CommentRecord(
                id=getattr(comment, "id", None),
                type="comment",
                author=login_of(getattr(comment, "user", None)),
                body=getattr(comment, "body", None),
                created_at=to_iso(created_at),
                updated_at=to_iso(updated_at),
                url=getattr(comment, "html_url", None),
            )
        )
    return records


def collect_period_timeline(issue: Any, start_at: datetime, end_at: datetime) -> list[TimelineRecord]:
    if not hasattr(issue, "get_timeline"):
        return []

    records: list[TimelineRecord] = []
    try:
        timeline_items: Iterable[Any] = issue.get_timeline()
        for item in timeline_items:
            created_at = getattr(item, "created_at", None)
            if not in_period(created_at, start_at, end_at):
                continue
            records.append(
                TimelineRecord(
                    id=getattr(item, "id", None),
                    event=getattr(item, "event", None),
                    actor=login_of(getattr(item, "actor", None)),
                    created_at=to_iso(created_at),
                    commit_id=getattr(item, "commit_id", None),
                    url=getattr(item, "url", None),
                )
            )
    except GithubException as exc:
        print(f"Warning: timeline fetch failed for #{issue.number}: {exc.data}")
    return records


def serialize_issue(issue: Any, start_at: datetime, end_at: datetime) -> dict[str, Any]:
    kind = "PR" if getattr(issue, "pull_request", None) else "Issue"
    comments = collect_period_comments(issue, start_at, end_at)
    timeline = collect_period_timeline(issue, start_at, end_at)

    return {
        "kind": kind,
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "labels": [label.name for label in issue.labels],
        "assignees": [assignee.login for assignee in issue.assignees],
        "state": issue.state,
        "url": issue.html_url,
        "created_at": to_iso(issue.created_at),
        "updated_at": to_iso(issue.updated_at),
        "closed_at": to_iso(issue.closed_at),
        "period": {
            "start_at": to_iso(start_at),
            "end_at": to_iso(end_at),
        },
        "comments": [asdict(comment) for comment in comments],
        "timeline": [asdict(item) for item in timeline],
    }


def fetch_updated_issues(repo_full_name: str, start_at: datetime, end_at: datetime) -> list[dict[str, Any]]:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set. Create a .env file or set the environment variable.")

    github = Github(auth=Auth.Token(token))
    try:
        repo = github.get_repo(repo_full_name)

        records: list[dict[str, Any]] = []
        for issue in repo.get_issues(state="all", since=start_at):
            if issue_touched_in_period(issue, start_at, end_at):
                records.append(serialize_issue(issue, start_at, end_at))

        records.sort(key=lambda item: (item["kind"], item["number"]))
        return records
    finally:
        github.close()


def save_issue_json_files(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = output_dir / safe_filename(record["kind"], record["number"], record["title"])
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub Issues/PRs updated in a period and save one JSON file per item."
    )
    parser.add_argument("repo", help="Repository full name, e.g. owner/repo")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output/issues",
        help="Directory to write JSON files. Default: output/issues",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days to look back from now. Default: {DEFAULT_DAYS}",
    )
    parser.add_argument(
        "--start-at",
        help="Period start in ISO 8601 format. Overrides --days when specified.",
    )
    parser.add_argument(
        "--end-at",
        help="Period end in ISO 8601 format. Default: current UTC time.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end_at = as_utc(parse_datetime(args.end_at) or utc_now())
    start_at = as_utc(parse_datetime(args.start_at) or (end_at - timedelta(days=args.days)))
    if start_at is None or end_at is None:
        raise ValueError("Could not determine the fetch period.")
    if start_at > end_at:
        raise ValueError("--start-at must be earlier than or equal to --end-at.")

    records = fetch_updated_issues(args.repo, start_at, end_at)
    save_issue_json_files(records, Path(args.output_dir))
    print(f"Saved {len(records)} issue/PR JSON files to {args.output_dir}")


if __name__ == "__main__":
    main()
