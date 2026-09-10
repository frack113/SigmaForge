#!/usr/bin/env python3
"""RAG tester via FastAPI test client — tests the full API layer.

Usage:
    uv run python AskRag/test_rag_ask.py

Requires: Qdrant running on localhost:6333, DuckDB initialised.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import httpx

# Ensure project root is on the path BEFORE any imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Minimal infrastructure bootstrap so SearchEngine / retrievers can work.
# The test does NOT need a running uvicorn server — only Qdrant + DuckDB.
# ---------------------------------------------------------------------------


def _bootstrap_infra() -> None:
    """Initialise DatabaseService (singleton) with an in-memory DuckDB."""
    from src.config.settings import TEMP_DIR

    # Ensure data dirs exist
    for d in (TEMP_DIR / "duckdb",):
        d.mkdir(parents=True, exist_ok=True)

    db_path = str((TEMP_DIR / "duckdb" / "test_rag.duckdb").resolve())
    os.environ["SIGMA_DUCKDB_PATH"] = db_path  # not used by Config but safe

    from src.infrastructure.database import DatabaseService

    if DatabaseService._instance is None:
        db = DatabaseService(db_path)
        db.initialize()


_bootstrap_infra()

STOP_WORDS = frozenset(
    {
        "what",
        "are",
        "the",
        "in",
        "for",
        "is",
        "of",
        "and",
        "to",
        "how",
        "does",
        "a",
        "an",
        "at",
        "on",
        "with",
        "as",
        "not",
        "be",
        "or",
        "from",
        "by",
        "it",
        "its",
        "that",
        "this",
        "which",
        "can",
        "when",
        "if",
        "where",
        "will",
        "use",
        "used",
        "vs",
        "between",
        "than",
        "but",
        "must",
        "should",
        "would",
        "shall",
        "may",
        "might",
        "need",
        "do",
        "did",
        "has",
        "have",
        "had",
        "being",
        "been",
        "were",
        "was",
        "into",
        "about",
        "up",
        "out",
        "all",
        "any",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "just",
        "also",
        "then",
        "now",
        "no",
        "yes",
        "q",
        "a",
    }
)


def _extract_keywords(text: str) -> set[str]:
    words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", text.lower()))
    return words - STOP_WORDS


def _extract_bigrams(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", text.lower())
    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i + 1]}")
    return bigrams


def _evaluate_result(
    top_text: str, score: float, question: str, expected_answer: str
) -> tuple[str, float]:
    if not top_text:
        return "FAIL", 0.0
    kw_question = _extract_keywords(question)
    kw_answer = _extract_keywords(expected_answer)
    combined_keywords = kw_question | kw_answer
    if not combined_keywords:
        return "FAIL", 0.0
    matched = sum(1 for kw in combined_keywords if kw in top_text.lower())
    coverage = matched / len(combined_keywords)
    bigrams = _extract_bigrams(question) | _extract_bigrams(expected_answer)
    if bigrams:
        bigram_matches = sum(1 for bg in bigrams if bg in top_text.lower())
        bigram_coverage = bigram_matches / len(bigrams)
        coverage = max(coverage, bigram_coverage)
    weighted = coverage * score
    if weighted >= 0.25 and score >= 0.3:
        return "PASS", weighted
    elif weighted >= 0.15 and score >= 0.2:
        return "PARTIAL", weighted
    else:
        return "FAIL", weighted



class _SearchRequest(BaseModel):
    query: str
    limit: int = 3


_engine: Any | None = None


async def _get_engine():
    global _engine
    if _engine is None:
        from src.core.search.engine import SearchEngine

        _engine = SearchEngine(collection_names=["sigma_spec"], top_k=3)
    return _engine


def _build_test_app():
    """Build a minimal FastAPI app with only the search endpoint."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="SigmaForge — test")

    @app.post("/search")
    async def search(request: _SearchRequest) -> dict:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Empty query")
        try:
            engine = await _get_engine()
            results = await engine.search(request.query, top_k=request.limit)
            return {"data": results, "meta": {"total": len(results)}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG search against ask_spec.md questions")
    parser.add_argument("--url", default="http://localhost:8000", help="RAG server URL")
    parser.add_argument("--max", type=int, default=0, help="Max questions to test (0 = all)")
    parser.add_argument(
        "--question",
        type=int,
        default=0,
        help="Test a single question by 1-based index (e.g. --question 43)",
    )
    args = parser.parse_args()

    base_url = args.url

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        # Quick connectivity check via search endpoint (must return 400 for empty query or 200)
        try:
            resp = await client.post("/api/v1/search", json={"query": "ping", "limit": 1})
            if resp.status_code not in (200, 400):
                print(f"App not reachable at {base_url} (HTTP {resp.status_code})")
                return -1
        except httpx.ConnectError as e:
            print(f"Cannot connect to app at {base_url}: {e}")
            return -1

        qa_path = Path("AskRag/ask_spec.md")
        content = qa_path.read_text(encoding="utf-8")

        pattern = re.compile(r"\*\*Q:\*\*\s*(.+?)\s*\n\*\*A:\*\*\s*(.+?)(?=\n\*\*|\Z)", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            print("No questions found in AskRag/ask_spec.md")
            return 0

        questions: list[tuple[str, str]] = []
        for q, a in matches:
            a_clean = re.sub(r"\n{3,}", "\n", a).strip()
            a_clean = re.sub(r"\n\s*#{1,3}\s+\S.*", "", a_clean).strip()
            questions.append((q.strip(), a_clean))

        total = len(questions)
        print(f"Loaded {total} questions from {qa_path}\n")
        print("=" * 80)

        passed = 0
        failed = 0
        partial = 0

        # Determine which questions to test
        if args.question and 1 <= args.question <= total:
            test_indices = [args.question - 1]
            print(f"Testing single question #{args.question} of {total}\n")
        else:
            max_questions = args.max if args.max else len(questions)
            test_indices = list(range(min(total, max_questions)))

        for idx in test_indices:
            question, expected_answer = questions[idx]
            preview = question[:80].replace("\n", " ")
            if args.question:
                print(f"\nQ#{args.question}: {preview}")
            else:
                print(f"\n[{idx + 1}/{total}] Q: {preview}")

            try:
                resp = await client.post(
                    "/api/v1/search",
                    json={"query": question, "limit": 3},
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1
                continue

            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                failed += 1
                continue

            body = resp.json()
            results = body.get("data", [])

            if not results:
                print("  No results from RAG search")
                failed += 1
                continue

            best_status = "FAIL"
            best_weighted = 0.0
            best_score = 0.0

            for rank, result in enumerate(results):
                text = result.get("text", "")
                score = result.get("score", 0)
                print(f"  [{rank + 1}] Score: {score:.4f}")

                status, weighted = _evaluate_result(text, score, question, expected_answer)
                print(f"       Weighted coverage: {weighted:.2f} -> {status}")

                status_rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0}
                if status_rank.get(status, 0) > status_rank.get(best_status, 0) or (
                    status == best_status and weighted > best_weighted
                ):
                    best_status = status
                    best_weighted = weighted
                    best_score = score

            print(
                f"  -> Best: {best_status} (score={best_score:.4f}, coverage={best_weighted:.2f})"
            )

            if best_status == "PASS":
                passed += 1
            elif best_status == "PARTIAL":
                partial += 1
            else:
                failed += 1

        print("\n" + "=" * 80)
        if args.question:
            tested = 1
            print(
                f"RESULTS for question #{args.question}: {passed} passed, {partial} partial, {failed} failed"
            )
        else:
            tested = min(total, max_questions)
        pass_rate = (passed + partial) / tested * 100 if tested else 0
        print(f"         {pass_rate:.1f}% acceptable (pass + partial)")
        print("=" * 80)

        return passed


if __name__ == "__main__":
    import logging
    import sys as _sys

    # Suppress verbose model loading logs and tqdm progress bars
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # Redirect stderr to suppress tqdm/progress bar output
    import io as _io

    _sys.stderr = _io.StringIO()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    result = asyncio.run(main())

    # Restore stderr
    _sys.stderr = _sys.__stderr__

    if result is not None and result < 10:
        sys.exit(1)
