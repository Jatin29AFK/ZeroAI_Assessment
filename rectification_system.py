"""Surgical patch-based article rectification system.

The grading command is ``python rectifier.py rectify-all``. This module keeps
that path robust by generating find/replace factual patches, applying only
anchored edits to the original AI text, and falling back to the current article
text if a model call fails.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".rectification_cache"
LOG_DIR = ROOT / ".rectification_logs"
MAPPING_FILE = ROOT / "article_mapping.json"

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 180

FENCE_RE = re.compile(r"^\s*```(?:json|JSON|text|txt)?\s*|\s*```\s*$", re.MULTILINE)
JSON_ARRAY_START_RE = re.compile(r"^\s*\[", re.DOTALL)
JSON_OBJECT_START_RE = re.compile(r"^\s*\{", re.DOTALL)
SPACE_RE = re.compile(r"\s+")


class RectificationError(Exception):
    """Rectification error with retry metadata for LLM calls."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    source_file: Path
    ai_generated_file: Path
    rectified_file: Path


@dataclass(frozen=True)
class Patch:
    find: str
    replace: str
    reason: str = ""
    pass_name: str = "first"

    def cache_dict(self) -> dict[str, str]:
        return {"find": self.find, "replace": self.replace}


@dataclass(frozen=True)
class PatchApplication:
    patch: Patch
    applied: bool
    method: str
    message: str = ""


class LLMClient:
    """Minimal OpenAI-compatible chat/completions client using stdlib only."""

    def __init__(self) -> None:
        load_env_file(ROOT / ".env")
        self.api_base = os.getenv("LLM_API_BASE", "").strip()
        self.model = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))

        if not self.api_base:
            raise RectificationError("LLM_API_BASE is not set")
        if not self.api_key:
            raise RectificationError("LLM_API_KEY is not set")

        self.endpoints = self._chat_endpoints(self.api_base)

    @staticmethod
    def _chat_endpoints(api_base: str) -> list[str]:
        base = api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return [base]
        if base.endswith("/v1"):
            return [f"{base}/chat/completions"]
        return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]

    def complete(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                return self._complete_once(messages)
            except (OSError, TimeoutError, urllib.error.URLError, RectificationError) as exc:
                last_error = exc
                if isinstance(exc, RectificationError) and not exc.retryable:
                    raise
                if attempt == 4:
                    break
                time.sleep(min(30, 2**attempt))
        raise RectificationError(f"LLM request failed after retries: {last_error}") from last_error

    def _complete_once(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        last_error: RectificationError | None = None
        for endpoint in self.endpoints:
            try:
                return self._post_chat_completion(endpoint, payload)
            except RectificationError as exc:
                last_error = exc
                if "LLM HTTP 404" in str(exc) or "LLM HTTP 405" in str(exc):
                    continue
                raise
        raise last_error or RectificationError("No LLM endpoint candidates were available")

    def _post_chat_completion(self, endpoint: str, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code >= 500 or exc.code in {408, 409, 425, 429}
            raise RectificationError(
                f"LLM HTTP {exc.code}: {body[:1000]}",
                retryable=retryable,
            ) from exc

        try:
            data = json.loads(body)
            return data["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RectificationError(f"Unexpected LLM response: {body[:1000]}") from exc


class PatchBasedRectifier:
    def __init__(self) -> None:
        self.logger = setup_logging()
        self._client: LLMClient | None = None

    def rectify_record(self, record: ArticleRecord) -> str:
        article_logger = article_logger_for(record.article_id)
        if not has_llm_config() and record.rectified_file.exists():
            article_logger.warning("LLM configuration missing; preserving existing rectified output")
            return record.rectified_file.read_text(encoding="utf-8")

        source_text = read_text(record.source_file)
        ai_text = read_text(record.ai_generated_file)
        current_text = ai_text

        cache_key = cache_key_for(source_text, ai_text)

        first_patches = self._safe_patch_pass(
            record=record,
            pass_name="first",
            cache_key=cache_key,
            source_text=source_text,
            ai_text=ai_text,
            current_text=current_text,
            applied_patches=[],
            article_logger=article_logger,
        )
        current_text, first_apps = apply_patches(current_text, first_patches)
        log_applications(article_logger, first_apps)

        second_patches = self._safe_patch_pass(
            record=record,
            pass_name="second",
            cache_key=cache_key,
            source_text=source_text,
            ai_text=ai_text,
            current_text=current_text,
            applied_patches=[app.patch for app in first_apps if app.applied],
            article_logger=article_logger,
        )
        current_text, second_apps = apply_patches(current_text, second_patches)
        log_applications(article_logger, second_apps)

        final_text = clean_final_output(current_text)
        try:
            validate_final_output(final_text, ai_text)
        except RectificationError as exc:
            article_logger.warning("Output validation failed; falling back to patch-based current text: %s", exc)
            final_text = clean_final_output(current_text or ai_text)
            if not final_text.strip() or looks_like_json(final_text):
                final_text = ai_text.strip()

        all_apps = first_apps + second_apps
        article_logger.info(
            "Finished with %s applied patches and %s skipped patches",
            sum(1 for app in all_apps if app.applied),
            sum(1 for app in all_apps if not app.applied),
        )
        return final_text.strip() + "\n"

    def _safe_patch_pass(
        self,
        *,
        record: ArticleRecord,
        pass_name: str,
        cache_key: str,
        source_text: str,
        ai_text: str,
        current_text: str,
        applied_patches: list[Patch],
        article_logger: logging.Logger,
    ) -> list[Patch]:
        try:
            return self._patch_pass(
                record=record,
                pass_name=pass_name,
                cache_key=cache_key,
                source_text=source_text,
                ai_text=ai_text,
                current_text=current_text,
                applied_patches=applied_patches,
                article_logger=article_logger,
            )
        except Exception as exc:  # noqa: BLE001 - batch output must survive LLM/parser failures.
            article_logger.warning("Skipping %s-pass patches after failure: %s", pass_name, exc)
            return []

    def _patch_pass(
        self,
        *,
        record: ArticleRecord,
        pass_name: str,
        cache_key: str,
        source_text: str,
        ai_text: str,
        current_text: str,
        applied_patches: list[Patch],
        article_logger: logging.Logger,
    ) -> list[Patch]:
        cache_path = CACHE_DIR / f"{record.article_id}.{cache_key}.{pass_name}.json"
        cached = load_cached_patches(cache_path, pass_name)
        if cached is not None:
            article_logger.info("Using cached %s-pass patches from %s", pass_name, cache_path)
            return cached

        messages = build_patch_messages(
            pass_name=pass_name,
            source_text=source_text,
            ai_text=ai_text,
            current_text=current_text,
            applied_patches=applied_patches,
        )
        article_logger.info("Requesting %s-pass patches", pass_name)
        raw_response = self.client.complete(messages)
        write_raw_response(record.article_id, pass_name, raw_response)

        patches = validate_patch_shapes(parse_patches(raw_response, pass_name))
        save_cached_patches(cache_path, patches)
        article_logger.info("Received %s %s-pass patches", len(patches), pass_name)
        return patches

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client


_RECTIFIER: PatchBasedRectifier | None = None


def run(
    ai_generated_content: str,
    *,
    source_content: str | None = None,
    article_id: str = "article",
) -> str:
    """Compatibility function used by the starter CLI.

    Prefer ``rectify_article_record`` when the source path is available. If the
    caller provides only AI text, return it unchanged rather than rewriting.
    """
    if source_content is None:
        return clean_final_output(ai_generated_content) + "\n"

    temp_record = ArticleRecord(
        article_id=article_id,
        source_file=Path("<memory-source>"),
        ai_generated_file=Path("<memory-ai>"),
        rectified_file=Path("<memory-output>"),
    )
    rectifier = get_rectifier()
    cache_key = cache_key_for(source_content, ai_generated_content)
    logger = article_logger_for(article_id)
    first = rectifier._safe_patch_pass(
        record=temp_record,
        pass_name="first",
        cache_key=cache_key,
        source_text=source_content,
        ai_text=ai_generated_content,
        current_text=ai_generated_content,
        applied_patches=[],
        article_logger=logger,
    )
    current, first_apps = apply_patches(ai_generated_content, first)
    second = rectifier._safe_patch_pass(
        record=temp_record,
        pass_name="second",
        cache_key=cache_key,
        source_text=source_content,
        ai_text=ai_generated_content,
        current_text=current,
        applied_patches=[app.patch for app in first_apps if app.applied],
        article_logger=logger,
    )
    current, _second_apps = apply_patches(current, second)
    return clean_final_output(current) + "\n"


def rectify_article_record(record: ArticleRecord) -> str:
    return get_rectifier().rectify_record(record)


def get_rectifier() -> PatchBasedRectifier:
    global _RECTIFIER
    if _RECTIFIER is None:
        _RECTIFIER = PatchBasedRectifier()
    return _RECTIFIER


def load_article_records() -> list[ArticleRecord]:
    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    records: list[ArticleRecord] = []
    for item in mapping:
        article_id = item["article_id"]
        records.append(
            ArticleRecord(
                article_id=article_id,
                source_file=ROOT / item["source_file"],
                ai_generated_file=ROOT / item["ai_generated_file"],
                rectified_file=ROOT / item["rectified_file"],
            )
        )
    return records


def build_patch_messages(
    *,
    pass_name: str,
    source_text: str,
    ai_text: str,
    current_text: str,
    applied_patches: list[Patch],
) -> list[dict[str, str]]:
    system = (
        "You are a surgical factual editor. The source article is authoritative. "
        "Return strict JSON only: an array of objects with find, replace, and reason. "
        "Never return markdown, commentary, or a rewritten article."
    )
    rules = """
Identify only factual inconsistencies relative to the source article.
Do not rewrite the article.
Do not improve style.
Do not paraphrase correct text.
Do not change synonyms.
Do not change headings unless factually wrong.
Preserve wording, tone, sentence structure, paragraph structure, formatting, and order.
Prefer the shortest possible replacement span.
If only a number, name, date, or place is wrong, replace only that entity.
If a phrase is factually wrong, replace only that phrase.
Never add information unless the AI article made a wrong factual claim requiring correction.
Every find value must be an exact substring from the current article text.
Return [] if no factual correction is needed.
""".strip()

    if pass_name == "first":
        user = f"""
{rules}

SOURCE ARTICLE:
<<<SOURCE
{source_text}
SOURCE

CURRENT AI ARTICLE TO PATCH:
<<<ARTICLE
{ai_text}
ARTICLE

Return strict JSON only.
""".strip()
    else:
        applied_json = json.dumps([patch.cache_dict() for patch in applied_patches], ensure_ascii=False)
        user = f"""
{rules}

This is the second and final validation pass. Return only additional missing corrections.
Do not repeat already applied patches.

SOURCE ARTICLE:
<<<SOURCE
{source_text}
SOURCE

ORIGINAL AI ARTICLE:
<<<ORIGINAL
{ai_text}
ORIGINAL

CURRENT RECTIFIED ARTICLE TO PATCH:
<<<CURRENT
{current_text}
CURRENT

APPLIED PATCHES:
{applied_json}

Return strict JSON only.
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_patches(raw_response: str, pass_name: str) -> list[Patch]:
    cleaned = strip_markdown_fences(raw_response).strip()
    data = try_json(cleaned)
    if data is None:
        array_text = extract_first_balanced_json(cleaned, "[", "]")
        if array_text is not None:
            data = try_json(array_text)
    if data is None:
        object_text = extract_first_balanced_json(cleaned, "{", "}")
        if object_text is not None:
            data = try_json(object_text)

    if isinstance(data, dict):
        for key in ("patches", "corrections", "edits", "replacements"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if data is None:
        raise RectificationError(f"Could not parse JSON patches from {pass_name}-pass response")
    if not isinstance(data, list):
        raise RectificationError(f"{pass_name}-pass response was not a JSON array")

    patches: list[Patch] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        find = item.get("find")
        replace = item.get("replace")
        reason = item.get("reason", "")
        if isinstance(find, str) and isinstance(replace, str):
            patches.append(Patch(find=find, replace=replace, reason=str(reason), pass_name=pass_name))
    return patches


def validate_patch_shapes(patches: list[Patch]) -> list[Patch]:
    valid: list[Patch] = []
    seen: set[tuple[str, str]] = set()
    for patch in patches:
        find = patch.find
        replace = patch.replace
        if not find or find == replace:
            continue
        key = (find, replace)
        if key in seen:
            continue
        seen.add(key)
        valid.append(patch)
    return valid


def apply_patches(text: str, patches: list[Patch]) -> tuple[str, list[PatchApplication]]:
    current = text
    applications: list[PatchApplication] = []
    for patch in patches:
        if patch.find in current:
            current = current.replace(patch.find, patch.replace, 1)
            applications.append(PatchApplication(patch, True, "exact"))
            continue

        span = find_whitespace_normalized_span(current, patch.find)
        if span is not None:
            start, end = span
            current = current[:start] + patch.replace + current[end:]
            applications.append(PatchApplication(patch, True, "whitespace-normalized"))
            continue

        applications.append(
            PatchApplication(
                patch=patch,
                applied=False,
                method="not-found",
                message="find string was not present in current article",
            )
        )
    return current, applications


def find_whitespace_normalized_span(text: str, needle: str) -> tuple[int, int] | None:
    normalized_needle = normalize_space(needle)
    if not normalized_needle:
        return None

    normalized_chars: list[str] = []
    index_map: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                normalized_chars.append(" ")
                index_map.append(index)
                previous_space = True
            continue
        normalized_chars.append(char)
        index_map.append(index)
        previous_space = False

    normalized_text = "".join(normalized_chars).strip()
    offset = len("".join(normalized_chars)) - len("".join(normalized_chars).lstrip())
    match_start = normalized_text.find(normalized_needle)
    if match_start == -1:
        return None
    mapped_start = match_start + offset
    mapped_end = mapped_start + len(normalized_needle) - 1
    if mapped_start >= len(index_map) or mapped_end >= len(index_map):
        return None
    return index_map[mapped_start], index_map[mapped_end] + 1


def clean_final_output(text: str) -> str:
    return strip_markdown_fences(text).strip()


def validate_final_output(final_text: str, ai_text: str) -> None:
    if not final_text.strip():
        raise RectificationError("final output is empty")
    if final_text.lstrip().startswith("```"):
        raise RectificationError("final output starts with a markdown fence")
    if looks_like_json(final_text):
        raise RectificationError("final output appears to be JSON")

    ai_len = max(len(ai_text.strip()), 1)
    ratio = len(final_text.strip()) / ai_len
    if ratio < 0.50 or ratio > 1.60:
        raise RectificationError(f"final output length ratio is suspicious: {ratio:.2f}")

    similarity = difflib.SequenceMatcher(None, ai_text, final_text).ratio()
    if similarity < 0.60:
        raise RectificationError(f"final output is too dissimilar to AI article: {similarity:.2f}")


def looks_like_json(text: str) -> bool:
    return bool(JSON_ARRAY_START_RE.match(text) or JSON_OBJECT_START_RE.match(text))


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|JSON|text|txt)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()
    return FENCE_RE.sub("", text).strip()


def extract_first_balanced_json(text: str, open_char: str, close_char: str) -> str | None:
    for start, char in enumerate(text):
        if char != open_char:
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == open_char:
                depth += 1
            elif current == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def load_cached_patches(cache_path: Path, pass_name: str) -> list[Patch] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None

    patches: list[Patch] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("find"), str) and isinstance(item.get("replace"), str):
            patches.append(Patch(find=item["find"], replace=item["replace"], pass_name=pass_name))
    return validate_patch_shapes(patches)


def save_cached_patches(cache_path: Path, patches: list[Patch]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [patch.cache_dict() for patch in patches]
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_raw_response(article_id: str, pass_name: str, response: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{article_id}.{pass_name}.raw.txt").write_text(response, encoding="utf-8")


def log_applications(logger: logging.Logger, applications: list[PatchApplication]) -> None:
    for app in applications:
        if app.applied:
            logger.info("Applied patch via %s: %r -> %r", app.method, app.patch.find, app.patch.replace)
        else:
            logger.warning("Skipped patch: %r -> %r (%s)", app.patch.find, app.patch.replace, app.message)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rectification")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "rectification.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console_handler)
    return logger


def article_logger_for(article_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"rectification.{article_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / f"{article_id}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def write_batch_summary(results: list[dict[str, Any]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "succeeded": sum(1 for item in results if item.get("success")),
        "failed": sum(1 for item in results if not item.get("success")),
        "results": results,
    }
    (LOG_DIR / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def has_llm_config() -> bool:
    load_env_file(ROOT / ".env")
    return bool(os.getenv("LLM_API_BASE", "").strip() and os.getenv("LLM_API_KEY", "").strip())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def cache_key_for(source_text: str, ai_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(ai_text.encode("utf-8"))
    return digest.hexdigest()[:16]
