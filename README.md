# AI Article Rectification Challenge

Completed assessment submission for the **AI Article Rectification Challenge**.

This repository implements an automated AI editor that compares each AI-generated article with its authoritative source article and writes a rectified version with factual errors corrected while preserving the original wording, structure, formatting, tone, and article order as much as possible.

## Evaluator Command

The submission is designed around the required grading command:

```bash
python rectifier.py rectify-all
```

When the command completes, `rectified_articles/` should contain all **104** article outputs using the expected filenames, for example `article_001.txt`.

Output files contain only final article text:

- no JSON artifacts
- no markdown fences
- no explanations
- no placeholder content
- no notebook or manual step dependency

## Quick Start

1. Create a `.env` file from `env.example`:

   ```env
   LLM_API_BASE="https://recllm.brahmastra.tech/"
   LLM_MODEL_NAME=groq/openai/gpt-oss-120b
   LLM_API_KEY=your-api-key
   ```

2. Run a small test:

   ```bash
   python rectifier.py test --count 3
   ```

3. Run the full rectification pipeline:

   ```bash
   python rectifier.py rectify-all
   ```

4. Validate generated outputs:

   ```bash
   python validate_outputs.py --expected-count 104
   find rectified_articles -name "article_*.txt" | wc -l
   python -m compileall .
   ```

Expected file count:

```text
104
```

## Repository Structure

```text
.
├── ai_generated_articles/      # AI-written articles containing factual issues
├── source_articles/            # Authoritative source articles
├── rectified_articles/         # Final generated outputs
├── article_mapping.json        # Source, AI, and output path mapping
├── rectifier.py                # CLI entry point used by the evaluator
├── rectification_system.py     # Patch-based rectification engine
├── validate_outputs.py         # Local output integrity checker
├── budget_checker.py           # Optional API budget helper
├── env.example                 # Required environment variable template
└── requirements.txt            # Runtime dependency list
```

## Approach

The solution uses a **surgical patch-based rectification strategy** instead of asking the LLM to rewrite the full article.

For each article, the system:

1. Loads the source article and AI-generated article from `article_mapping.json`.
2. Sends both articles to the configured OpenAI-compatible LLM endpoint.
3. Instructs the model to identify factual inconsistencies only.
4. Requires strict JSON patches in this shape:

   ```json
   [
     {
       "find": "exact wrong substring from AI article",
       "replace": "corrected substring based only on source article",
       "reason": "brief internal reason"
     }
   ]
   ```

5. Applies only anchored `find`/`replace` edits to the original AI text.
6. Runs one additional validation pass to catch missed factual corrections.
7. Writes the final article text to `rectified_articles/article_XXX.txt`.

The `reason` field is treated as internal model metadata and is not written to output files.

## Preservation Rules

The prompt and patch application logic are designed to preserve the AI article as much as possible:

- do not rewrite the article
- do not improve style
- do not paraphrase correct text
- do not change headings unless factually wrong
- do not add unnecessary details
- prefer the smallest factual correction span
- replace only the wrong number, name, date, place, or phrase when possible
- treat the source article as ground truth

## Robustness Features

The pipeline is built to complete the full batch even when individual LLM calls fail.

- Temperature is fixed at `0`.
- LLM calls retry with exponential backoff.
- The batch continues if one article or one pass fails.
- Responses are cached per article in `.rectification_cache/`.
- Logs are written to `.rectification_logs/`.
- Markdown fences are stripped from accidental model output.
- Messy responses are scanned for the first valid balanced JSON array.
- Every patch is validated before application.
- Exact substring matching is attempted first.
- Conservative whitespace-normalized matching is used as a fallback.
- Unapplied patches are logged and skipped.
- Final outputs are checked for empty content, JSON artifacts, markdown fences, and suspicious full rewrites.

If all LLM calls fail for an article, the system writes a safe fallback instead of leaving the output missing. This protects the evaluator contract that all 104 mapped articles produce files.

## Environment Variables

The code reads all model configuration from `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `LLM_API_BASE` | Yes | OpenAI-compatible API base URL |
| `LLM_MODEL_NAME` | Yes | Model name, e.g. `groq/openai/gpt-oss-120b` |
| `LLM_API_KEY` | Yes | API key for the provided endpoint |
| `LLM_MAX_TOKENS` | No | Optional response token cap, defaults to `4096` |

No real `.env` file or API key should be committed.

## Dependencies

The rectification pipeline uses the Python standard library only. `requirements.txt` is intentionally minimal, which avoids heavyweight or system-level installation requirements.

The project can still be set up with:

```bash
pip install -r requirements.txt
```

## Validation Strategy

Recommended pre-submission checks:

```bash
python rectifier.py test --count 3
python rectifier.py test --count 10
python rectifier.py rectify-all
python validate_outputs.py --expected-count 104
find rectified_articles -name "article_*.txt" | wc -l
python -m compileall .
```

`validate_outputs.py` checks that:

- all mapped article IDs have output files
- the output count matches the expected count
- no output file is empty
- no output file starts with markdown fences, `[`, or `{`

## Notes On API Reliability

The provided LLM proxy can occasionally return transient `502 Bad Gateway` errors. The implementation retries these failures and continues processing. If many `502` responses occur during a full run, rerun with a lower worker count:

```bash
python rectifier.py rectify-all --workers 1
```

Cached successful patch responses are reused on later runs, reducing repeated token usage.

## Submission Checklist

Before sharing the repository:

- `python rectifier.py rectify-all` completes
- `rectified_articles/` contains 104 `article_*.txt` files
- `.env` is not committed
- `.rectification_cache/` and `.rectification_logs/` are not committed
- `python validate_outputs.py --expected-count 104` passes
- `python -m compileall .` passes
