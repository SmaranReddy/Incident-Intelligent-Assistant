"""
Incident Knowledge Extraction Pipeline
=======================================

Converts raw incident reports + Slack threads into structured knowledge records
that power the RAG system's retrieval and reasoning capabilities.

This pipeline runs at INGESTION TIME — it builds a structured knowledge base
once, not at query time.

Usage
-----
    python extract_incidents.py                        # process all incidents
    python extract_incidents.py --id INC-1000          # process single incident
    python extract_incidents.py --dry-run              # preview prompts, skip LLM
    python extract_incidents.py --overwrite            # reprocess existing outputs
    python extract_incidents.py --model llama-3.1-8b-instant

Output
------
    data/structured_incidents/INC-xxxx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# ── Environment & config ─────────────────────────────────────────────────────

load_dotenv()

INCIDENTS_DIR  = Path("data/incidents")
SLACK_DIR      = Path("data/slack")
OUTPUT_DIR     = Path("data/structured_incidents")

# Default to a capable model; override with --model or EXTRACTION_MODEL env var
DEFAULT_MODEL     = os.getenv("EXTRACTION_MODEL", "llama-3.3-70b-versatile")
MAX_SLACK_MSGS    = 60     # truncate very long threads to avoid token overflow
RETRY_ATTEMPTS    = 3
RETRY_DELAY_SEC   = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output schema defaults ───────────────────────────────────────────────────

EMPTY_RECORD: dict = {
    "incident_id":         None,
    "service":             None,
    "symptoms":            [],
    "hypotheses_tested":   [],
    "failed_attempts":     [],
    "confirmed_root_cause": None,
    "resolution_action":   None,
    "services_affected":   [],
    "confidence":          0.0,
}

# ── Prompt constants ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior Site Reliability Engineer conducting post-incident analysis.

Your task: extract structured diagnostic signals from raw incident data.

STRICT EXTRACTION RULES
-----------------------
1. Only extract what is EXPLICITLY stated. Never infer, fabricate, or guess.
2. Prefer an EMPTY field over an INCORRECT extraction.
3. Distinguish signal types with precision:
   - symptoms          → observable effects: error rates, latency spikes, user-facing failures, alerts
   - hypotheses_tested → what engineers SUSPECTED and investigated (including wrong guesses)
   - failed_attempts   → specific mitigations tried that DID NOT fix the issue
   - confirmed_root_cause → the verified technical root cause (not a symptom, not a guess)
   - resolution_action → the exact change or action that FIXED the incident
4. A wrong hypothesis belongs in hypotheses_tested, NOT in confirmed_root_cause.
5. A mitigation that partially helped but didn't resolve the issue belongs in failed_attempts.
6. Output ONLY valid JSON. No markdown, no prose, no code fences.

CONFIDENCE SCORING (0.0 – 1.0)
--------------------------------
0.9–1.0  All fields populated; root cause explicitly confirmed in postmortem/resolution
0.7–0.8  Root cause and resolution clear; minor fields missing
0.5–0.6  Root cause likely but not explicitly confirmed; partial picture
0.3–0.4  Significant gaps; mostly symptoms without clear root cause
0.0–0.2  Insufficient signal to extract meaningful structured data\
"""

# One-shot example is embedded in the extraction prompt to anchor output format
EXTRACTION_PROMPT = """\
## INPUT DATA

### Incident Report
{incident_report}

### Slack Thread(s)
{slack_threads}

---

## TASK

Analyze the input and extract structured incident knowledge.

### FAILED ATTEMPTS EXTRACTION (CRITICAL)

Identify ALL debugging or mitigation actions that were attempted but did NOT resolve the issue.

**Definition:**
A failed attempt is any SYSTEM-CHANGING action taken by engineers during debugging that did
not fix the incident or was later abandoned.

A system-changing action MUST do at least one of the following:
- Modify service configuration or parameters
- Restart, redeploy, or scale a service
- Change code, queries, or data
- Apply a setting that alters runtime behavior

If the action does NOT modify system behavior, it is NOT a failed attempt.

You MUST extract failed attempts from:
- Explicit statements (e.g., "we restarted the service but it didn't help")
- Implicit signals (e.g., multiple steps tried before the final fix)
- Slack discussions (engineers suggesting and trying fixes)

Include:
- Restarting services
- Clearing cache
- Scaling instances
- Config changes that didn't work
- Query changes that didn't work
- Temporary mitigations that failed

DO NOT include:
- The final successful resolution
- Hypotheses that were never acted upon
- Monitoring or observation steps (e.g., "checking logs", "looking at dashboards")
- Checking external tools (e.g., "checked CloudFlare", "looked at Datadog", "reviewed Grafana")
- Observing or reading metrics, traces, or alerts
- Any action described with: "checking", "looking at", "reviewing", "investigating",
  "verifying", "confirming", "monitoring" — these are observations, not attempts

Remember: Hypothesis = thought. Observation = checking. Attempt = system-changing action.
Only "Attempt" belongs in failed_attempts.

**Extraction Rules:**
1. Extract ONLY actions that were actually executed, not just discussed.
2. The action MUST be system-changing — it must alter configuration, code, or runtime
   behavior. If it only gathers information, exclude it.
3. If an attempt does not explicitly say "failed", infer failure if:
   - A different solution was applied afterwards, OR
   - The incident persisted after the action was taken.
4. Keep each attempt concise (1 line each).
5. Preserve technical meaning — do NOT oversimplify.
6. Before including any item, ask: "Did this action change the system?"
   - YES → include it (if it didn't fix the incident)
   - NO  → exclude it

**Examples:**

Example 1 — explicit failure:
  Input:  "We restarted the payment service but the issue persisted"
  Output: ["restart payment service"]

Example 2 — multiple implicit failures:
  Input:  "Tried increasing DB connections and clearing cache — no improvement"
  Output: ["increase DB connections", "clear cache"]

Example 3 — hypothesis never acted upon (NOT a failed attempt):
  Input:  "Maybe it's Redis? Not sure"
  Output: []

Example 4 — observation disguised as action (NOT a failed attempt):
  Input:  "Checked CloudFlare — traffic looked normal. Reviewed DB slow query log."
  Output: []  ← neither action changed the system; both are observations

**Edge case:** If NO failed attempts are found, return an empty list [].
DO NOT hallucinate attempts that are not evidenced in the input.

**Output field:**
  "failed_attempts": ["...", "..."]

---

### Output Schema
{{
  "incident_id":          "<string>",
  "service":              "<string: primary failing service, or null>",
  "symptoms":             ["<observable effect 1>", "..."],
  "hypotheses_tested":    ["<thing engineers suspected/investigated>", "..."],
  "failed_attempts":      ["<mitigation tried that did NOT fix it>", "..."],
  "confirmed_root_cause": "<specific technical cause, or null>",
  "resolution_action":    "<exact fix that resolved the incident, or null>",
  "services_affected":    ["<service that experienced degradation>", "..."],
  "confidence":           <float 0.0–1.0>
}}

---

### Worked Example

INPUT (incident report excerpt):
  incident_id: INC-9999
  service: payment-service
  symptoms:
    - Duplicate charges appearing in transaction log
    - payment_transactions table showing duplicate rows
  root_cause: Race condition — concurrent requests both read status=PENDING before either committed
  resolution: Added SELECT FOR UPDATE to serialize concurrent reads; deployed to prod

INPUT (Slack excerpt):
  [2024-01-10 14:03] eng_a: Could be a network glitch causing client retries
  [2024-01-10 14:07] eng_b: CloudFlare looks clean, traffic is normal
  [2024-01-10 14:10] eng_b: Race condition confirmed in payment_processor.py
  [2024-01-10 14:12] eng_a: Tried adding unique index — caused IntegrityError cascade instead
  [2024-01-10 14:25] eng_b: SELECT FOR UPDATE fix deployed, duplicates stopped

OUTPUT:
{{
  "incident_id": "INC-9999",
  "service": "payment-service",
  "symptoms": [
    "Duplicate charges appearing in transaction log",
    "payment_transactions table showing duplicate rows"
  ],
  "hypotheses_tested": [
    "network glitch causing client retries",
    "race condition in payment_processor.py"
  ],
  "failed_attempts": [
    "added unique index — caused IntegrityError cascade, did not resolve"
  ],
  "confirmed_root_cause": "Race condition — concurrent requests both read status=PENDING before either committed",
  "resolution_action": "Added SELECT FOR UPDATE to serialize concurrent reads; deployed to prod",
  "services_affected": ["payment-service"],
  "confidence": 0.94
}}

---

Now extract from the INPUT DATA above. Output ONLY the JSON object.\
"""


# ── Step 1: Load data ────────────────────────────────────────────────────────

def load_incidents() -> dict[str, dict]:
    """Load all incident JSON files, keyed by incident_id."""
    incidents: dict[str, dict] = {}
    for path in sorted(INCIDENTS_DIR.glob("INC-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            iid = data.get("incident_id")
            if iid:
                incidents[iid] = data
        except Exception as exc:
            log.warning("Skipping %s — %s", path.name, exc)
    log.info("Loaded %d incident reports from %s", len(incidents), INCIDENTS_DIR)
    return incidents


def load_slack_threads() -> dict[str, list[dict]]:
    """
    Load all Slack threads, indexed by incident_id.
    One incident can have multiple threads (e.g. #incidents + #ops-war-room).
    """
    by_incident: dict[str, list[dict]] = {}
    for path in sorted(SLACK_DIR.glob("SLACK-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            iid = data.get("incident_id")
            if iid:
                by_incident.setdefault(iid, []).append(data)
        except Exception as exc:
            log.warning("Skipping %s — %s", path.name, exc)
    log.info("Loaded Slack threads for %d incidents from %s", len(by_incident), SLACK_DIR)
    return by_incident


def load_data() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Load all incidents and Slack threads. Returns (incidents, slack_by_incident)."""
    return load_incidents(), load_slack_threads()


# ── Step 2: Build prompt ─────────────────────────────────────────────────────

def _format_incident_report(incident: dict) -> str:
    """Render the incident report as a compact, readable block for the prompt."""
    lines = [
        f"incident_id : {incident.get('incident_id', 'unknown')}",
        f"title       : {incident.get('title', '')}",
        f"service     : {incident.get('service', '')}",
        f"severity    : {incident.get('severity', '')}",
        f"started_at  : {incident.get('timestamp', '')}",
        f"resolved_at : {incident.get('resolved_at', 'not resolved')}",
    ]
    if symptoms := incident.get("symptoms"):
        lines.append("symptoms    :")
        for s in symptoms:
            lines.append(f"  - {s}")
    if root_cause := incident.get("root_cause"):
        lines.append(f"root_cause  : {root_cause}")
    if resolution := incident.get("resolution"):
        lines.append(f"resolution  : {resolution}")
    if affected := incident.get("services_affected"):
        lines.append(f"services_affected : {', '.join(affected)}")
    return "\n".join(lines)


def _format_slack_thread(thread: dict) -> str:
    """Render a Slack thread as a chronological transcript."""
    header = [
        f"thread_id : {thread.get('thread_id', '')}",
        f"channel   : {thread.get('channel', '')}",
        "",
    ]
    messages: list[dict] = thread.get("messages", [])

    # Truncate very long threads: keep first half + last half
    if len(messages) > MAX_SLACK_MSGS:
        keep = MAX_SLACK_MSGS // 2
        omitted = len(messages) - MAX_SLACK_MSGS
        messages = (
            messages[:keep]
            + [{"author": "system", "text": f"[{omitted} messages omitted for brevity]", "timestamp": ""}]
            + messages[-keep:]
        )

    lines: list[str] = header
    for msg in messages:
        author = msg.get("author", "unknown")
        text   = (msg.get("text") or "").strip()
        ts     = (msg.get("timestamp") or "")[:16].replace("T", " ")
        if text:
            lines.append(f"[{ts}] {author}: {text}")
    return "\n".join(lines)


def build_prompt(incident: dict, slack_threads: list[dict]) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the LLM extraction call.
    """
    incident_block = _format_incident_report(incident)

    if slack_threads:
        slack_block = "\n\n---\n\n".join(
            _format_slack_thread(t) for t in slack_threads
        )
    else:
        slack_block = "(no Slack thread available for this incident)"

    user_prompt = EXTRACTION_PROMPT.format(
        incident_report=incident_block,
        slack_threads=slack_block,
    )
    return SYSTEM_PROMPT, user_prompt


# ── Step 3: LLM extraction ───────────────────────────────────────────────────

def _strip_to_json(text: str) -> str:
    """
    Remove markdown fences and any surrounding prose, returning only the JSON object.
    Handles: ```json {...} ```, ``` {...} ```, and bare JSON.
    """
    text = text.strip()
    # Try to pull from code fences first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Fall back to first {...} block
    bare = re.search(r"\{.*\}", text, re.DOTALL)
    if bare:
        return bare.group(0).strip()
    return text


def extract_signals(
    system_prompt: str,
    user_prompt: str,
    client: Groq,
    model: str,
    incident_id: str,
) -> Optional[dict]:
    """
    Call the Groq LLM and return the parsed extraction dict.
    Returns None if all retry attempts fail.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            json_str = _strip_to_json(raw_text)
            return json.loads(json_str)

        except json.JSONDecodeError as exc:
            log.warning(
                "%s  attempt %d/%d  JSON parse error: %s",
                incident_id, attempt, RETRY_ATTEMPTS, exc,
            )
        except Exception as exc:
            log.warning(
                "%s  attempt %d/%d  API error: %s",
                incident_id, attempt, RETRY_ATTEMPTS, exc,
            )

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY_SEC * attempt)  # simple exponential back-off

    log.error("%s  all %d attempts failed", incident_id, RETRY_ATTEMPTS)
    return None


# ── Step 4: Validate and normalise output ────────────────────────────────────

def _confidence_from_completeness(record: dict) -> float:
    """
    Compute a data-driven confidence score based on how many key fields
    the LLM was able to populate. Used as a fallback / sanity check.
    """
    score = 0.0
    if record.get("confirmed_root_cause"):
        score += 0.35
    if record.get("resolution_action"):
        score += 0.25
    if record.get("symptoms"):
        score += 0.15
    if record.get("services_affected"):
        score += 0.10
    if record.get("hypotheses_tested"):
        score += 0.08
    if record.get("failed_attempts"):
        score += 0.07
    return round(min(score, 1.0), 2)


def validate_output(raw: Optional[dict], incident: dict) -> dict:
    """
    Validate LLM output against the required schema.

    - Merges against EMPTY_RECORD defaults so every key is always present
    - Cleans list fields (removes empty strings)
    - Validates / clamps confidence score
    - Falls back gracefully on None input
    """
    result: dict = dict(EMPTY_RECORD)
    inc_id = incident.get("incident_id", "UNKNOWN")

    if raw is None:
        result["incident_id"] = inc_id
        result["service"]     = incident.get("service")
        result["confidence"]  = 0.0
        return result

    # ── Scalar fields ────────────────────────────────────────────────────────
    result["incident_id"] = str(raw.get("incident_id") or inc_id).strip()
    result["service"]     = _clean_str(raw.get("service") or incident.get("service"))

    result["confirmed_root_cause"] = _clean_str(raw.get("confirmed_root_cause"))
    result["resolution_action"]    = _clean_str(raw.get("resolution_action"))

    # ── List fields ──────────────────────────────────────────────────────────
    for field in ("symptoms", "hypotheses_tested", "failed_attempts", "services_affected"):
        raw_val = raw.get(field, [])
        if isinstance(raw_val, list):
            result[field] = [str(item).strip() for item in raw_val if item and str(item).strip()]
        else:
            result[field] = []  # malformed — default to empty

    # ── Confidence ───────────────────────────────────────────────────────────
    llm_conf = raw.get("confidence")
    if isinstance(llm_conf, (int, float)) and 0.0 <= float(llm_conf) <= 1.0:
        result["confidence"] = round(float(llm_conf), 2)
    else:
        result["confidence"] = _confidence_from_completeness(result)

    # Sanity: cap confidence if root cause is unknown
    if not result["confirmed_root_cause"] and result["confidence"] > 0.5:
        result["confidence"] = 0.5

    return result


def _clean_str(value) -> Optional[str]:
    """Return stripped string or None if empty/null."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


# ── Step 5: Save output ──────────────────────────────────────────────────────

def save_structured_data(record: dict) -> Path:
    """Write structured record to data/structured_incidents/<incident_id>.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{record['incident_id']}.json"
    out_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ── Orchestration ────────────────────────────────────────────────────────────

def process_incident(
    incident: dict,
    slack_threads: list[dict],
    client: Optional[Groq],
    model: str,
    dry_run: bool = False,
) -> dict:
    """
    Run the full extraction pipeline for one incident.
    Returns the validated structured record.
    """
    system_prompt, user_prompt = build_prompt(incident, slack_threads)

    if dry_run:
        inc_id = incident.get("incident_id", "?")
        log.info(
            "[DRY RUN] %s — %d slack messages — prompt %d chars",
            inc_id,
            sum(len(t.get("messages", [])) for t in slack_threads),
            len(user_prompt),
        )
        stub = dict(EMPTY_RECORD)
        stub["incident_id"] = inc_id
        stub["service"]     = incident.get("service")
        return stub

    raw    = extract_signals(system_prompt, user_prompt, client, model, incident["incident_id"])
    record = validate_output(raw, incident)
    return record


def run_pipeline(
    target_id: Optional[str] = None,
    dry_run:   bool = False,
    overwrite: bool = False,
    model:     str  = DEFAULT_MODEL,
) -> None:
    """
    Main orchestrator. Loads all data, processes each incident, saves output.
    """
    incidents, slack_by_incident = load_data()

    # ── Filter to single incident if requested ───────────────────────────────
    if target_id:
        if target_id not in incidents:
            log.error("Incident '%s' not found in %s", target_id, INCIDENTS_DIR)
            sys.exit(1)
        to_process = {target_id: incidents[target_id]}
    else:
        to_process = dict(incidents)

    # ── Skip already-extracted unless overwrite ──────────────────────────────
    if not overwrite and not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        already_done = {p.stem for p in OUTPUT_DIR.glob("INC-*.json")}
        skipped = [k for k in list(to_process) if k in already_done]
        for k in skipped:
            del to_process[k]
        if skipped:
            log.info(
                "Skipping %d already-extracted incidents — use --overwrite to reprocess",
                len(skipped),
            )

    if not to_process:
        log.info("Nothing to process. All incidents already extracted.")
        return

    # ── Initialise Groq client ───────────────────────────────────────────────
    client: Optional[Groq] = None
    if not dry_run:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            log.error("GROQ_API_KEY is not set. Add it to your .env file.")
            sys.exit(1)
        client = Groq(api_key=api_key)
        log.info("Using model: %s", model)

    log.info("Processing %d incidents...", len(to_process))

    success = 0
    failures: list[str] = []

    for inc_id, incident in tqdm(to_process.items(), desc="Extracting", unit="incident"):
        threads = slack_by_incident.get(inc_id, [])
        try:
            record = process_incident(incident, threads, client, model, dry_run)
            if not dry_run:
                out_path = save_structured_data(record)
                log.info(
                    "%-12s  confidence=%.2f  root_cause=%s  threads=%d  → %s",
                    inc_id,
                    record["confidence"],
                    "Y" if record["confirmed_root_cause"] else "N",
                    len(threads),
                    out_path.name,
                )
            success += 1
        except Exception as exc:
            log.error("%s failed: %s", inc_id, exc, exc_info=True)
            failures.append(inc_id)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(
        "Finished. success=%d  failed=%d  output=%s",
        success, len(failures), OUTPUT_DIR,
    )
    if failures:
        log.warning("Failed incidents: %s", ", ".join(failures))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured incident knowledge from raw reports and Slack threads. "
            "Outputs to data/structured_incidents/."
        )
    )
    parser.add_argument(
        "--id",
        dest="incident_id",
        metavar="INC-XXXX",
        help="Process a single incident by ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts but skip LLM calls (useful for prompt inspection)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process incidents that already have output files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    run_pipeline(
        target_id=args.incident_id,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        model=args.model,
    )


if __name__ == "__main__":
    main()
