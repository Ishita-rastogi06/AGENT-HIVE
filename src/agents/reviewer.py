"""
Reviewer agent for AgentHive.

Compares the specialist's output against the subtask's expected_output using
an LLM call and emits a structured ReviewResult with decision, confidence,
and actionable feedback.  Falls back to a heuristic-only review when the LLM
is unavailable so the workflow never silently stalls.
"""

from __future__ import annotations

import json
import logging
import re

from src.agents.base import BaseAgent
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState, ReviewResult

logger = logging.getLogger(__name__)

# ── prompt templates ──────────────────────────────────────────────────────────

_REVIEW_SYSTEM = (
    "You are AgentHive's strict quality and correctness reviewer. "
    "Your primary duty is to perform a granular requirement-by-requirement audit. "
    "Compare EVERY explicit constraint or requirement keyword in the subtask instruction "
    "(e.g., 'monthly', 'group by', 'formatted table', 'sqlite3', 'pandas', etc.) directly against what the code/output actually implements. "
    "If ANY explicit requirement is missing, incomplete, or incorrectly implemented "
    "(for example: 'monthly revenue' requested but code only does total overall group by item without date truncation or strftime), "
    "you MUST reject the output with decision='retry', set confidence below threshold (e.g. 0.30-0.40), "
    "and state the exact missing requirement in feedback. "
    "Note: strftime('%Y-%m', date) already includes both year (%Y) and month (%m) — do NOT flag this as missing year unless the format string itself lacks %Y. "
    "Score the output from 0.0 to 1.0 and decide: approve (score >= threshold ONLY IF ALL explicit requirements are met), "
    "retry (recoverable problems or missing explicit requirements), or escalate (empty or dangerous content). "
    "Reply with ONLY a JSON object — no extra text:\n"
    '{"decision": "approve|retry|escalate", "confidence": 0.0-1.0, '
    '"feedback": "one sentence of specific, actionable feedback naming the exact missing requirement", '
    '"issues": ["issue1", "issue2"]}'
)

_REVIEW_USER_TEMPLATE = """\
Subtask title: {title}
Instruction: {instruction}
Expected output criteria: {expected_output}
Confidence threshold: {threshold}

Specialist output to audit (Evaluate fresh output):
---
{specialist_output}
---

Perform a strict requirement audit:
1. Extract all specific requirement keywords from the Instruction (e.g. 'monthly', 'grouped by item', 'pandas', 'sqlite3').
2. Check if the code/text actually satisfies EVERY single requirement.
3. If ANY requirement is missing or incomplete, set decision='retry', provide specific detailed feedback, list exact missing issues, and assign a confidence score proportional to completeness (between 0.10 and 0.55).
4. If ALL requirements are satisfied, set decision='approve' and confidence between 0.75 and 1.00.
5. MANDATORY CHECK FOR MONTHLY REVENUE TASKS: ONLY IF the subtask instruction explicitly requests 'monthly' or date-based grouping ('by month', 'per month'), verify that the SQL query includes date truncation (e.g. strftime('%Y-%m', date) or pandas dt.to_period('M')). If the subtask instruction does NOT request monthly date grouping, DO NOT require date truncation or strftime — standard GROUP BY (e.g. GROUP BY status) is completely valid and correct.
6. FACTUAL GROUNDING & ACRONYM CHECK: If the output cites specific acronyms, factual expansions, numbers, or web search claims, verify that they are factually accurate and non-hallucinated. If an acronym expansion is fabricated or contradictory (e.g., expanding 'FAME' as 'Faradere Incentives' instead of 'Faster Adoption and Manufacturing of Electric Vehicles'), set decision='retry', issues=['factual hallucination in acronym/fact expansion'], and confidence < 0.50.
7. GROUNDING MANDATE REFUSAL CHECK: If the specialist output explicitly states that search data/information was insufficient or limited (e.g. 'search data was insufficient', 'search results were limited', 'cannot identify specific breakthroughs'), do NOT set decision='retry' demanding fabricated specifics. Set decision='escalate', issues=['grounding mandate refusal due to insufficient search data'], and feedback stating that search data was insufficient.

Score this fresh output and respond with the JSON review object only."""


_GROUNDING_PATTERNS = [
    # Explicit grounding mandate / notice markers
    re.compile(r"\b(grounding mandate|search data notice|strict grounding)\b", re.IGNORECASE),
    # Co-occurrence of search/data/info/snippet/result with insufficient/limited/lacking/empty/thin/scarce/inadequate
    re.compile(
        r"\b(search|data|result|information|snippet|source|finding)s?\b[\s\w,]{0,60}\b(insufficient|limited|empty|thin|lacking|lack|scarce|inadequate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(insufficient|limited|empty|thin|lacking|lack|scarce|inadequate)\b[\s\w,]{0,60}\b(search|data|result|information|snippet|source|finding)s?\b",
        re.IGNORECASE,
    ),
    # Co-occurrence of search/data/info/result with unable/cannot/could not/can't/couldn't
    re.compile(
        r"\b(search|data|result|information|snippet|source)s?\b[\s\w,]{0,60}\b(unable|cannot|could not|can't|couldn't|fail|failed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(unable|cannot|could not|can't|couldn't)\b[\s\w,]{0,60}\b(identify|verify|confirm|specify|find|determine)\b",
        re.IGNORECASE,
    ),
    # Explicit "no relevant snippets" / "returned no results" / "insufficient to identify" / "insufficient or limited"
    re.compile(
        r"\b(returned no|no relevant|insufficient to|unable to identify|cannot identify|could not find|lack of data|insufficient or limited|limited or insufficient)\b",
        re.IGNORECASE,
    ),
]


def _is_grounding_refusal(specialist_output: str, tool_logs: list | None = None) -> bool:
    """
    Check if specialist output explicitly states search data was insufficient per grounding mandate.
    Only applies when web_search was actually attempted in tool_logs.
    """
    if not specialist_output:
        return False

    web_search_attempted = False
    if tool_logs:
        web_search_attempted = any(
            isinstance(l, dict) and l.get("tool_name") == "web_search"
            for l in tool_logs
        )

    if not web_search_attempted:
        return False

    return any(pattern.search(specialist_output) is not None for pattern in _GROUNDING_PATTERNS)


def _extract_ungrounded_claims(specialist_output: str, search_text: str, web_search_attempted: bool) -> list[str]:
    """
    Extract specific monetary figures, month-year dates, and acronym expansions from specialist_output
    and check if they are grounded in web_search results when web search was executed.

    Returns a list of ungrounded claim issue descriptions.
    """
    if not web_search_attempted:
        return []

    ungrounded: list[str] = []
    search_lower = (search_text or "").lower()

    # 1. Check specific monetary figures (e.g. ₹19,000, $5,000, 19,000)
    monetary_matches = re.findall(r"(?:₹|Rs\.?|\$)\s*[\d,]+|\b\d{1,3}(?:,\d{3})+\b", specialist_output)
    for num_str in monetary_matches:
        digits_only = re.sub(r"[^\d]", "", num_str)
        if len(digits_only) >= 4 and digits_only not in search_lower:
            issue_msg = f"ungrounded monetary/numeric figure '{num_str}' not present in web search results"
            if issue_msg not in ungrounded:
                ungrounded.append(issue_msg)

    # 2. Check specific month-year extension dates (e.g. March 2023, March 2024)
    date_matches = re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", specialist_output, re.IGNORECASE)
    for date_str in date_matches:
        if date_str.lower() not in search_lower:
            issue_msg = f"ungrounded date claim '{date_str}' not present in web search results"
            if issue_msg not in ungrounded:
                ungrounded.append(issue_msg)

    # 3. Check multi-word acronym expansions (e.g. FAME (Faraday Fellowship...))
    acronym_matches = re.findall(r"\b([A-Z]{2,10})\s*\(([^)]+)\)", specialist_output)
    ignored_acronyms = {"ev", "api", "sql", "http", "https", "url", "json", "html", "pdf", "ui", "ux", "ai", "ml", "llm", "db", "gdp"}
    for ac_name, ac_expansion in acronym_matches:
        ac_name_lower = ac_name.lower()
        if ac_name_lower in ignored_acronyms:
            continue
        exp_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", ac_expansion)]
        if exp_words and not any(w in search_lower for w in exp_words):
            issue_msg = f"ungrounded acronym expansion '{ac_name} ({ac_expansion})' not present in web search results"
            if issue_msg not in ungrounded:
                ungrounded.append(issue_msg)

    return ungrounded


class Reviewer(BaseAgent):
    """
    LLM-powered reviewer.

    Decision logic
    ──────────────
    • ``approve``   – confidence >= low_confidence_threshold
    • ``retry``     – confidence < threshold AND output is non-empty
    • ``escalate``  – output is empty/error, or LLM explicitly says escalate
    """

    name = "reviewer"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def review(self, specialist_output: str = "", instruction: str = "", expected_output: str = "", task_type: str = "research", state: AgentHiveState | None = None) -> dict:
        """Core review logic. External calls and unit test mocks patch this method."""
        st = state or {}
        current_subtask = dict(st.get("current_subtask") or {"instruction": instruction, "expected_output": expected_output})
        if "tool_logs" not in current_subtask and st.get("tool_logs"):
            current_subtask["tool_logs"] = st.get("tool_logs")
        specialist_out = specialist_output or (st.get("specialist_output") or "").strip()
        threshold = float(st.get("low_confidence_threshold") or settings.low_confidence_threshold)

        if not specialist_out:
            return {
                "decision": "escalate",
                "approved": False,
                "confidence": 0.0,
                "feedback": "Specialist returned no output.",
                "issues": ["Empty specialist output"],
            }

        review = self._llm_review(specialist_out, current_subtask, threshold)
        if review is None:
            logger.warning("Reviewer: LLM unavailable — using heuristic review")
            spec_conf = float((state or {}).get("specialist_confidence") or 0.5)
            review = self._heuristic_review(specialist_out, current_subtask, threshold, spec_conf)
        return review

    # ── LangGraph node ────────────────────────────────────────────────────────

    def run(self, state: AgentHiveState) -> dict:
        specialist_output = (state.get("specialist_output") or "").strip()
        current_subtask = state.get("current_subtask") or {}
        instruction = str(current_subtask.get("instruction") or "")
        expected_output = str(current_subtask.get("expected_output") or "")
        task_type = str(state.get("selected_agent") or "research")
        current_trace = state.get("trace", [])

        review = self.review(
            specialist_output=specialist_output,
            instruction=instruction,
            expected_output=expected_output,
            task_type=task_type,
            state=state,
        )

        decision = (review or {}).get("decision", "approve")
        return self._build_return(review, state, current_trace, decision)

        # ── 3. Fallback: heuristic if LLM unavailable ─────────────────────────
        if review is None:
            logger.warning("Reviewer: LLM unavailable — using heuristic review")
            review = self._heuristic_review(
                specialist_output,
                current_subtask,
                threshold,
                float(state.get("specialist_confidence") or 0.5),
            )

        return self._build_return(review, state, current_trace, review["decision"])

    # ── LLM review ────────────────────────────────────────────────────────────

    def _llm_review(
        self,
        specialist_output: str,
        subtask: dict,
        threshold: float,
    ) -> ReviewResult | None:
        """Call the LLM and parse its JSON review. Returns None on any failure."""
        prompt = _REVIEW_USER_TEMPLATE.format(
            title=subtask.get("title", "Unknown subtask"),
            instruction=subtask.get("instruction", ""),
            expected_output=subtask.get("expected_output", "A correct, complete response."),
            threshold=f"{threshold:.2f}",
            # Evaluate full specialist output without artificial prompt truncation
            specialist_output=specialist_output,
        )

        try:
            raw = self.llm.ask(prompt, _REVIEW_SYSTEM, num_predict=200)
        except TypeError:
            raw = self.llm.ask(prompt, _REVIEW_SYSTEM)
        if not raw:
            return None

        return self._parse_review_response(raw, threshold, specialist_output, subtask)

    def _parse_review_response(
        self,
        raw: str,
        threshold: float,
        specialist_output: str = "",
        subtask: dict | None = None,
    ) -> ReviewResult | None:
        """Extract and validate the JSON review object from the LLM response."""
        json_str = self._extract_json(raw)
        if not json_str:
            logger.warning("Reviewer: could not extract JSON from LLM response: %r", raw[:200])
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("Reviewer: JSON parse error — %s", exc)
            return None

        decision_raw = str(data.get("decision") or "").strip().lower()
        if decision_raw not in ("approve", "retry", "escalate"):
            decision_raw = "retry"   # safe default

        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        feedback = str(data.get("feedback") or "").strip() or "No specific feedback provided."
        issues = [str(i) for i in (data.get("issues") or []) if i]

        # ── Deterministic Requirement Verification Overrides ───────────────────
        st_info = subtask or {}
        instr_lower = (
            str(st_info.get("instruction") or "") + " " + str(st_info.get("expected_output") or "")
        ).lower()
        code_lower = specialist_output.lower()

        monthly_keywords = ["monthly", "by month", "per month", "each month", "month-by-month"]
        requires_monthly = any(kw in instr_lower for kw in monthly_keywords)

        if requires_monthly:
            if "group by" in code_lower and "strftime" not in code_lower and "to_period" not in code_lower and "month" not in code_lower:
                decision_raw = "retry"
                confidence = 0.35
                if "missing monthly date truncation in SQL query" not in issues:
                    issues.append("missing monthly date truncation in SQL query")
                feedback = "The code performs SQL aggregation but lacks monthly date truncation (e.g., strftime('%Y-%m', date))."
        else:
            # Task instruction does NOT request monthly date grouping: strip any hallucinated monthly date truncation issues
            monthly_issue_patterns = [
                "missing monthly date truncation",
                "monthly date truncation",
                "date truncation",
                "strftime",
            ]
            orig_len = len(issues)
            issues = [
                iss for iss in issues
                if not any(pat in iss.lower() for pat in monthly_issue_patterns)
            ]
            if len(issues) < orig_len:
                logger.info("Stripped irrelevant monthly date truncation issue because subtask does not request monthly grouping.")
                if len(issues) == 0:
                    decision_raw = "approve"
                    confidence = max(confidence, 0.85)
                    feedback = "Subtask requirements satisfied."

        # ── General Fact-Grounding & Acronym/Figure Verification Overrides ────
        tool_logs = list(st_info.get("tool_logs") or [])
        web_search_attempted = any(
            isinstance(l, dict) and l.get("tool_name") == "web_search"
            for l in tool_logs
        )
        search_text = " ".join([
            str(l.get("result_summary") or l.get("output") or "").lower()
            for l in tool_logs
            if isinstance(l, dict) and l.get("tool_name") == "web_search" and l.get("success")
        ])

        ungrounded_claims = _extract_ungrounded_claims(specialist_output, search_text, web_search_attempted)
        if ungrounded_claims:
            decision_raw = "retry"
            confidence = 0.30
            for ug in ungrounded_claims:
                if ug not in issues:
                    issues.append(ug)
            feedback = f"The research output contains ungrounded factual claims not supported by web search results: {', '.join(ungrounded_claims)}"

        # ── Grounding Mandate Refusal Verification Overrides ──────────────────
        is_refusal = _is_grounding_refusal(specialist_output, tool_logs=tool_logs)
        logger.info(
            "[LIVE_CHECK] Reviewer evaluated output. Grounding refusal detected: %s. Raw LLM decision: %s",
            is_refusal,
            decision_raw,
        )
        if is_refusal and decision_raw == "retry":
            decision_raw = "escalate"
            confidence = 0.0
            if "grounding mandate refusal due to insufficient search data" not in issues:
                issues.append("grounding mandate refusal due to insufficient search data")
            feedback = "Specialist output correctly reported search data was insufficient to satisfy task requirements. Escalating immediately to human review to prevent pointless retries."

        # ── Decision & Confidence Reconciliation Rules ──────────────────────────
        if decision_raw == "escalate":
            confidence = 0.0
        elif len(issues) > 0:
            # Substantively-identified flaws MUST remain a retry regardless of numerical confidence score
            decision_raw = "retry"
            confidence = min(confidence, max(0.40, threshold - 0.05))
        elif confidence < threshold:
            # Low confidence with no explicit issues still requires retry
            decision_raw = "retry"
            confidence = min(confidence, max(0.40, threshold - 0.05))
        else:
            # High confidence (>= threshold) AND 0 concrete issues -> Approve
            decision_raw = "approve"

        # Prevent multi-pass retries when specialist output is complete valid code with 0 issues.
        if "```" in specialist_output and len(specialist_output) >= 150 and len(issues) == 0 and decision_raw == "approve":
            confidence = max(confidence, threshold)

        return ReviewResult(
            decision=decision_raw,       # type: ignore[arg-type]
            approved=(decision_raw == "approve"),
            confidence=round(confidence, 3),
            feedback=feedback,
            issues=issues,
        )

    # ── heuristic fallback ────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_review(
        specialist_output: str,
        subtask: dict,
        threshold: float,
        specialist_confidence: float,
    ) -> ReviewResult:
        """
        Score the output without an LLM call.

        Criteria:
        • Length ≥ 80 chars                          +0.20
        • Contains code block (```)                  +0.15
        • Keyword overlap with expected_output       +0.10
        • Specialist's own confidence (weighted 0.5) +up to 0.55
        """
        tool_logs = list(subtask.get("tool_logs") or [])
        if _is_grounding_refusal(specialist_output, tool_logs=tool_logs):
            return ReviewResult(
                decision="escalate",
                approved=False,
                confidence=0.0,
                feedback="Specialist correctly reported search data was insufficient. Escalating to human review.",
                issues=["grounding mandate refusal due to insufficient search data"],
            )

        score = specialist_confidence * 0.55

        if len(specialist_output) >= 80:
            score += 0.20
        if "```" in specialist_output:
            score += 0.15
        expected = subtask.get("expected_output", "")
        if expected:
            keywords = [w for w in expected.lower().split() if len(w) > 4]
            if keywords:
                hits = sum(1 for kw in keywords if kw in specialist_output.lower())
                score += min(0.10, 0.10 * hits / len(keywords))

        score = round(max(0.0, min(1.0, score)), 3)
        decision = "approve" if score >= threshold else "retry"
        feedback = (
            "Heuristic review: output meets length and content criteria."
            if decision == "approve"
            else "Heuristic review: output may be incomplete or too short."
        )

        return ReviewResult(
            decision=decision,          # type: ignore[arg-type]
            approved=(decision == "approve"),
            confidence=score,
            feedback=feedback,
            issues=[] if decision == "approve" else ["Low confidence score"],
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Pull the first JSON object from an LLM response."""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)
        bare = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
        if bare:
            return bare.group(1)
        return None

    @staticmethod
    def _build_return(
        review: ReviewResult,
        state: AgentHiveState,
        current_trace: list,
        decision: str,
    ) -> dict:
        """Assemble the state update dict returned to LangGraph."""
        approved = review.get("approved", False)
        confidence = review.get("confidence", 0.0)
        feedback = review.get("feedback", "")

        trace_entry = (
            f"Reviewer: {decision.upper()} "
            f"(confidence={confidence:.2f}) — {feedback[:80]}"
        )

        update: dict = {
            "review": review,
            "final_answer": state.get("specialist_output", "") if approved else "",
            "trace": [*current_trace, trace_entry],
        }

        return update
