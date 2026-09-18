"""
Simple web-search tool for AgentHive research specialists.

Uses the DuckDuckGo Instant Answer JSON API — no key required, no scraping.
Falls back gracefully when the network is unavailable or the API returns no
useful result.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.config import settings
from src.tools.base import BaseTool, ToolContext, ToolResult, tool_registry

logger = logging.getLogger(__name__)

# DuckDuckGo Instant Answer endpoint — no rate-limit key needed for
# low-volume, non-commercial use.
_DDG_URL = "https://api.duckduckgo.com/"


class WebSearchTool(BaseTool):
    """
    Perform a lightweight web search via the DuckDuckGo Instant Answer API.

    Returns the best available snippet: AbstractText → RelatedTopics → raw
    query suggestion, in that priority order.  The tool never raises on
    network failure; it returns a ToolResult with ``success=False`` so the
    calling specialist can degrade gracefully.
    """

    name = "web_search"
    description = (
        "Search the web for factual information using the DuckDuckGo Instant "
        "Answer API. Accepts a plain-text query; returns the best available "
        "text snippet. No API key required."
    )

    # Clamp query to a reasonable URL-safe length.
    MAX_QUERY_CHARS = 300
    # Cap the returned snippet so it fits inside a prompt comfortably.
    MAX_RESULT_CHARS = 1_200

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        super().validate_arguments(arguments)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The 'query' argument must be a non-empty string.")

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = arguments["query"].strip()[: self.MAX_QUERY_CHARS]

        # Reject disallowed domains (reuses the allowlist check from config,
        # but DDG is always permitted because requests go to duckduckgo.com).
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
        )
        url = f"{_DDG_URL}?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AgentHive/1.0 (research agent; +https://github.com/agenthive)"},
            )
            with urllib.request.urlopen(req, timeout=settings.api_timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            logger.warning("WebSearchTool: network error for query=%r — %s", query, exc)
            return ToolResult(
                success=False,
                error=f"Network error: {exc}",
                metadata={"query": query},
            )
        except Exception as exc:
            logger.warning("WebSearchTool: unexpected error — %s", exc)
            return ToolResult(
                success=False,
                error=f"Unexpected error: {exc}",
                metadata={"query": query},
            )

        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolResult(
                success=False,
                error=f"Could not parse API response: {exc}",
                metadata={"query": query},
            )

        snippet = self._extract_snippet(data, query)

        if not snippet:
            return ToolResult(
                success=False,
                error="No useful result returned by DuckDuckGo for this query.",
                metadata={"query": query},
            )

        return ToolResult(
            success=True,
            output=snippet[: self.MAX_RESULT_CHARS],
            metadata={
                "query": query,
                "source": data.get("AbstractURL") or data.get("AbstractSource") or "DuckDuckGo",
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_snippet(data: dict, query: str) -> str:
        """
        Pull the most useful text from a DDG Instant Answer response.

        Priority: AbstractText → first RelatedTopics Text → Heading.
        """
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            return abstract

        # RelatedTopics is a list of dicts; each has a "Text" key.
        for topic in data.get("RelatedTopics") or []:
            if isinstance(topic, dict):
                text = (topic.get("Text") or "").strip()
                if text:
                    return text

        heading = (data.get("Heading") or "").strip()
        if heading:
            return f"Topic: {heading} (no detailed abstract available)."

        return ""


# Register on import — idempotent thanks to the updated ToolRegistry.register.
web_search_tool = WebSearchTool()
tool_registry.register(web_search_tool)
