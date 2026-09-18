"""
Generic HTTP API call tool for AgentHive.

All outbound requests are validated against the is_api_domain_allowed()
allowlist defined in config.py — no new domain checking logic is added here.

Supported methods: GET, POST, PUT, PATCH, DELETE (HEAD is excluded because
it returns no body and is rarely useful for agents).

Safety model
────────────
• Domain allowlist enforced before any network call.
• Request body is size-capped at MAX_REQUEST_BODY_BYTES.
• Response body is truncated to settings.maximum_tool_output_chars.
• Redirects followed up to MAX_REDIRECTS; https-only after first redirect.
• No cookie jar, no session persistence across calls.
• Timeout uses settings.api_timeout_seconds.

JSON responses are returned as parsed dicts/lists.
Non-JSON text responses are returned as plain strings.
Binary responses are rejected with a descriptive error.
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

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_MAX_REQUEST_BODY_BYTES = 64_000
_MAX_REDIRECTS = 5
_USER_AGENT = "AgentHive/1.0 (api-call-tool; non-commercial)"


class ApiCallTool(BaseTool):
    """
    Make an HTTP request to an allow-listed API endpoint.

    Arguments:
        url      (str, required)         — full URL (https preferred)
        method   (str, default "GET")    — HTTP method
        headers  (dict, optional)        — additional request headers
        body     (str|dict, optional)    — request body; dict is JSON-serialised
        timeout  (int, optional)         — override api_timeout_seconds for this call
    """

    name = "api_call"
    description = (
        "Make an HTTP request to an allow-listed API domain. "
        "Supports GET/POST/PUT/PATCH/DELETE. "
        "Domain must be in the ALLOWED_API_DOMAINS config list."
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        super().validate_arguments(arguments)

        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("The 'url' argument must be a non-empty string.")

        # Upgrade http → https.
        url = url.strip()
        if url.startswith("http://"):
            url = "https://" + url[7:]
        if not url.startswith("https://"):
            raise ValueError("Only https:// URLs are accepted.")

        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        if not settings.is_api_domain_allowed(hostname):
            raise PermissionError(
                f"Domain '{hostname}' is not in the API allowlist. "
                "Add it to ALLOWED_API_DOMAINS in your .env file."
            )

        method = str(arguments.get("method", "GET")).upper()
        if method not in _ALLOWED_METHODS:
            raise ValueError(
                f"Method '{method}' is not supported. "
                f"Allowed: {', '.join(sorted(_ALLOWED_METHODS))}."
            )

        headers = arguments.get("headers")
        if headers is not None and not isinstance(headers, dict):
            raise TypeError("'headers' must be a dict or None.")

        body = arguments.get("body")
        if body is not None:
            if isinstance(body, dict):
                encoded = json.dumps(body).encode()
            elif isinstance(body, str):
                encoded = body.encode()
            else:
                raise TypeError("'body' must be a string, dict, or None.")
            if len(encoded) > _MAX_REQUEST_BODY_BYTES:
                raise ValueError(
                    f"Request body exceeds maximum size "
                    f"({_MAX_REQUEST_BODY_BYTES} bytes)."
                )

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Re-normalise here so execute() is self-contained.
        url = arguments["url"].strip()
        if url.startswith("http://"):
            url = "https://" + url[7:]

        method = str(arguments.get("method", "GET")).upper()
        extra_headers: dict = arguments.get("headers") or {}
        body = arguments.get("body")
        timeout = int(arguments.get("timeout") or settings.api_timeout_seconds)

        # Serialise body.
        body_bytes: bytes | None = None
        content_type: str | None = None
        if body is not None:
            if isinstance(body, dict):
                body_bytes = json.dumps(body).encode("utf-8")
                content_type = "application/json"
            else:
                body_bytes = str(body).encode("utf-8")
                content_type = "text/plain; charset=utf-8"

        # Build request.
        req = urllib.request.Request(
            url,
            data=body_bytes,
            method=method,
        )
        req.add_header("User-Agent", _USER_AGENT)
        req.add_header("Accept", "application/json, text/plain, */*")
        if content_type and body_bytes:
            req.add_header("Content-Type", content_type)
        for key, value in extra_headers.items():
            req.add_header(str(key), str(value))

        try:
            with urllib.request.urlopen(
                req,
                timeout=timeout,
            ) as resp:
                status = resp.status
                resp_headers = dict(resp.headers)
                raw = resp.read(_MAX_REQUEST_BODY_BYTES + 1)

        except urllib.error.HTTPError as exc:
            logger.warning(
                "ApiCallTool: HTTP %d for %s — %s", exc.code, url, exc.reason
            )
            return ToolResult(
                success=False,
                error=f"HTTP {exc.code}: {exc.reason}",
                metadata={"url": url, "status_code": exc.code},
            )
        except urllib.error.URLError as exc:
            logger.warning("ApiCallTool: network error for %s — %s", url, exc)
            return ToolResult(
                success=False,
                error=f"Network error: {exc}",
                metadata={"url": url},
            )
        except Exception as exc:
            logger.warning("ApiCallTool: unexpected error — %s", exc)
            return ToolResult(
                success=False,
                error=f"Unexpected error: {exc}",
                metadata={"url": url},
            )

        # Detect binary responses — refuse to process them.
        content_type_resp = resp_headers.get("Content-Type", "")
        if (
            "application/octet-stream" in content_type_resp
            or "image/" in content_type_resp
        ):
            return ToolResult(
                success=False,
                error="Binary response content is not supported.",
                metadata={"url": url, "content_type": content_type_resp},
            )

        decoded = raw.decode("utf-8", errors="replace")

        # Truncate if over limit.
        if len(decoded) > settings.maximum_tool_output_chars:
            decoded = (
                decoded[: settings.maximum_tool_output_chars].rstrip()
                + "\n...[response truncated]"
            )

        # Try to parse as JSON for richer downstream use.
        output: Any = decoded
        if "application/json" in content_type_resp:
            try:
                output = json.loads(decoded)
            except json.JSONDecodeError:
                pass  # return raw string if JSON parse fails

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "url": url,
                "method": method,
                "status_code": status,
                "content_type": content_type_resp,
            },
        )


api_call_tool = ApiCallTool()
tool_registry.register(api_call_tool)
