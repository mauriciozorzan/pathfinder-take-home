"""Optional photo-analysis support for anchored receipt items.

The default behavior is safe and local: without environment configuration the
pipeline only records whether a photo source is reachable. When enabled, the AI
path fetches the reference image, asks for structured visual evidence, and
returns data that the resolver still must adjudicate before using.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import ReceiptItem


@dataclass(slots=True)
class PhotoEvidence:
    """Secondary evidence returned by a selective photo-analysis stage."""

    used: bool
    success: bool
    summary: str | None
    extracted_signals: list[str] = field(default_factory=list)
    suggested_title: str | None = None
    suggested_description: str | None = None
    confidence_delta: float = 0.0
    notes: str | None = None
    analyzer_name: str = "noop"
    model_confidence: float | None = None
    model_suggested_title: str | None = None
    model_suggested_description: str | None = None
    model_notes: str | None = None
    is_sufficient_for_exact_identification: bool = False


class PhotoAnalyzer(Protocol):
    """Pluggable interface for photo analyzers."""

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        """Analyze a reference photo and return structured photo evidence."""


class NoopPhotoAnalyzer:
    """Safe default that disables photo assistance without changing behavior."""

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        return PhotoEvidence(
            used=False,
            success=False,
            summary=None,
            extracted_signals=[],
            suggested_title=None,
            suggested_description=None,
            confidence_delta=0.0,
            notes="No photo analyzer configured.",
            analyzer_name="noop",
        )


class BasicPhotoAnalyzer:
    """Small heuristic stand-in for a future vision-backed analyzer.

    This analyzer is intentionally narrow. It keeps the photo path modular and
    selective, and optionally probes that an image source is reachable. It does
    not perform real visual understanding and intentionally avoids hardcoded
    product-specific hints. A future VLM or provider-backed analyzer can replace
    this class without changing the resolver contract.
    """

    def __init__(self, *, probe_sources: bool = True, timeout_seconds: float = 2.0) -> None:
        self.probe_sources = probe_sources
        self.timeout_seconds = timeout_seconds

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        source_ok, source_note = self._probe_source(image_source)
        return PhotoEvidence(
            used=False,
            success=False,
            summary=None,
            extracted_signals=[],
            suggested_title=None,
            suggested_description=None,
            confidence_delta=0.0,
            notes=source_note
            or (
                "Photo source was available, but no vision backend is configured. "
                "Basic analyzer records availability only and does not infer product identity."
                if source_ok
                else "Photo source was unavailable and no vision backend is configured."
            ),
            analyzer_name="basic",
        )

    def _probe_source(self, image_source: str) -> tuple[bool, str | None]:
        if not self.probe_sources:
            return True, "Image probing disabled; analyzer used configured fallback behavior."

        parsed = urlparse(image_source)
        if parsed.scheme in {"http", "https"}:
            try:
                request = Request(image_source, headers={"User-Agent": "exact-item-ai/0.1"})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    content_type = response.headers.get("Content-Type", "unknown")
                return True, f"Reference photo source was reachable ({content_type})."
            except Exception as exc:  # pragma: no cover - exercised opportunistically on real data
                return False, f"Reference photo probe failed: {exc!s}"

        local_path = Path(image_source)
        if local_path.exists():
            return True, f"Loaded local reference photo from {local_path.name}."
        return False, "Reference photo source was unavailable."


@dataclass(slots=True)
class PhotoAIConfig:
    """Environment-backed configuration for the AI photo analyzer."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout_seconds: float = 20.0
    max_image_bytes: int = 4_000_000
    max_photo_analyses: int | None = None

    @classmethod
    def from_env(cls) -> "PhotoAIConfig":
        """Build photo-AI settings from environment variables and an optional .env file."""

        env_file_values = _load_env_file(Path.cwd() / ".env")
        return cls(
            enabled=_env_bool("PHOTO_AI_ENABLED", default=False, env_file_values=env_file_values),
            provider=_config_value("PHOTO_AI_PROVIDER", "openai", env_file_values).strip().lower(),
            model=_config_value("PHOTO_AI_MODEL", "gpt-4o-mini", env_file_values).strip(),
            api_key=_config_value("PHOTO_AI_API_KEY", "", env_file_values)
            or _config_value("OPENAI_API_KEY", "", env_file_values)
            or None,
            timeout_seconds=_env_float("PHOTO_AI_TIMEOUT_SECONDS", default=20.0, env_file_values=env_file_values),
            max_image_bytes=_env_int("PHOTO_AI_MAX_IMAGE_BYTES", default=4_000_000, env_file_values=env_file_values),
            max_photo_analyses=_env_optional_int("PHOTO_AI_MAX_ANALYSES", env_file_values=env_file_values),
        )


class AIBasedPhotoAnalyzer:
    """Vision-backed analyzer for selective low-confidence anchored items.

    The analyzer fetches the actual reference image, sends it with structured
    receipt-item context to a configured vision-capable provider, and returns
    conservative structured evidence. It fails closed: missing config, fetch
    errors, provider errors, and weak model confidence all return non-mutating
    evidence.
    """

    def __init__(self, config: PhotoAIConfig | None = None) -> None:
        self.config = config if config is not None else PhotoAIConfig.from_env()
        self.analysis_count = 0
        self.last_photo_fetch_ms = 0.0
        self.last_photo_model_ms = 0.0
        self.last_external_api_call_count = 0

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        """Fetch and analyze a photo, failing closed whenever configuration or evidence is weak."""

        self.last_photo_fetch_ms = 0.0
        self.last_photo_model_ms = 0.0
        self.last_external_api_call_count = 0
        if not self.config.enabled:
            # Missing or disabled config returns non-mutating evidence instead of
            # raising, so local pipeline runs stay deterministic.
            return self._disabled("PHOTO_AI_ENABLED is not true.")
        if self.config.provider != "openai":
            return self._disabled(f"Unsupported PHOTO_AI_PROVIDER: {self.config.provider}.")
        if not self.config.api_key:
            return self._disabled("Missing PHOTO_AI_API_KEY or OPENAI_API_KEY.")
        if self.config.max_photo_analyses is not None and self.analysis_count >= self.config.max_photo_analyses:
            return self._disabled("PHOTO_AI_MAX_ANALYSES limit reached.")

        fetch_start = time.perf_counter()
        image = fetch_image_bytes(
            image_source,
            timeout_seconds=self.config.timeout_seconds,
            max_image_bytes=self.config.max_image_bytes,
        )
        self.last_photo_fetch_ms = round((time.perf_counter() - fetch_start) * 1000, 2)
        if not image.success or image.content is None:
            return PhotoEvidence(
                used=False,
                success=False,
                summary=None,
                extracted_signals=[],
                confidence_delta=0.0,
                notes=image.error or "Image fetch failed.",
                analyzer_name="ai_openai",
            )

        self.analysis_count += 1
        try:
            model_start = time.perf_counter()
            model_payload = self._call_openai_vision(image.content, image.mime_type, item_context)
            self.last_photo_model_ms = round((time.perf_counter() - model_start) * 1000, 2)
            self.last_external_api_call_count = 1
            return self._evidence_from_model_payload(model_payload)
        except Exception as exc:
            return PhotoEvidence(
                used=False,
                success=False,
                summary=None,
                extracted_signals=[],
                confidence_delta=0.0,
                notes=f"Photo AI provider call failed: {exc!s}",
                analyzer_name="ai_openai",
            )

    def _disabled(self, note: str) -> PhotoEvidence:
        """Return a standard no-op evidence object for unavailable AI analysis."""

        return PhotoEvidence(
            used=False,
            success=False,
            summary=None,
            extracted_signals=[],
            confidence_delta=0.0,
            notes=note,
            analyzer_name="ai_openai",
        )

    def _call_openai_vision(self, image_bytes: bytes, mime_type: str, item_context: ReceiptItem) -> dict[str, Any]:
        """Send the encoded image and receipt context to the configured vision model."""

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{image_b64}"
        prompt = build_photo_analysis_prompt(item_context)
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 600,
        }
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "exact-item-ai/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
        return json.loads(content)

    def _evidence_from_model_payload(self, payload: dict[str, Any]) -> PhotoEvidence:
        """Normalize the model JSON response into the resolver's evidence contract."""

        model_confidence = _coerce_float(payload.get("confidence"), default=0.0)
        sufficient = bool(payload.get("is_sufficient_for_exact_identification")) and model_confidence >= 0.72
        suggested_title = _string_or_none(payload.get("suggested_title"))
        suggested_description = _string_or_none(payload.get("suggested_description"))
        signals = _list_of_strings(payload.get("signals"))
        summary = _build_summary_from_payload(payload)

        confidence_delta = 0.0
        if sufficient and suggested_title:
            confidence_delta = min(0.22, max(0.08, (model_confidence - 0.55) * 0.35))
        elif model_confidence >= 0.55 and signals:
            confidence_delta = min(0.08, max(0.02, (model_confidence - 0.5) * 0.15))

        return PhotoEvidence(
            used=sufficient and bool(suggested_title),
            success=True,
            summary=summary,
            extracted_signals=signals,
            suggested_title=suggested_title if sufficient else None,
            suggested_description=suggested_description if sufficient else None,
            confidence_delta=round(confidence_delta, 2),
            notes=_string_or_none(payload.get("notes")),
            analyzer_name="ai_openai",
            model_confidence=round(model_confidence, 2),
            model_suggested_title=suggested_title,
            model_suggested_description=suggested_description,
            model_notes=_string_or_none(payload.get("notes")),
            is_sufficient_for_exact_identification=sufficient,
        )


@dataclass(slots=True)
class ImageFetchResult:
    success: bool
    content: bytes | None
    mime_type: str
    error: str | None = None


def create_default_photo_analyzer() -> PhotoAnalyzer:
    """Create the default analyzer from environment configuration."""
    config = PhotoAIConfig.from_env()
    if config.enabled:
        return AIBasedPhotoAnalyzer(config)
    return BasicPhotoAnalyzer()


def fetch_image_bytes(
    image_source: str,
    *,
    timeout_seconds: float,
    max_image_bytes: int,
) -> ImageFetchResult:
    """Load a remote or local image while enforcing size and MIME-type limits."""

    parsed = urlparse(image_source)
    if parsed.scheme in {"http", "https"}:
        # Remote image reads are capped so an unexpected large response cannot
        # dominate latency or memory usage.
        try:
            request = Request(image_source, headers={"User-Agent": "exact-item-ai/0.1"})
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    return ImageFetchResult(False, None, "application/octet-stream", f"Image fetch HTTP {status}.")
                mime_type = response.headers.get("Content-Type", "").split(";")[0].strip() or "application/octet-stream"
                content = response.read(max_image_bytes + 1)
        except Exception as exc:
            return ImageFetchResult(False, None, "application/octet-stream", f"Image fetch failed: {exc!s}")
    else:
        path = Path(image_source)
        if not path.exists():
            return ImageFetchResult(False, None, "application/octet-stream", "Local image path does not exist.")
        content = path.read_bytes()
        mime_type = guess_type(path.name)[0] or "application/octet-stream"

    if len(content) > max_image_bytes:
        return ImageFetchResult(False, None, mime_type, "Image exceeds PHOTO_AI_MAX_IMAGE_BYTES.")
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif"}:
        return ImageFetchResult(False, None, mime_type, f"Unsupported image content type: {mime_type}.")
    return ImageFetchResult(True, content, mime_type, None)


def build_photo_analysis_prompt(item_context: ReceiptItem) -> str:
    """Build the vision-model prompt for exact-item photo evidence extraction."""

    return f"""You are helping identify the exact product represented by a parsed receipt line item.

Use the image as secondary evidence together with the structured item context.
Do not perform receipt OCR. Do not guess if the image is unclear or if identity clues are not visible.
Prefer ambiguity over fabricated precision.

Structured item context:
- merchant: {item_context.merchant}
- item_name: {item_context.item_name}
- item_id: {item_context.item_id}
- item_price: {item_context.item_price}

Return only a JSON object with this shape:
{{
  "visible_title": string or null,
  "product_type": string or null,
  "brand_or_author": string or null,
  "manufacturer_or_publisher": string or null,
  "suggested_title": string or null,
  "suggested_description": string or null,
  "confidence": number between 0 and 1,
  "is_sufficient_for_exact_identification": boolean,
  "signals": array of short strings,
  "notes": string
}}

Only set is_sufficient_for_exact_identification=true when the image and context justify a specific exact item."""


def _load_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE lines from a local .env file."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _config_value(name: str, default: str, env_file_values: dict[str, str]) -> str:
    """Return an environment value, then .env value, then a default."""

    return os.getenv(name) or env_file_values.get(name) or default


def _env_bool(name: str, *, default: bool, env_file_values: dict[str, str]) -> bool:
    """Parse a boolean configuration value."""

    raw = os.getenv(name) or env_file_values.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int, env_file_values: dict[str, str]) -> int:
    """Parse an integer configuration value with a safe default."""

    raw = os.getenv(name) or env_file_values.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(name: str, *, env_file_values: dict[str, str]) -> int | None:
    """Parse an optional integer configuration value."""

    raw = os.getenv(name) or env_file_values.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str, *, default: float, env_file_values: dict[str, str]) -> float:
    """Parse a float configuration value with a safe default."""

    raw = os.getenv(name) or env_file_values.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    """Convert model-returned numeric values to floats."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_or_none(value: Any) -> str | None:
    """Normalize arbitrary model fields into optional strings."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_strings(value: Any) -> list[str]:
    """Normalize arbitrary model fields into a clean string list."""

    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_summary_from_payload(payload: dict[str, Any]) -> str:
    """Create a readable one-line summary from visual evidence fields."""

    pieces = []
    visible_title = _string_or_none(payload.get("visible_title"))
    product_type = _string_or_none(payload.get("product_type"))
    brand_or_author = _string_or_none(payload.get("brand_or_author"))
    if visible_title:
        pieces.append(f"visible title: {visible_title}")
    if product_type:
        pieces.append(f"type: {product_type}")
    if brand_or_author:
        pieces.append(f"brand/author: {brand_or_author}")
    notes = _string_or_none(payload.get("notes"))
    if notes:
        pieces.append(notes)
    return "; ".join(pieces) if pieces else "Photo AI returned no clear identity clues."
