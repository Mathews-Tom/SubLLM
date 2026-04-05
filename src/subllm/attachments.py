"""Attachment validation and normalization for message content arrays."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from subllm.errors import InvalidAttachmentError
from subllm.types import (
    FileContentPart,
    ImageContentPart,
    ProviderMessage,
    RequestMessage,
    ResolvedImageInput,
    TextContentPart,
)

MAX_ATTACHMENT_BYTES = 5_000_000
TEXT_FILE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def provider_message_from_request_message(message: RequestMessage) -> ProviderMessage:
    if isinstance(message.content, str):
        return {
            "role": message.role,
            "content": message.content,
            "images": [],
        }

    text_parts: list[str] = []
    images: list[ResolvedImageInput] = []
    for part in message.content:
        if isinstance(part, TextContentPart):
            text_parts.append(part.text)
        elif isinstance(part, ImageContentPart):
            image = _resolve_image_url(part.image_url.url)
            images.append(image)
            text_parts.append(f"[Attached image: {image.filename}]")
        elif isinstance(part, FileContentPart):
            text_fragment, image_input = _resolve_file_input(part.file_path)
            text_parts.append(text_fragment)
            if image_input is not None:
                images.append(image_input)

    content = "\n\n".join(part for part in text_parts if part)
    return {
        "role": message.role,
        "content": content,
        "images": images,
    }


def message_has_images(message: ProviderMessage) -> bool:
    return bool(message.get("images", []))


def _resolve_image_url(url: str) -> ResolvedImageInput:
    if url.startswith("data:"):
        return _resolve_data_url(url)

    parsed = urlparse(url)
    if parsed.scheme == "file":
        return _resolve_local_path(Path(parsed.path))

    path = Path(url)
    if path.is_absolute():
        return _resolve_local_path(path)

    raise InvalidAttachmentError(
        message="Image inputs must use a data URL or an absolute local file path",
    )


def _resolve_data_url(url: str) -> ResolvedImageInput:
    try:
        header, encoded = url.split(",", 1)
    except ValueError as exc:
        raise InvalidAttachmentError(message="Malformed image data URL") from exc

    if ";base64" not in header:
        raise InvalidAttachmentError(message="Image data URLs must use base64 encoding")

    media_type = header[5:].split(";", 1)[0]
    if not media_type.startswith("image/"):
        raise InvalidAttachmentError(message="Only image data URLs are supported")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise InvalidAttachmentError(message="Image data URL is not valid base64") from exc

    if len(payload) > MAX_ATTACHMENT_BYTES:
        raise InvalidAttachmentError(
            message=f"Attachment exceeds the maximum size of {MAX_ATTACHMENT_BYTES} bytes"
        )

    extension = mimetypes.guess_extension(media_type) or ".bin"
    return ResolvedImageInput(
        filename=f"embedded-image{extension}",
        media_type=media_type,
        data=payload,
    )


def _resolve_file_input(file_path: str) -> tuple[str, ResolvedImageInput | None]:
    path = Path(file_path)
    if not path.is_absolute():
        raise InvalidAttachmentError(message="File inputs must use absolute paths")
    if not path.exists() or not path.is_file():
        raise InvalidAttachmentError(message=f"File not found: {file_path}")

    stat = path.stat()
    if stat.st_size > MAX_ATTACHMENT_BYTES:
        raise InvalidAttachmentError(
            message=f"Attachment exceeds the maximum size of {MAX_ATTACHMENT_BYTES} bytes"
        )

    media_type, _ = mimetypes.guess_type(path.name)
    resolved_media_type = media_type or "application/octet-stream"

    if resolved_media_type.startswith("image/"):
        image = _resolve_local_path(path)
        return f"[Attached image file: {image.filename}]", image

    if path.suffix.lower() not in TEXT_FILE_EXTENSIONS and not resolved_media_type.startswith(
        "text/"
    ):
        raise InvalidAttachmentError(
            message=(
                f"Unsupported file type '{resolved_media_type}'. "
                "Only text files and images are supported."
            )
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidAttachmentError(
            message=f"File '{path.name}' must be valid UTF-8 text"
        ) from exc

    return f'<attached_file name="{path.name}">\n{text}\n</attached_file>', None


def _resolve_local_path(path: Path) -> ResolvedImageInput:
    if not path.exists() or not path.is_file():
        raise InvalidAttachmentError(message=f"File not found: {path}")
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("image/"):
        raise InvalidAttachmentError(message=f"Unsupported image file type for '{path.name}'")
    if path.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise InvalidAttachmentError(
            message=f"Attachment exceeds the maximum size of {MAX_ATTACHMENT_BYTES} bytes"
        )
    return ResolvedImageInput(
        filename=path.name,
        media_type=media_type,
        file_path=str(path),
    )
