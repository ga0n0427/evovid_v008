"""Build chat messages from role-marked prompt templates."""

from __future__ import annotations

from typing import Any, Literal


SYSTEM_MARKER = "<|system|>"
USER_MARKER = "<|user|>"


def split_role_prompt(prompt: str) -> tuple[str | None, str]:
    """Split a prompt into optional system text and required user text."""
    text = prompt.strip()
    has_system = SYSTEM_MARKER in text
    has_user = USER_MARKER in text
    if not has_system and not has_user:
        return None, text
    if not text.startswith(SYSTEM_MARKER) or text.count(SYSTEM_MARKER) != 1 or text.count(USER_MARKER) != 1:
        raise ValueError("Role-marked prompt must contain one leading <|system|> and one <|user|> marker.")
    system_text, user_text = text[len(SYSTEM_MARKER) :].split(USER_MARKER, 1)
    system_text = system_text.strip()
    user_text = user_text.strip()
    if not system_text or not user_text:
        raise ValueError("System and user prompt blocks must both be non-empty.")
    return system_text, user_text


def _multimodal_content(text: str, modality: Literal["image", "video"]) -> list[dict[str, Any]]:
    placeholder = f"<{modality}>"
    content: list[dict[str, Any]] = []
    for index, part in enumerate(text.split(placeholder)):
        if index != 0:
            content.append({"type": modality})
        if part:
            content.append({"type": "text", "text": part})
    return content


def build_chat_messages(
    prompt: str,
    *,
    modality: Literal["image", "video"] | None = None,
) -> list[dict[str, Any]]:
    """Convert a role-marked prompt into Transformers chat-template messages."""
    system_text, user_text = split_role_prompt(prompt)
    messages: list[dict[str, Any]] = []
    if system_text is not None:
        messages.append({"role": "system", "content": system_text})
    user_content: str | list[dict[str, Any]]
    if modality is None:
        user_content = user_text
    else:
        user_content = _multimodal_content(user_text, modality)
    messages.append({"role": "user", "content": user_content})
    return messages
