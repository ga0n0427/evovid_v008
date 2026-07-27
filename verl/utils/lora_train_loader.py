"""Load a saved PEFT LoRA adapter so its weights can continue training."""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel


def resolve_lora_adapter(adapter_path: str) -> Path:
    """Accept either ``.../lora_adapter`` or its actor checkpoint directory."""
    root = Path(adapter_path).expanduser().resolve()
    for candidate in (root, root / "lora_adapter"):
        has_config = (candidate / "adapter_config.json").is_file()
        has_weights = any(
            (candidate / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if has_config and has_weights:
            return candidate
    raise FileNotFoundError(
        f"No PEFT LoRA adapter found below {root}. "
        "Expected adapter_config.json and adapter_model weights."
    )


def load_lora_for_training(base_model, adapter_path: str) -> PeftModel:
    """Return ``Base + LoRA`` with the loaded adapter trainable."""
    adapter_dir = resolve_lora_adapter(adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=True,
    )
    model.train()
    return model
