"""Shim next to config.yaml: older LiteLLM releases only resolve custom guardrail
classes from a file beside the config, not from an installed package path."""
from gheim.litellm import GheimGuardrail  # noqa: F401
