"""LiteLLM proxy guardrail — gheim PII round-trip as a drop-in plugin.

``config.yaml``::

    guardrails:
      - guardrail_name: gheim-pii
        litellm_params:
          guardrail: gheim.litellm.GheimGuardrail
          mode: pre_call
          default_on: true

Requires the ``gheim[litellm]`` extra (and ``gheim[local]`` for the local model).
"""
from ._guardrail import GheimGuardrail

__all__ = ["GheimGuardrail"]
