"""Test configuration for gheim-py.

All tests run by default. The real-model tests in ``test_live_local.py``
prefer the local checkpoint at ``checkpoints/gheim-ch`` if it
exists, otherwise they download ``joelbarmettler/gheim-ch-560m`` from
the HuggingFace Hub on first run (~2.2 GB, cached after that).

Override with ``GHEIM_TEST_MODEL=<id-or-path>`` if you want to pin
the tests to a specific checkpoint.
"""
