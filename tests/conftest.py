"""Registro de marcadores de pytest para tests/test_pk_tokens.py."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests de integración sobre disco")
    config.addinivalue_line(
        "markers",
        "requires_lost_data: tests que dependen de datos realmente perdidos "
        "(cohortes v5/v6, windows_v2/windows_v3 — paths.LOST_COHORTS)")

