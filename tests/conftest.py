"""Registro de marcadores de pytest para tests/test_pk_tokens.py."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests de integración sobre disco")
