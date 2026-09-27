"""Typed pipeline failures (v7 §46): every failure carries a stable machine-readable code."""

from __future__ import annotations


class PipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
