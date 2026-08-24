"""Prototype RF Environment, Scanner, Detector, and Evaluator package.

Provides lightweight dataset-driven simulation and evaluation:
- `RFEnvironment`: loads tabular RF observation datasets (`data/prototype/temporary_rf_dataset.csv`).
- `Receiver`: simulates receiver frontend measurements.
- `BaselineScanner`, `RandomScanner`, `AdaptiveScanner`, `make_scanner`: frequency band scanning strategies.
- `PowerThresholdDetector`: power-threshold rule-based signal detection.
- `Evaluator`: ground-truth verification and confusion matrix metric calculation.
"""
from __future__ import annotations

from src.detector import PowerThresholdDetector
from src.environment import RFEnvironment
from src.evaluator import Evaluator
from src.receiver import Receiver
from src.scanner import AdaptiveScanner, BaselineScanner, RandomScanner, make_scanner

__all__ = [
    "RFEnvironment",
    "Receiver",
    "BaselineScanner",
    "RandomScanner",
    "AdaptiveScanner",
    "make_scanner",
    "PowerThresholdDetector",
    "Evaluator",
]
