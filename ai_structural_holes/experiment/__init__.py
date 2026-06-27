"""Experiment orchestration: run trials, collect tidy result frames."""

from .runner import ExperimentRunner, trials_to_frame
from .planning import CallPlan, compute_plan

__all__ = ["ExperimentRunner", "trials_to_frame", "CallPlan", "compute_plan"]
