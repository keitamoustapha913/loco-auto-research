"""Compatibility wrapper for older dashboard callers."""
from dashboard.presentation import experiment_rows


def run_table(runs):
    return experiment_rows(runs)
