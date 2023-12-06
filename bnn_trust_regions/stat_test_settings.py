from dataclasses import dataclass


@dataclass
class StatTestSettings:
    """Class for settings for statistical tests."""
    # binomial test settings
    confidence_interval: float = 0.95
    alpha: float = 0.05
