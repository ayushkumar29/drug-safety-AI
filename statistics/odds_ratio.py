"""
Deterministic Odds Ratio Engine.

Calculates treatment odds, control odds, Odds Ratio (OR), and 95% Confidence Intervals
using standard log-transformed Woolf normal approximation with optional continuity correction.
"""

import math
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class OddsRatioInput(BaseModel):
    treatment_events: int = Field(..., description="Number of events in treatment group")
    treatment_n: int = Field(..., description="Total sample size of treatment group")
    control_events: int = Field(..., description="Number of events in control group")
    control_n: int = Field(..., description="Total sample size of control group")

    @model_validator(mode="after")
    def validate_counts(self) -> "OddsRatioInput":
        if self.treatment_n <= 0:
            raise ValueError("treatment_n must be greater than 0.")
        if self.control_n <= 0:
            raise ValueError("control_n must be greater than 0.")
        if self.treatment_events < 0:
            raise ValueError("treatment_events cannot be negative.")
        if self.control_events < 0:
            raise ValueError("control_events cannot be negative.")
        if self.treatment_events > self.treatment_n:
            raise ValueError(f"treatment_events ({self.treatment_events}) cannot exceed treatment_n ({self.treatment_n}).")
        if self.control_events > self.control_n:
            raise ValueError(f"control_events ({self.control_events}) cannot exceed control_n ({self.control_n}).")
        return self


class OddsRatioResult(BaseModel):
    treatment_events: int
    treatment_n: int
    control_events: int
    control_n: int
    a: int
    b: int
    c: int
    d: int
    treatment_odds: float
    control_odds: float
    odds_ratio: Optional[float]
    log_odds_ratio: Optional[float]
    standard_error: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_method: str
    zero_cell_correction_applied: bool


def calculate_odds_ratio(
    treatment_events: int,
    treatment_n: int,
    control_events: int,
    control_n: int,
    alpha: float = 0.05
) -> OddsRatioResult:
    """
    Calculates deterministic Odds Ratio (OR) and (1-alpha) confidence interval.

    Derivations:
      a = treatment_events
      b = treatment_n - treatment_events
      c = control_events
      d = control_n - control_events

    Standard Woolf log-OR method:
      log(OR) = log((a * d) / (b * c))
      SE(log OR) = sqrt(1/a + 1/b + 1/c + 1/d)
      95% CI = exp(log(OR) +/- z * SE)

    Haldane-Anscombe continuity correction (+0.5 added to a, b, c, d) is applied
    if any cell (a, b, c, or d) equals zero.
    """
    inp = OddsRatioInput(
        treatment_events=treatment_events,
        treatment_n=treatment_n,
        control_events=control_events,
        control_n=control_n
    )

    a = inp.treatment_events
    b = inp.treatment_n - inp.treatment_events
    c = inp.control_events
    d = inp.control_n - inp.control_events

    tr_odds = a / b if b > 0 else float("inf")
    ctrl_odds = c / d if d > 0 else float("inf")

    zero_cell_applied = (a == 0 or b == 0 or c == 0 or d == 0)

    if zero_cell_applied:
        a_adj = a + 0.5
        b_adj = b + 0.5
        c_adj = c + 0.5
        d_adj = d + 0.5

        or_val = (a_adj * d_adj) / (b_adj * c_adj)
        log_or = math.log(or_val)
        se_log_or = math.sqrt((1.0 / a_adj) + (1.0 / b_adj) + (1.0 / c_adj) + (1.0 / d_adj))
        ci_method = "Log-transformed Normal Approximation (Woolf's Method with Haldane-Anscombe Continuity Correction)"
    else:
        or_val = (a * d) / (b * c)
        log_or = math.log(or_val)
        se_log_or = math.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))
        ci_method = "Log-transformed Normal Approximation (Woolf's Method)"

    z = 1.959963984540054

    ci_lower = math.exp(log_or - z * se_log_or)
    ci_upper = math.exp(log_or + z * se_log_or)

    return OddsRatioResult(
        treatment_events=inp.treatment_events,
        treatment_n=inp.treatment_n,
        control_events=inp.control_events,
        control_n=inp.control_n,
        a=a,
        b=b,
        c=c,
        d=d,
        treatment_odds=tr_odds,
        control_odds=ctrl_odds,
        odds_ratio=or_val,
        log_odds_ratio=log_or,
        standard_error=se_log_or,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_method=ci_method,
        zero_cell_correction_applied=zero_cell_applied
    )
