"""
Deterministic Risk Ratio Engine.

Calculates treatment risk, control risk, Risk Ratio (RR), and 95% Confidence Intervals
using standard log-transformed normal approximation with optional continuity correction.
"""

import math
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class RiskRatioInput(BaseModel):
    treatment_events: int = Field(..., description="Number of events in treatment group")
    treatment_n: int = Field(..., description="Total sample size of treatment group")
    control_events: int = Field(..., description="Number of events in control group")
    control_n: int = Field(..., description="Total sample size of control group")

    @model_validator(mode="after")
    def validate_counts(self) -> "RiskRatioInput":
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


class RiskRatioResult(BaseModel):
    treatment_events: int
    treatment_n: int
    control_events: int
    control_n: int
    treatment_risk: float
    control_risk: float
    risk_ratio: Optional[float]
    log_rr: Optional[float]
    se_log_rr: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_method: str
    zero_event_correction_applied: bool


def calculate_risk_ratio(
    treatment_events: int,
    treatment_n: int,
    control_events: int,
    control_n: int,
    alpha: float = 0.05
) -> RiskRatioResult:
    """
    Calculates deterministic Risk Ratio (RR) and (1-alpha) confidence interval.

    Standard log-RR method:
      log(RR) = log(p1 / p2)
      SE(log(RR)) = sqrt(1/a - 1/n1 + 1/c - 1/n2)
      CI = exp(log(RR) +/- z * SE)

    Haldane-Anscombe continuity correction (0.5 added to all cells) is applied
    if either treatment_events == 0 or control_events == 0.
    """
    # Validate inputs using Pydantic model
    inp = RiskRatioInput(
        treatment_events=treatment_events,
        treatment_n=treatment_n,
        control_events=control_events,
        control_n=control_n
    )

    tr_e = float(inp.treatment_events)
    tr_n = float(inp.treatment_n)
    ctrl_e = float(inp.control_events)
    ctrl_n = float(inp.control_n)

    # Raw risk calculation without modification
    tr_risk = tr_e / tr_n
    ctrl_risk = ctrl_e / ctrl_n

    zero_event_applied = (tr_e == 0 or ctrl_e == 0)

    # Check zero event edge cases
    if zero_event_applied:
        # Apply Haldane-Anscombe continuity correction (+0.5 to all cells)
        tr_e_adj = tr_e + 0.5
        ctrl_e_adj = ctrl_e + 0.5
        tr_n_adj = tr_n + 1.0
        ctrl_n_adj = ctrl_n + 1.0
        
        tr_risk_adj = tr_e_adj / tr_n_adj
        ctrl_risk_adj = ctrl_e_adj / ctrl_n_adj
        
        rr = tr_risk_adj / ctrl_risk_adj
        log_rr = math.log(rr)
        se_log_rr = math.sqrt(
            (1.0 / tr_e_adj) - (1.0 / tr_n_adj) + (1.0 / ctrl_e_adj) - (1.0 / ctrl_n_adj)
        )
        ci_method = "Log-transformed Normal Approximation (Haldane-Anscombe Continuity Correction)"
    else:
        rr = tr_risk / ctrl_risk
        log_rr = math.log(rr)
        se_log_rr = math.sqrt(
            (1.0 / tr_e) - (1.0 / tr_n) + (1.0 / ctrl_e) - (1.0 / ctrl_n)
        )
        ci_method = "Log-transformed Normal Approximation (Katz et al.)"

    z = 1.959963984540054

    ci_lower = math.exp(log_rr - z * se_log_rr)
    ci_upper = math.exp(log_rr + z * se_log_rr)

    return RiskRatioResult(
        treatment_events=inp.treatment_events,
        treatment_n=inp.treatment_n,
        control_events=inp.control_events,
        control_n=inp.control_n,
        treatment_risk=tr_risk,
        control_risk=ctrl_risk,
        risk_ratio=rr,
        log_rr=log_rr,
        se_log_rr=se_log_rr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_method=ci_method,
        zero_event_correction_applied=zero_event_applied
    )
