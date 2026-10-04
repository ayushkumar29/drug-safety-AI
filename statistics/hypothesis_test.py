"""
Deterministic Hypothesis Testing Engine.

Performs Fisher's Exact Test and Pearson Chi-Square Test on 2x2 contingency tables
using SciPy stats without empirical corrections or LLM intervention.
"""

from typing import List, Tuple
from pydantic import BaseModel, Field, model_validator
from scipy import stats


class HypothesisTestInput(BaseModel):
    treatment_events: int = Field(..., description="Number of events in treatment group")
    treatment_n: int = Field(..., description="Total sample size of treatment group")
    control_events: int = Field(..., description="Number of events in control group")
    control_n: int = Field(..., description="Total sample size of control group")

    @model_validator(mode="after")
    def validate_counts(self) -> "HypothesisTestInput":
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


class HypothesisTestResult(BaseModel):
    treatment_events: int
    treatment_n: int
    control_events: int
    control_n: int
    a: int
    b: int
    c: int
    d: int
    fisher_odds_ratio: float
    fisher_p_value: float
    fisher_alternative: str = "two-sided"
    chi_square_statistic: float
    chi_square_p_value: float
    chi_square_degrees_of_freedom: int
    expected_frequencies: List[List[float]]


def run_hypothesis_test(
    treatment_events: int,
    treatment_n: int,
    control_events: int,
    control_n: int
) -> HypothesisTestResult:
    """
    Runs deterministic Fisher's Exact Test and Pearson Chi-Square Test on a 2x2 table.

    Contingency Table Structure:
      [[a, b],
       [c, d]]
    where:
      a = treatment_events
      b = treatment_n - treatment_events
      c = control_events
      d = control_n - control_events
    """
    inp = HypothesisTestInput(
        treatment_events=treatment_events,
        treatment_n=treatment_n,
        control_events=control_events,
        control_n=control_n
    )

    a = inp.treatment_events
    b = inp.treatment_n - inp.treatment_events
    c = inp.control_events
    d = inp.control_n - inp.control_events

    table = [[a, b], [c, d]]

    # 1. Fisher's Exact Test (SciPy stats.fisher_exact)
    fisher_or, fisher_p = stats.fisher_exact(table, alternative="two-sided")

    # 2. Pearson Chi-Square Test (SciPy stats.chi2_contingency with correction=False)
    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(table, correction=False)

    return HypothesisTestResult(
        treatment_events=inp.treatment_events,
        treatment_n=inp.treatment_n,
        control_events=inp.control_events,
        control_n=inp.control_n,
        a=a,
        b=b,
        c=c,
        d=d,
        fisher_odds_ratio=float(fisher_or),
        fisher_p_value=float(fisher_p),
        fisher_alternative="two-sided",
        chi_square_statistic=float(chi2_stat),
        chi_square_p_value=float(chi2_p),
        chi_square_degrees_of_freedom=int(dof),
        expected_frequencies=expected.tolist()
    )
