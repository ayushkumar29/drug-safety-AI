"""
Unit test suite for Multi-Study Conflict Analysis (analysis/conflict_analysis.py).
"""

import unittest
from llm.evidence import ExtractedEvidence
from analysis.study_comparison import normalize_study_evidence
from analysis.conflict_analysis import (
    analyze_multi_study_conflicts,
    determine_effect_direction,
    determine_ci_relation,
    StudyEffectSummary,
    MultiStudyConflictAnalysisPayload
)


class TestMultiStudyConflictAnalysis(unittest.TestCase):

    def setUp(self):
        self.base_kwargs = {
            "study_id": "EMP-001",
            "drug": "empagliflozin",
            "comparator": "placebo",
            "outcome": "genital infection",
            "treatment_n": 4221,
            "treatment_events": 259,
            "control_n": 4203,
            "control_events": 67,
            "extracted_metric_type": "event_count",
            "dose": "10 mg",
            "source_file": "EMP-001.pdf",
            "chunk_id": "EMP-001_CHK_004",
            "evidence_quote": "Events consistent with genital infection 259 67"
        }

    def test_A_three_rr_studies_with_directional_conflict(self):
        """Test A: 3 RR studies (RR 3.8, RR 2.1, RR 0.9) produces directional_conflict = True."""
        k1 = self.base_kwargs.copy()  # RR ~ 3.85
        k2 = self.base_kwargs.copy()
        k2["study_id"] = "EMP-002"
        k2["treatment_events"] = 100  # RR ~ 2.37
        k2["chunk_id"] = "EMP-002_CHK_001"

        k3 = self.base_kwargs.copy()
        k3["study_id"] = "EMP-003"
        k3["treatment_events"] = 50   # RR ~ 0.74 (lower_effect)
        k3["chunk_id"] = "EMP-003_CHK_001"

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2), ExtractedEvidence(**k3)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        rr_group = g_res.representation_groups["risk_ratio"]

        self.assertTrue(rr_group.directional_conflict)
        self.assertTrue(g_res.directional_conflict)
        self.assertIn("higher_effect", rr_group.effect_directions_present)
        self.assertIn("lower_effect", rr_group.effect_directions_present)

    def test_B_same_direction_studies_no_directional_conflict(self):
        """Test B: Same-direction studies (RR 3.8, RR 2.37) produces directional_conflict = False."""
        k1 = self.base_kwargs.copy()
        k2 = self.base_kwargs.copy()
        k2["study_id"] = "EMP-002"
        k2["treatment_events"] = 150
        k2["chunk_id"] = "EMP-002_CHK_001"

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        rr_group = g_res.representation_groups["risk_ratio"]

        self.assertFalse(rr_group.directional_conflict)
        self.assertFalse(g_res.directional_conflict)

    def test_C_null_point_estimate(self):
        """Test C: RR == 1.0 produces null_effect."""
        direction = determine_effect_direction(1.0)
        self.assertEqual(direction, "null_effect")

    def test_D_ci_above_null(self):
        """Test D: RR 2.0 CI 1.4-2.8 produces above_null."""
        rel = determine_ci_relation(1.4, 2.8)
        self.assertEqual(rel, "above_null")

    def test_E_ci_below_null(self):
        """Test E: RR 0.6 CI 0.4-0.9 produces below_null."""
        rel = determine_ci_relation(0.4, 0.9)
        self.assertEqual(rel, "below_null")

    def test_F_ci_includes_null(self):
        """Test F: RR 1.2 CI 0.8-1.8 produces includes_null."""
        rel = determine_ci_relation(0.8, 1.8)
        self.assertEqual(rel, "includes_null")

    def test_G_mixed_ci_relation_differs(self):
        """Test G: One above_null study and one includes_null study produces confidence_interval_null_relation_differs = True."""
        k1 = self.base_kwargs.copy()  # RR ~ 3.85, CI ~ 2.9-5.0 (above_null)
        k2 = self.base_kwargs.copy()  # RR ~ 1.04, CI includes null
        k2["study_id"] = "EMP-004"
        k2["treatment_events"] = 70
        k2["chunk_id"] = "EMP-004_CHK_001"

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        rr_group = g_res.representation_groups["risk_ratio"]

        self.assertTrue(rr_group.confidence_interval_null_relation_differs)
        self.assertTrue(g_res.confidence_interval_null_relation_differs)

    def test_H_hr_vs_rr_separated(self):
        """Test H: HR study + RR study placed into separate effect-measure groups without direct comparison."""
        k1 = self.base_kwargs.copy()  # 2x2 event_count -> risk_ratio
        k2 = self.base_kwargs.copy()
        k2["study_id"] = "EMP-005"
        k2["extracted_metric_type"] = "hazard_ratio"
        k2["treatment_events"] = None
        k2["chunk_id"] = "EMP-005_CHK_004"

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        self.assertIn("risk_ratio", g_res.effect_measure_groups)
        self.assertIn("hazard_ratio", g_res.effect_measure_groups)
        self.assertIn("effect_measure_groups_are_separate", g_res.observations)

    def test_I_or_vs_rr_separated(self):
        """Test I: Odds ratio and Risk ratio separated without direct cross-metric conflict."""
        k1 = self.base_kwargs.copy()  # risk_ratio
        k2 = self.base_kwargs.copy()
        k2["study_id"] = "EMP-006"
        k2["extracted_metric_type"] = "odds_ratio"
        k2["treatment_events"] = None

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        self.assertIn("risk_ratio", g_res.effect_measure_groups)
        self.assertIn("odds_ratio", g_res.effect_measure_groups)


    def test_J_incidence_rate_evidence_no_unsupported_comparison(self):
        """Test J: Incidence-rate evidence preserved cleanly without unsupported numerical comparison."""
        k1 = self.base_kwargs.copy()
        k1["extracted_metric_type"] = "incidence_rate"
        k1["treatment_events"] = None

        evs = [ExtractedEvidence(**k1)]
        comp = normalize_study_evidence(evs)
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        self.assertFalse(g_res.directional_conflict)

    def test_K_missing_ci_handled_safely(self):
        """Test K: Missing CI is preserved with ci_relation == 'not_available'."""
        rel = determine_ci_relation(None, None)
        self.assertEqual(rel, "not_available")

    def test_L_provenance_preserved(self):
        """Test L: Provenance citation metadata preserved in StudyEffectSummary."""
        ev = ExtractedEvidence(**self.base_kwargs)
        comp = normalize_study_evidence([ev])
        analysis = analyze_multi_study_conflicts(comp)

        g_res = list(analysis.results.values())[0]
        summary = g_res.study_effects[0]
        self.assertEqual(summary.study_id, "EMP-001")
        self.assertEqual(summary.source_file, "EMP-001.pdf")
        self.assertEqual(summary.chunk_id, "EMP-001_CHK_004")

    def test_M_no_ranking_or_clinical_judgement_keys(self):
        """Test M: Verify payload models do not contain forbidden ranking/safety fields."""
        ev = ExtractedEvidence(**self.base_kwargs)
        comp = normalize_study_evidence([ev])
        analysis = analyze_multi_study_conflicts(comp)

        dump_str = analysis.model_dump_json().lower()
        forbidden_terms = ["best", "worst", "winner", "loser", "safety_score", "drug_is_safe", "drug_is_unsafe"]
        for term in forbidden_terms:
            self.assertNotIn(term, dump_str)


if __name__ == "__main__":
    unittest.main()
