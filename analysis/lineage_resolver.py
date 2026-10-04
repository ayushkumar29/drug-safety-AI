"""
Lineage Resolver and Independence Verification Module.

Provides deterministic lookup for publication lineages, attaches EvidenceLineage metadata
to ExtractedEvidence objects, evaluates study independence, and detects double-counting risks.
"""

import os
import json
from typing import Dict, Any, List, Optional
try:
    from analysis.study_lineage import StudyLineage, EvidenceLineage
except ImportError:
    from study_lineage import StudyLineage, EvidenceLineage


_LINEAGE_REGISTRY_CACHE: Optional[Dict[str, StudyLineage]] = None


def load_lineage_registry() -> Dict[str, StudyLineage]:
    """
    Loads and caches the publication lineage registry from data/metadata/study_lineage.json.
    """
    global _LINEAGE_REGISTRY_CACHE
    if _LINEAGE_REGISTRY_CACHE is not None:
        return _LINEAGE_REGISTRY_CACHE

    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "metadata", "study_lineage.json"),
        os.path.join(os.getcwd(), "data", "metadata", "study_lineage.json"),
        "data/metadata/study_lineage.json"
    ]

    reg_path = None
    for p in possible_paths:
        if os.path.exists(p):
            reg_path = p
            break

    registry: Dict[str, StudyLineage] = {}
    if reg_path and os.path.exists(reg_path):
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pub_map = data.get("publications", {})
            for pub_id, pub_dict in pub_map.items():
                registry[pub_id] = StudyLineage(**pub_dict)

    _LINEAGE_REGISTRY_CACHE = registry
    return _LINEAGE_REGISTRY_CACHE


def get_study_lineage(publication_id: str) -> StudyLineage:
    """
    Returns the registered StudyLineage for a publication_id.
    Returns a safe unknown default if the publication_id is not in the registry.
    """
    registry = load_lineage_registry()
    if publication_id in registry:
        return registry[publication_id]

    return StudyLineage(
        publication_id=publication_id,
        source_file=f"{publication_id}.pdf",
        publication_type="Unknown",
        underlying_study_ids=[],
        primary_study=False,
        overlap_group_id=None,
        independence_status="unknown",
        lineage_notes="Publication not found in study lineage registry."
    )


def attach_evidence_lineage(evidence: Any) -> EvidenceLineage:
    """
    Constructs and returns EvidenceLineage metadata for an ExtractedEvidence object,
    preserving all existing evidence and statistical fields intact.
    """
    pub_id = getattr(evidence, "study_id", None) or getattr(evidence, "source_file", "unknown").split(".")[0]
    lineage = get_study_lineage(pub_id)

    return EvidenceLineage(
        publication_id=lineage.publication_id,
        underlying_study_ids=lineage.underlying_study_ids,
        overlap_group_id=lineage.overlap_group_id,
        independence_status=lineage.independence_status
    )


def are_independent(lineage_a: StudyLineage, lineage_b: StudyLineage) -> Dict[str, Any]:
    """
    Evaluates whether two study lineages represent independent primary clinical study datasets.

    Rules:
    1. If both are independent_primary with non-empty, disjoint underlying_study_ids and different overlap_group_ids, return independent=True.
    2. If they share any underlying_study_id, return independent=False.
    3. If one is a pooled analysis, meta-analysis, or review that incorporates or overlaps with the other, return independent=False.
    4. If either status is 'unknown', return independent=False.
    """
    if lineage_a.publication_id == lineage_b.publication_id:
        return {
            "independent": False,
            "reason": f"Same publication ID '{lineage_a.publication_id}'.",
            "shared_underlying_studies": lineage_a.underlying_study_ids
        }

    if lineage_a.independence_status == "unknown" or lineage_b.independence_status == "unknown":
        return {
            "independent": False,
            "reason": "Unknown lineage status; safety fallback enforces non-independence.",
            "shared_underlying_studies": []
        }

    # Check for shared underlying study IDs
    shared_studies = list(set(lineage_a.underlying_study_ids).intersection(set(lineage_b.underlying_study_ids)))
    if shared_studies:
        return {
            "independent": False,
            "reason": f"Shared underlying study dataset(s): {shared_studies}.",
            "shared_underlying_studies": shared_studies
        }

    # Check if either publication is a non-primary pooled/meta/review study in the same overlap group
    if lineage_a.overlap_group_id and lineage_a.overlap_group_id == lineage_b.overlap_group_id:
        if not lineage_a.primary_study or not lineage_b.primary_study:
            return {
                "independent": False,
                "reason": f"Both publications belong to overlap group '{lineage_a.overlap_group_id}' and at least one is a secondary analysis ({lineage_a.publication_type} / {lineage_b.publication_type}).",
                "shared_underlying_studies": []
            }

    # Independent primary studies
    if lineage_a.primary_study and lineage_b.primary_study:
        return {
            "independent": True,
            "reason": "Both are distinct independent primary clinical trial publications.",
            "shared_underlying_studies": []
        }

    return {
        "independent": False,
        "reason": f"Non-independent publication pairing ({lineage_a.publication_type} vs {lineage_b.publication_type}).",
        "shared_underlying_studies": []
    }


def validate_study_independence(lineages: List[StudyLineage]) -> List[Dict[str, Any]]:
    """
    Scans a list of StudyLineage objects and identifies all non-independent, double-counting pairs.
    """
    overlapping_pairs = []
    n = len(lineages)
    for i in range(n):
        for j in range(i + 1, n):
            res = are_independent(lineages[i], lineages[j])
            if not res["independent"]:
                overlapping_pairs.append({
                    "pair": [lineages[i].publication_id, lineages[j].publication_id],
                    "reason": res["reason"],
                    "shared_underlying_studies": res["shared_underlying_studies"]
                })
    return overlapping_pairs
