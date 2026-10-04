"""
Test Hypothesis Parser Script
Executes test questions, validates expected fields, formats JSON output,
and tests controlled validation error handling for underspecified questions.
"""

import json
try:
    from llm.hypothesis import parse_hypothesis
except ImportError:
    from hypothesis import parse_hypothesis



TEST_CASES = [
    {
        "id": "TEST 1",
        "question": "Does empagliflozin increase the risk of genital infection compared with placebo?",
        "expected": {
            "drug": "empagliflozin",
            "comparator": "placebo",
            "outcome": "genital infection",
            "hypothesis_direction": "greater_risk"
        }
    },
    {
        "id": "TEST 2",
        "question": "Is empagliflozin associated with cardiovascular benefit compared with placebo?",
        "expected": {
            "drug": "empagliflozin",
            "comparator": "placebo",
            "outcome": "cardiovascular benefit",
            "hypothesis_direction": "association"
        }
    },
    {
        "id": "TEST 3",
        "question": "Does empagliflozin reduce the risk of cardiovascular death?",
        "expected": {
            "drug": "empagliflozin",
            "comparator": "unknown",  # Must NOT invent placebo
            "outcome": "cardiovascular death",
            "hypothesis_direction": "lower_risk"
        }
    },
    {
        "id": "TEST 4 (Underspecified Question)",
        "question": "Tell me something about empagliflozin.",
        "expected_behavior": "Controlled ValueError exception"
    }
]


def run_tests():
    print("=" * 80)
    print("TASK 7: NATURAL LANGUAGE HYPOTHESIS PARSER AUDIT & TEST")
    print("=" * 80)

    for case in TEST_CASES:
        test_id = case["id"]
        q = case["question"]

        print(f"\n[{test_id}]: \"{q}\"")
        print("-" * 80)

        if "expected" in case:
            parsed = parse_hypothesis(q)
            print("Parsed Hypothesis JSON:")
            print(json.dumps(parsed, indent=2))

            expected = case["expected"]
            for field, exp_val in expected.items():
                actual_val = parsed.get(field)
                assert actual_val == exp_val, f"Assertion Failed for {test_id} field '{field}': expected '{exp_val}', got '{actual_val}'"
                print(f"  [OK] Verified field '{field}': '{actual_val}'")

            print(f"Result: PASS [OK]")

        elif "expected_behavior" in case:
            try:
                parsed = parse_hypothesis(q)
                print("Parsed Hypothesis JSON (Unexpected success):")
                print(json.dumps(parsed, indent=2))
                assert False, "Test 4 Failed: Expected ValueError for underspecified question, but call succeeded."
            except ValueError as ve:
                print(f"  [OK] Successfully caught controlled validation error: {ve}")
                print(f"Result: PASS [OK] (Behaves as documented: Raises controlled ValueError when question cannot form a valid hypothesis)")

    print("\n" + "=" * 80)
    print("ALL HYPOTHESIS PARSER TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
