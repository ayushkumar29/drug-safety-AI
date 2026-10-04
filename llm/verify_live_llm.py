"""
Verify Live OpenAI Structured Output Path
Checks for OPENAI_API_KEY, executes live API calls if present, and reports results cleanly.
Never prints or exposes the API key.
"""

import json
import os
from dotenv import load_dotenv
from hypothesis import parse_hypothesis, StructuredHypothesis, validate_hypothesis_dict, PROMPT_SYSTEM_INSTRUCTIONS

# Load .env if present
load_dotenv()


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.startswith("your_"):
        print("================================================================================")
        print("REAL LLM PATH TEST RESULT")
        print("================================================================================")
        print("Real LLM path cannot be tested because OPENAI_API_KEY is not configured.")
        print("Fallback parser was NOT used to fake a live LLM test.")
        print("================================================================================")
        return

    print("================================================================================")
    print("EXECUTING REAL OPENAI LLM STRUCTURED OUTPUT PATH TEST")
    print("================================================================================")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    test_questions = [
        {
            "id": "Test A",
            "question": "Does empagliflozin increase the risk of genital infection compared with placebo?",
            "expected_comparator": "placebo",
            "expected_direction": "greater_risk"
        },
        {
            "id": "Test B",
            "question": "Does empagliflozin reduce the risk of cardiovascular death?",
            "expected_comparator": "unknown",
            "expected_direction": "lower_risk"
        }
    ]

    all_passed = True

    for test in test_questions:
        t_id = test["id"]
        q = test["question"]
        print(f"\n[{t_id}]: \"{q}\"")
        print("-" * 80)

        # Call live OpenAI API directly with beta structured output parser
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": f"Parse the following research question into a structured hypothesis: '{q}'"}
            ],
            response_format=StructuredHypothesis,
        )

        parsed_model = completion.choices[0].message.parsed
        parsed_dict = parsed_model.model_dump()
        validated_dict = validate_hypothesis_dict(q, parsed_dict)

        print("Returned Pydantic StructuredHypothesis JSON:")
        print(json.dumps(validated_dict, indent=2))

        # Check required assertions
        comp = validated_dict.get("comparator")
        direction = validated_dict.get("hypothesis_direction")

        assert comp == test["expected_comparator"], f"{t_id} failed comparator: expected '{test['expected_comparator']}', got '{comp}'"
        assert direction == test["expected_direction"], f"{t_id} failed direction: expected '{test['expected_direction']}', got '{direction}'"

        print(f"  [OK] Comparator verified: '{comp}'")
        print(f"  [OK] Direction verified: '{direction}'")
        print(f"  [OK] {t_id} PASSES live LLM structured output test.")

    print("\n" + "=" * 80)
    print("LIVE LLM TEST SUMMARY")
    print("=" * 80)
    print("Real LLM path executed: YES")
    print("Model used: gpt-4o-mini")
    print("Structured-output mechanism: client.beta.chat.completions.parse (Pydantic schema)")
    print("Live tests passed: YES (Test A & Test B)")
    print("Fallback used: NO")
    print("================================================================================")


if __name__ == "__main__":
    main()
