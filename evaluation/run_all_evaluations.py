"""
Reproducibility Evaluation Runner (evaluation/run_all_evaluations.py)

Executes a reproducible evaluation sequence for all local benchmark artifacts and test suites.

Usage:
  python evaluation/run_all_evaluations.py          # Deterministic evaluation & test suite
  python evaluation/run_all_evaluations.py --live   # Re-run live end-to-end benchmark + summary & tests
"""

import sys
import os
import argparse
import subprocess

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def main():
    parser = argparse.ArgumentParser(description="Run reproducible drug safety AI evaluation suite.")
    parser.add_argument("--live", action="store_true", help="Re-run live end-to-end pipeline benchmark before building summary.")
    args = parser.parse_args()

    print("================ REPRODUCIBLE EVALUATION RUNNER ================\n")

    if args.live:
        print("[STEP 1/4] Running live end-to-end pipeline benchmark (run_end_to_end_evaluation.py)...")
        res1 = subprocess.run([sys.executable, "evaluation/run_end_to_end_evaluation.py"])
        if res1.returncode != 0:
            print("Error: Live evaluation benchmark failed.")
            sys.exit(res1.returncode)
    else:
        print("[STEP 1/4] Skipping live LLM pipeline execution (use --live for live benchmark re-execution).")

    print("[STEP 2/4] Building evaluation summary artifacts (build_evaluation_summary.py)...")
    res2 = subprocess.run([sys.executable, "evaluation/build_evaluation_summary.py"])
    if res2.returncode != 0:
        print("Error: Evaluation summary builder failed.")
        sys.exit(res2.returncode)

    print("[STEP 3/4] Running evaluation framework unit tests (test_end_to_end_evaluation.py & test_evaluation_summary.py)...")
    res3a = subprocess.run([sys.executable, "-m", "unittest", "evaluation/test_end_to_end_evaluation.py"])
    res3b = subprocess.run([sys.executable, "-m", "unittest", "evaluation/test_evaluation_summary.py"])
    if res3a.returncode != 0 or res3b.returncode != 0:
        print("Error: Evaluation unit tests failed.")
        sys.exit(1)

    print("[STEP 4/4] Running full project test suite (python -m unittest discover)...")
    res4 = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"])
    if res4.returncode != 0:
        print("Error: Project test suite failed.")
        sys.exit(res4.returncode)

    print("\n================ EVALUATION RUNNER COMPLETE ================")
    print("All evaluation steps and unit test suites passed successfully.")


if __name__ == "__main__":
    main()
