import json
from pathlib import Path

from rag import answer_question

EVAL_FILE = Path("eval_set.json")


def contains_any(text: str, options: list[str]) -> bool:
    text = text.lower()
    return any(option.lower() in text for option in options)


def check(case: dict, result) -> list[str]:
    if result is None:
        return ["AI failed (the API would return 502)"]

    problems = []
    expected = case["expected_status"]
    allowed = expected if isinstance(expected, list) else [expected]
    if result.status not in allowed:
        problems.append(f"status was '{result.status}', expected one of {allowed}")

    expected_source = case.get("expected_source")
    if expected_source and expected_source not in [s.source for s in result.sources]:
        problems.append(f"expected source '{expected_source}' is missing")

    for options in case.get("must_include", []):
        if not contains_any(result.answer, options):
            problems.append(f"answer is missing one of {options}")

    for options in case.get("must_not_include", []):
        if contains_any(result.answer, options):
            problems.append(f"answer must not contain any of {options}")

    return problems


def main():
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    passed = 0

    for number, case in enumerate(cases, start=1):
        result = answer_question(case["question"])
        problems = check(case, result)
        reason = result.reason if result else "ai_failed"

        if problems:
            print(f"{number:>2}. FAIL  [{reason}]  {case['question']}")
            for problem in problems:
                print(f"       - {problem}")
            if result:
                print(f"       answer: {result.answer}")
        else:
            passed += 1
            print(f"{number:>2}. PASS  [{reason}]  {case['question']}")

    print(f"\nScore: {passed}/{len(cases)} ({passed / len(cases):.0%})")


if __name__ == "__main__":
    main()