"""Original decision-only study example. No model or network calls."""


def next_step(passed, issues, previous, attempts, max_attempts):
    """Choose the next action after a completed attempt; max_attempts >= 1."""
    if passed:
        return "성공"
    if previous is not None and issues == previous:
        return "정체"
    if attempts >= max_attempts:
        return "한계"
    return "다시 시도"


if __name__ == "__main__":
    assert next_step(True, [], ["근거 누락"], 2, 4) == "성공"
    assert next_step(False, ["근거 누락"], ["근거 누락"], 2, 4) == "정체"
    assert next_step(False, ["조건 누락"], ["근거 누락"], 4, 4) == "한계"
    assert next_step(False, ["조건 누락"], ["근거 누락"], 2, 4) == "다시 시도"
    assert next_step(True, [], [], 4, 4) == "성공"
    assert next_step(False, ["근거 누락"], ["근거 누락"], 4, 4) == "정체"
    print(next_step(False, ["근거 누락"], ["근거 누락"], 2, 4))
    print("PASS: four actions and two overlapping stop conditions")
