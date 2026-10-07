"""Standalone synthetic stock-validation example; no model calls."""
def checked_total(rows, expected_centers):
    seen = set()
    total = 0
    for row in rows:
        center = row["center"]
        value = row["stock"]
        if center in seen:
            raise ValueError("센터가 중복되었습니다")
        if type(value) is not int or value < 0:
            raise ValueError("재고는 음이 아닌 정수여야 합니다")
        seen.add(center)
        total += value
    if seen != set(expected_centers):
        raise ValueError("센터가 누락되었거나 추가되었습니다")
    return total


rows = [
    {"center": "봄", "stock": 145},
    {"center": "가을", "stock": 80},
]
print(checked_total(rows, {"봄", "가을"}))  # 225
