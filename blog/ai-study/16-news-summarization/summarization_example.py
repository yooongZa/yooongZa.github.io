"""TextRank의 그래프 순위 아이디어를 TF-IDF 유사도로 확인하는 자체 예제."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SENTENCES = [
    "시립 도서관은 다음 달부터 주말 운영 시간을 늘린다.",
    "도서관은 주말 이용자가 늘어 운영 시간을 두 시간 연장하기로 했다.",
    "연장 운영은 토요일과 일요일에 적용된다.",
    "도서관은 주말 운영 연장에 맞춰 안내 인력을 추가로 배치한다.",
    "인근 공원에는 봄꽃이 피었다.",
]


def summarize(sentences, count=2):
    # 조사 분리 없이 공백 단위에 가까운 토큰을 사용하는 학습용 예제다.
    vectors = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(sentences)
    similarities = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarities, 0.0)
    n = len(sentences)
    totals = similarities.sum(axis=1, keepdims=True)
    transitions = np.divide(
        similarities, totals, out=np.full((n, n), 1 / n), where=totals != 0
    )
    rank = np.full(n, 1 / n)
    for _ in range(100):
        updated = 0.15 / n + 0.85 * transitions.T @ rank
        if np.abs(updated - rank).sum() < 1e-10:
            rank = updated
            break
        rank = updated
    selected = sorted(np.argsort(-rank, kind="stable")[:count].tolist())
    return rank, selected, [sentences[i] for i in selected]


def main():
    rank, selected, summary = summarize(SENTENCES)
    print("scores:", [round(float(x), 4) for x in rank])
    print("selected sentence numbers:", [i + 1 for i in selected])
    print("summary:")
    for sentence in summary:
        print(sentence)


if __name__ == "__main__":
    main()
