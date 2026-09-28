"""가상 벡터로 WEAT 계산을 확인한다. 실제 단어 편향을 측정한 결과가 아니다."""
from itertools import combinations
import numpy as np


def unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('0벡터의 cosine은 정의되지 않습니다.')
    return vectors / norms


def association(words, a, b):
    words, a, b = map(unit_rows, (words, a, b))
    return (words @ a.T).mean(axis=1) - (words @ b.T).mean(axis=1)


def weat(x, y, a, b):
    sx, sy = association(x, a, b), association(y, a, b)
    scores = np.concatenate([sx, sy])
    spread = scores.std(ddof=0)
    if spread == 0:
        raise ValueError('모든 연관 점수가 같아 효과 크기를 계산할 수 없습니다.')
    effect = (sx.mean() - sy.mean()) / spread
    observed = sx.sum() - sy.sum()
    # 같은 크기의 두 target 집합으로 나누는 모든 경우를 계산한다.
    statistics = []
    for indices in combinations(range(len(scores)), len(sx)):
        left = np.zeros(len(scores), dtype=bool)
        left[list(indices)] = True
        statistics.append(scores[left].sum() - scores[~left].sum())
    # 단측 검정, 관측값과 같은 경우도 센다.
    p = np.mean(np.asarray(statistics) >= observed)
    return sx, sy, effect, observed, p, len(statistics)


if __name__ == '__main__':
    # X=가상의 꽃 3개, Y=가상의 곤충 3개.
    # A=가상의 유쾌함, B=가상의 불쾌함. 방향을 의도적으로 정했다.
    x = [[1, 0], [1, 1], [1, 2]]
    y = [[-1, 0], [-1, 1], [-1, 2]]
    a, b = [[1, 0], [1, 1]], [[-1, 0], [-1, 1]]
    sx, sy, effect, statistic, p, count = weat(x, y, a, b)
    print('X 연관:', np.round(sx, 4).tolist())
    print('Y 연관:', np.round(sy, 4).tolist())
    print('효과 크기 d:', round(float(effect), 4))
    print('검정 통계량 S:', round(float(statistic), 4))
    print('전체 분할 수:', count)
    print('단측 p (동점 포함):', round(float(p), 4))
    print('X/Y 교환 d:', round(float(weat(y, x, a, b)[2]), 4))
