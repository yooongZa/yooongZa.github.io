import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, auc, confusion_matrix,
    f1_score, fbeta_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

# 지표 계산용으로 만든 20개 정답과 점수. 1이 관심 클래스다.
# 실제 모델의 검증·테스트 성능을 나타내는 데이터는 아니다.
y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
scores = np.array([.96, .92, .87, .82, .78, .72, .68, .63, .58, .52,
                   .46, .41, .36, .32, .28, .23, .19, .14, .09, .03])

print(f"양성 비율: {y.mean():.2f}")
print(f"전부 0으로 예측한 정확도: {accuracy_score(y, np.zeros_like(y)):.3f}")
for threshold in [0.90, 0.65, 0.50]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"threshold={threshold:.2f}: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Accuracy={accuracy_score(y, pred):.3f} "
          f"Precision={precision_score(y, pred, zero_division=0):.3f} "
          f"Recall={recall_score(y, pred):.3f}")
    print(f"  F1={f1_score(y, pred):.3f} "
          f"F2={fbeta_score(y, pred, beta=2):.3f}")

# 곡선 전체를 평가할 때는 잘라낸 0/1 대신 연속 점수를 넣는다.
precision, recall, _ = precision_recall_curve(y, scores)
print(f"AP: {average_precision_score(y, scores):.4f}")
print(f"사다리꼴 PR AUC: {auc(recall, precision):.4f}")
print(f"ROC AUC: {roc_auc_score(y, scores):.4f}")
