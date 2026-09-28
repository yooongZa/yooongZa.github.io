import numpy as np
from sklearn.metrics import accuracy_score, log_loss

# 계산을 확인하려고 만든 예측값이다. 모델을 학습한 결과는 아니다.
y = np.array([0, 1, 0, 1])
probabilities = {
    "A": np.array([0.20, 0.45, 0.55, 0.45]),
    "B": np.array([0.40, 0.60, 0.40, 0.001]),
}

for name, p1 in probabilities.items():
    prediction = (p1 >= 0.5).astype(int)
    p_correct = np.where(y == 1, p1, 1 - p1)
    losses = -np.log(p_correct)
    print(f"{name}: Accuracy={accuracy_score(y, prediction):.0%}, "
          f"Log loss={log_loss(y, p1, labels=[0, 1]):.4f}")
    print("정답 클래스 확률:", np.round(p_correct, 3).tolist())
    print("샘플별 손실:", np.round(losses, 4).tolist())
    print(f"직접 계산한 평균 손실: {losses.mean():.4f}")
