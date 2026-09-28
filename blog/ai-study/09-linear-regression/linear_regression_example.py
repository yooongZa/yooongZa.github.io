import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 설명용 합성 데이터: 특성 2개, 정답은 5 + 3*x1 - 2*x2 + 작은 잡음.
rng = np.random.RandomState(42)
X = rng.uniform(-3, 3, size=(80, 2))
y = 5 + X @ np.array([3.0, -2.0]) + rng.normal(0, 0.3, size=80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

W = np.zeros(X_train_s.shape[1], dtype=float)
b = 0.0
learning_rate = 0.05
steps = 2000
losses = []

for step in range(steps):
    prediction = X_train_s @ W + b
    error = prediction - y_train
    dW = (2 / len(y_train)) * (X_train_s.T @ error)
    db = 2 * error.mean()
    W -= learning_rate * dW
    b -= learning_rate * db
    losses.append(np.mean((X_train_s @ W + b - y_train) ** 2))

test_prediction = X_test_s @ W + b
reference = LinearRegression().fit(X_train_s, y_train)
reference_prediction = reference.predict(X_test_s)

# 표준화된 좌표의 계수를 원래 입력 단위로 돌린다.
W_original = W / scaler.scale_
b_original = b - scaler.mean_ @ W_original

print("Train / Test:", X_train_s.shape, X_test_s.shape)
print("W / prediction:", W.shape, test_prediction.shape)
print(f"학습 MSE (1회 갱신 후 -> {steps}회): {losses[0]:.4f} -> {losses[-1]:.4f}")
print("원래 단위의 계수:", np.round(W_original, 3).tolist())
print(f"원래 단위의 절편: {b_original:.3f}")
test_mse = mean_squared_error(y_test, test_prediction)
print(f"직접 구현 Test MSE: {test_mse:.4f}")
print(f"직접 구현 Test RMSE: {np.sqrt(test_mse):.4f}")
print(f"직접 구현 Test R²: {r2_score(y_test, test_prediction):.4f}")
print(f"sklearn Test MSE: {mean_squared_error(y_test, reference_prediction):.4f}")
print(f"두 구현의 최대 예측 차이: {np.max(np.abs(test_prediction - reference_prediction)):.2e}")
