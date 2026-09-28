import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 설명용으로 만든 숫자다. 실제 조사 데이터는 아니다.
rng = np.random.RandomState(10)
X = np.arange(1, 21, dtype=float).reshape(-1, 1)
y = 12 + 4 * X.ravel() + rng.normal(0, 2, size=20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("X / y:", X.shape, y.shape)
print("Train / Test:", X_train.shape, X_test.shape)
print("계수 / 절편:", round(model.coef_[0], 3), round(model.intercept_, 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print("R²:", round(r2_score(y_test, y_pred), 3))
print("실제:", np.round(y_test, 2).tolist())
print("예측:", np.round(y_pred, 2).tolist())
