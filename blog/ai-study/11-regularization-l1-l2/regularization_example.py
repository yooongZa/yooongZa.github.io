import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 관계를 직접 정한 합성 회귀 자료. 뒤의 두 특성은 정답 식에 넣지 않았다.
rng = np.random.RandomState(42)
X = rng.normal(size=(120, 4))
y = 3 + X @ np.array([4.0, -2.0, 0.0, 0.0]) + rng.normal(0, 0.4, 120)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

# alpha는 효과 관찰용 고정값이다. 두 모델에서 같은 강도를 뜻하지 않는다.
models = {
    "LinearRegression": LinearRegression(),
    "Lasso(alpha=0.15)": Lasso(alpha=0.15, max_iter=10000),
    "Ridge(alpha=10)": Ridge(alpha=10),
}
fitted = {}
print("Train / Test:", X_train.shape, X_test.shape)
for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    fitted[name] = pipeline
    mse = mean_squared_error(y_test, prediction)
    print(name)
    print("  표준화 입력의 계수:", np.round(model.coef_, 4).tolist())
    print("  정확히 0인 계수 수:", int(np.sum(model.coef_ == 0)))
    print(f"  MAE={mean_absolute_error(y_test, prediction):.4f} "
          f"MSE={mse:.4f} RMSE={np.sqrt(mse):.4f}")

# 스칼라 목적함수 0.5*(w-a)**2 + penalty의 해를 비교한다.
a = np.array([0.2, 3.0])
lam = 0.5
l1_solution = np.sign(a) * np.maximum(np.abs(a) - lam, 0)
l2_solution = a / (1 + 2 * lam)
print("스칼라 예제 a:", a.tolist())
print("L1 해:", l1_solution.tolist())
print("L2 해:", l2_solution.tolist())
