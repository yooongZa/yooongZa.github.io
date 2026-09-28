import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 각 행은 표본 하나, 각 열은 특성 하나다.
X_train = np.array([[10], [20], [np.nan], [30]], dtype=float)
X_test = np.array([[40], [np.nan]], dtype=float)

preprocess = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler()),
])

train_ready = preprocess.fit_transform(X_train)
test_ready = preprocess.transform(X_test)

print("Train:", train_ready.ravel().tolist())
print("Test:", test_ready.ravel().tolist())
print("중앙값:", preprocess.named_steps["fill"].statistics_[0])
print("학습 최솟값·최댓값:",
      preprocess.named_steps["scale"].data_min_[0],
      preprocess.named_steps["scale"].data_max_[0])
