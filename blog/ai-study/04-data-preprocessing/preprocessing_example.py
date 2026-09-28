import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

X_train = np.array([[10], [20], [np.nan], [30]], dtype=float)
X_test = np.array([[40], [np.nan]], dtype=float)

# 학습 데이터의 중앙값으로 빈칸 채우기
imputer = SimpleImputer(strategy="median")
train_filled = imputer.fit_transform(X_train)
test_filled = imputer.transform(X_test)

# 학습 데이터의 범위로 스케일링
scaler = MinMaxScaler()
train_ready = scaler.fit_transform(train_filled)
test_ready = scaler.transform(test_filled)

print("Train:", train_ready.ravel().tolist())
print("Test:", test_ready.ravel().tolist())
