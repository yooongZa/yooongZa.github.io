import numpy as np
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

wine = load_wine()
X, y = wine.data, wine.target
print("X / y:", X.shape, y.shape)
print("클래스별 개수:", np.bincount(y).tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 가장 많은 클래스로만 답하는 기준 모델과 비교한다.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Train / Test:", X_train.shape, X_test.shape)
print("학습 정확도:", round(model.score(X_train, y_train), 3))
print("테스트 정확도:", round(accuracy_score(y_test, y_pred), 3))
print("기준 모델 정확도:", round(accuracy_score(y_test, baseline_pred), 3))
print("혼동행렬:\n", confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
