전처리 수업에서 배운 내용을 간단히 정리해둔다. 이번에는 결측치를 채우고 숫자의 범위를 맞추는 과정까지.

## 전처리 순서

먼저 각 열이 무슨 뜻인지, 단위와 자료형은 맞는지 확인한다. 중복된 행도 살펴본다. 같은 내용이 두 번 수집된 건지, 실제로 반복된 기록인지 구분해야 한다.

그다음 학습용과 테스트용 데이터를 나눈다. **빈칸을 채울 값이나 스케일링 기준은 학습 데이터에서 구한다.** 전체 데이터로 기준을 잡으면 테스트 정보가 학습 과정에 섞이는 Data leakage(데이터 누수)가 생길 수 있다.

이번 숫자 예제의 순서는 이렇게 잡았다.

```text
데이터 확인 → 학습/테스트 분리 → 결측치 채우기 → 스케일링
```

이상치 처리나 범주형 데이터 변환은 데이터와 모델에 맞춰 추가하면 된다.

## 결측치와 이상치

Missing value(결측치)는 값이 비어 있는 상태다. 0도 의미 있는 값일 수 있으니 빈칸과 구분해야 한다. 상황에 따라 행을 지우거나 평균·중앙값 등으로 채운다. 아래 예제에서는 중앙값을 썼다.

Outlier(이상치)는 다른 값과 크게 떨어진 값이다. 입력 오류일 수도 있고 실제로 드문 사례일 수도 있다. 숫자가 크다는 이유만으로 지우기보다는 원인을 먼저 확인한다.

## 작은 예제

학습 데이터는 `10, 20, 빈칸, 30`, 테스트 데이터는 `40, 빈칸`으로 직접 만들었다. 각 행에 숫자 하나가 있는 배열이다.

```python
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
```

```text
Train: [0.0, 0.5, 0.5, 1.0]
Test: [1.5, 0.5]
```

학습 데이터의 중앙값은 20이다. 학습과 테스트의 빈칸을 모두 이 값으로 채운다.

MinMaxScaler는 기본 설정에서 학습 데이터의 최솟값을 0, 최댓값을 1로 맞춘다. 여기서는 최솟값 10, 최댓값 30이므로 `(값 - 10) / (30 - 10)`으로 계산된다.

테스트의 40은 `(40 - 10) / 20 = 1.5`가 된다. **새 데이터는 0~1 범위를 벗어날 수도 있다.**

## 기억할 것

- 학습 데이터에는 `fit_transform()`: 기준을 구하고 변환한다.
- 테스트 데이터에는 `transform()`: 학습에서 구한 기준을 그대로 쓴다.
- 전처리 후에는 행 수, 결측치, 자료형이 어떻게 바뀌었는지 확인한다.

이번에 기억해둘 건 이 두 메서드의 차이. 테스트 데이터가 들어왔다고 중앙값이나 최솟값·최댓값을 다시 구하지 않는다.

---

AIFFEL 수업을 바탕으로 정리한 개인 학습 기록. 숫자 예제는 따로 만들었다. 음성은 기존 복습용 TTS(음성 합성) 파일이며, 인코딩·구간화 등 글보다 넓은 내용을 다룬다.

참고: [데이터 누수](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
