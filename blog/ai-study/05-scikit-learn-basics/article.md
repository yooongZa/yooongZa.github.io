준비한 데이터를 모델에 넣어 학습한 뒤 예측 결과를 정답과 비교한다. scikit-learn에서는 데이터 분할, 학습, 예측, 평가의 순서로 코드를 읽는다.

```text
문제 정하기 → X와 y 준비 → 데이터 분리
→ 모델 만들기 → fit → predict → 평가
```

회귀와 분류의 목표는 다르지만 코드의 큰 흐름은 비슷하다. 작은 회귀 예제와 Wine 분류 예제를 각각 실행하면서 확인했다.

## 1. 무엇을 예측할지 먼저 정하기

Supervised learning(지도학습)은 입력과 함께 정답이 주어지는 학습이다. 예측할 정답의 종류에 따라 회귀와 분류로 나뉜다.

- **Regression(회귀)**: 가격, 기온, 판매량처럼 수치를 예측한다.
- **Classification(분류)**: 스팸 여부, 상품 종류, 합격·불합격처럼 범주를 예측한다.

정답을 0, 1, 2로 저장했다고 해서 회귀가 되는 것은 아니다. 0, 1, 2가 와인의 종류를 뜻한다면 분류다. 숫자의 의미를 봐야 한다.

Unsupervised learning(비지도학습)은 정답 없이 데이터의 구조를 찾는다. 비슷한 고객을 묶는 Clustering(군집화), 많은 특성을 적은 축으로 줄이는 Dimensionality reduction(차원 축소)이 여기에 들어간다.

Reinforcement learning(강화학습)은 행동의 결과로 받은 보상을 이용해 누적 보상을 크게 만드는 행동 방식을 학습한다. 이번 글의 실습 범위는 지도학습의 회귀와 분류다.

수업에서 본 [알고리즘 선택 가이드](https://scikit-learn.org/stable/machine_learning_map.html)는 문제 유형과 데이터 크기 등을 보고 후보를 좁힐 때 참고할 수 있다. 처음에는 단순한 Baseline(기준 모델)을 만들고, 같은 조건에서 다른 모델과 비교하는 쪽으로 생각해두면 좋겠다.

## 2. scikit-learn에서 반복해서 쓰는 메서드

설치 패키지 이름은 `scikit-learn`, Python에서 불러오는 이름은 `sklearn`이다. 모델을 바꿔도 사용 순서가 비슷해서 기본 흐름을 익혀두면 다른 알고리즘을 시도하기 편하다.

scikit-learn에서 `fit()`으로 데이터를 학습하는 객체를 Estimator(추정기)라고 부른다. 예측 모델뿐 아니라 전처리 도구도 이 공통 방식을 따른다.

- `fit(X_train, y_train)`: 학습 데이터에서 예측 규칙을 배운다.
- `predict(X_test)`: 배운 규칙으로 새로운 입력의 답을 예측한다.
- `transform(X)`: 학습한 기준에 따라 데이터를 변환한다.
- `score(X, y)`: 해당 모델이 기본으로 정한 지표로 평가한다.

모든 추정기가 위 메서드를 전부 가진 것은 아니다. 예를 들어 `StandardScaler`는 `fit()`에서 평균과 표준편차를 구하고 `transform()`에서 값을 바꾼다. 분류 모델은 `predict()`로 클래스를 예측한다. [기본 사용법](https://scikit-learn.org/stable/getting_started.html)

`RandomForestClassifier(n_estimators=100)`처럼 모델을 만들 때 정하는 값은 Hyperparameter(하이퍼파라미터)다. 선형회귀의 계수처럼 `fit()` 과정에서 데이터로부터 구하는 값은 학습된 Parameter(파라미터)다. 두 시점을 구분해두면 코드를 읽기 쉽다.

## 3. X와 y의 모양부터 확인

X는 예측에 쓸 Feature matrix(특성 행렬), y는 정답을 담은 Target vector(타깃 벡터)다. 여기서 다루는 일반적인 표 데이터의 형태는 다음과 같다.

```text
X.shape = (표본 수, 특성 수)
y.shape = (표본 수,)
```

학생 다섯 명의 공부 시간과 수면 시간으로 점수를 예측한다면 X는 `(5, 2)`, y는 `(5,)`다. 한 행은 학생 한 명이고, 한 열은 공부 시간이나 수면 시간 같은 특성이다.

특성이 하나여도 X에는 열 방향이 필요하다. 숫자 20개가 들어 있는 `(20,)` 배열을 한 열로 바꾸면 `(20, 1)`이 된다. 아래 회귀 예제의 `reshape(-1, 1)`이 이 역할을 한다. `-1`은 전체 원소 수에 맞춰 행 수를 계산하라는 뜻이다.

반대로 특성 13개를 가진 새 표본 한 개라면 모양은 `(1, 13)`이어야 한다. 한 열짜리 표본 13개인 `(13, 1)`과 구분해야 한다.

X와 y는 길이뿐 아니라 행의 대응도 맞아야 한다. X의 첫 번째 학생에 y의 다른 학생 점수가 붙으면 학습할 관계가 바뀐다. 정렬하거나 행을 삭제할 때 두 데이터의 대응을 함께 확인한다.

## 4. 학습 데이터와 테스트 데이터 나누기

학습한 데이터로 다시 점수를 계산하면 학습 자료를 얼마나 잘 맞히는지 볼 수 있다. 새로운 데이터에 대한 Generalization(일반화) 성능을 보려면 따로 남겨둔 데이터가 필요하다.

`train_test_split()`에 X와 y를 함께 넣으면 대응하는 행을 같은 방식으로 나누고, 아래 순서로 네 값을 돌려준다.

```text
X_train, X_test, y_train, y_test
```

이번 예제의 설정은 이렇게 정했다.

- `test_size=0.2`: 전체의 약 20%를 테스트에 사용한다.
- `random_state=42`: 같은 데이터에서 분할 결과를 재현하도록 난수 시드를 고정한다.
- `stratify=y`: 분류에서 각 클래스의 비율이 학습과 테스트에 비슷하게 유지되도록 한다.

`stratify`는 뒤의 Wine 분류 예제에만 사용한다. 표본이 아주 적은 클래스가 있으면 나누기 어려울 수 있다. 회귀의 연속값 y에 그대로 적용하는 설정은 아니다. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

42라는 숫자가 성능을 좋게 만드는 것은 아니다. 모델을 바꿀 때 같은 시험 문제로 비교하려고 분할을 고정한다. 시간 데이터는 시간순 분리, 같은 고객의 반복 기록은 고객 단위 분리가 필요한지도 별도로 판단한다.

실제 모델 선택에는 Validation set(검증 세트)이나 Cross-validation(교차 검증)을 사용하고, 테스트 세트는 마지막 평가에 남긴다. 테스트 점수를 계속 보면서 설정을 고르면 테스트 정보도 모델 선택에 반영된다. 이번에는 흐름을 익히려고 정해둔 모델을 한 번씩 평가한다.

## 5. 선형회귀로 숫자 예측하기

Linear regression(선형회귀)은 입력과 정답의 관계를 계수와 절편으로 표현한다. 특성이 하나라면 익숙한 직선 식이다.

```text
예측값 = 계수 × 입력값 + 절편
```

아래 데이터는 `12 + 4 × 입력값`에 작은 난수를 더해 직접 만들었다. 실제 조사 자료와 구분해서 읽어야 한다. 학습을 마친 모델이 이 관계에 가까운 계수와 절편을 구하는지 확인한다.

```python
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
```

실행 결과:

```text
X / y: (20, 1) (20,)
Train / Test: (16, 1) (4, 1)
계수 / 절편: 4.047 11.466
MAE: 1.434
RMSE: 1.864
R²: 0.996
실제: [18.66, 84.27, 76.89, 21.43]
예측: [15.51, 84.3, 76.21, 19.56]
```

`coef_`는 학습된 계수, `intercept_`는 절편이다. 이번 실행에서는 약 `4.047 × 입력값 + 11.466`이라는 식을 얻었다. 데이터에 난수를 더했고 일부만 학습에 사용했기 때문에 처음 정한 4와 12에 정확히 일치하지는 않는다. [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

특성이 여러 개면 계수도 특성마다 생긴다. 계수의 절댓값을 곧바로 중요도 순위로 읽기 전에 단위와 스케일을 확인해야 한다. 같은 무게라도 kg로 쓸 때와 g으로 쓸 때 계수의 크기는 달라진다.

### 회귀 지표 읽기

MAE(평균 절대 오차)는 실제값과 예측값 차이의 절댓값을 평균낸다. RMSE(평균 제곱근 오차)는 오차를 제곱해 평균낸 뒤 제곱근을 구하므로 큰 오차에 더 민감하다. 둘 다 정답과 같은 단위로 읽을 수 있고, 같은 평가 데이터에서는 작을수록 오차가 작다.

R²(결정계수)는 평가 데이터의 정답 평균으로만 예측하는 기준과 비교한다. 정답에 변동이 있는 일반적인 경우, 완벽한 예측은 1이고 평균 기준과 같은 제곱오차면 0이다. 기준보다 못하면 음수도 나올 수 있다. 분류의 정확도와는 계산 의미가 다르다. [R² 설명](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

이번 R²는 0.996이지만, 처음부터 거의 직선 관계로 만든 데이터의 테스트 네 행에서 나온 값이다. 이 숫자로 실제 문제에서의 성능을 판단할 수는 없다. 예제에서는 입력·정답·예측과 오차가 어떻게 연결되는지 확인하는 데 집중한다.

## 6. datasets와 Bunch 살펴보기

연습할 데이터를 직접 만들지 않을 때는 `sklearn.datasets`의 작은 내장 데이터셋을 사용할 수 있다. 회귀용 `load_diabetes()`, 분류용 `load_wine()` 등이 있다. 이번 분류 실습은 별도 다운로드가 필요 없는 Wine 데이터를 사용한다.

`wine = load_wine()`처럼 불러오면 기본 반환값은 Bunch(번치) 객체다. 사전과 비슷한 묶음이며 `wine.data`와 `wine["data"]`처럼 접근할 수 있다.

- `data`: 입력 특성
- `target`: 정답 클래스
- `feature_names`: 각 특성의 이름
- `target_names`: 클래스 이름
- `DESCR`: 데이터셋 설명

Wine은 178개 표본, 13개 수치형 특성, 세 클래스로 구성된다. y의 값은 0·1·2이고 클래스별 개수는 59·71·48이다. y의 0·1·2는 와인 종류를 구분하는 레이블이다. 품질 점수나 좋고 나쁨의 순서를 뜻하지 않는다. [Wine 데이터 설명](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

데이터를 불러온 뒤에는 `DESCR`와 특성 이름을 읽고 X·y의 모양을 확인한다. 분류 문제라면 클래스별 개수도 함께 보는 습관을 들여야겠다.

## 7. Wine 데이터로 분류해보기

이번에는 Random forest(랜덤 포레스트)를 사용한다. 여러 Decision tree(결정 트리)의 예측을 종합하는 모델이다. `n_estimators=100`은 트리 100개를 사용한다는 설정이다. 모델 안에서도 무작위 선택이 있으므로 `random_state=42`를 정했다. [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Wine 데이터에는 결측치가 없고, 이 예제의 트리 모델은 일반적으로 스케일 변화에 덜 민감해서 스케일링 없이 먼저 실행한다. 스케일에 민감한 모델을 사용할 때는 전처리 과정을 함께 검토해야 한다.

```python
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
```

실행 결과:

```text
X / y: (178, 13) (178,)
클래스별 개수: [59, 71, 48]
Train / Test: (142, 13) (36, 13)
학습 정확도: 1.0
테스트 정확도: 1.0
기준 모델 정확도: 0.389
혼동행렬:
 [[12  0  0]
 [ 0 14  0]
 [ 0  0 10]]
```

178행 중 142행으로 학습하고, 남긴 36행으로 평가했다. `predict(X_test)`에는 테스트 입력만 들어간다. 정답인 `y_test`는 예측을 마친 뒤 점수 계산에 사용한다.

`DummyClassifier(strategy="most_frequent")`는 학습 데이터에서 가장 많은 클래스로만 답하는 기준 모델이다. 이번에는 클래스 1로만 예측해 테스트 36개 중 14개를 맞혔다. `14 / 36 ≈ 0.389`가 출력된 이유다.

랜덤 포레스트는 이번 분할의 36개를 모두 맞혔다. 이 결과는 여기서 정한 데이터·분할·모델의 실행 기록이다. 새로운 모든 와인을 100% 맞힌다는 의미로 확대해서 읽지 않는다. 점수를 보고 유리한 시드로 바꾸는 대신 정해둔 조건과 결과를 함께 남겨둔다.

## 8. 정확도와 혼동행렬 같이 보기

Accuracy(정확도)는 전체 예측 중 맞은 비율이다. 이해하기 쉽지만 클래스가 불균형하면 놓치는 부분이 생긴다.

100개 중 불량품이 5개라면 전부 정상이라고 답해도 정확도는 95%다. 그런데 불량품을 찾아내려는 목적은 달성하지 못한다. 그래서 정답 분포와 어떤 오분류를 했는지 함께 살펴본다.

Confusion matrix(혼동행렬)는 실제 클래스와 예측 클래스를 교차해서 센 표다. scikit-learn의 `confusion_matrix()`에서는 **행이 실제, 열이 예측**이다. 위 코드에서 순서를 `[0, 1, 2]`로 지정했으므로 왼쪽 위 12는 실제 0을 0으로 맞힌 개수다. 가운데 14와 오른쪽 아래 10도 각각 맞힌 개수이며, 대각선 밖에는 오분류가 들어간다. [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

이진 분류에서 Precision(정밀도)은 양성이라고 예측한 것 중 실제 양성의 비율, Recall(재현율)은 실제 양성 중 찾아낸 비율이다. F1은 정밀도와 재현율의 조화평균이다. 세 종류 이상인 문제에서도 클래스별로 계산하고, 평균을 어떤 방식으로 냈는지 확인한다.

`score()`도 이름만 보고 읽으면 헷갈린다. 이번 `RandomForestClassifier.score()`의 기본 지표는 정확도이고, `LinearRegression.score()`의 기본 지표는 R²다. 그래서 예제에서는 `accuracy_score()`나 `mean_squared_error()`처럼 지표 이름이 드러나는 함수를 함께 사용했다.

## 9. 전처리와 모델을 Pipeline으로 묶기

평균·중앙값·스케일링 기준은 학습 데이터에서 구해야 한다. 전체 데이터로 전처리를 먼저 끝낸 뒤 분리하면 테스트 정보가 이미 들어갈 수 있다.

단계가 늘어나면 Pipeline(파이프라인)으로 전처리와 모델을 묶을 수 있다. 예를 들어 스케일러와 분류 모델을 연결하면 내부 흐름은 이렇게 된다.

```text
pipeline.fit(X_train, y_train)
  → 스케일러: X_train에서 기준을 구하고 변환
  → 모델: 변환된 X_train과 y_train으로 학습

pipeline.predict(X_test)
  → 스케일러: 학습 때 구한 기준으로 X_test 변환
  → 모델: 변환된 값으로 예측
```

전처리 순서를 매번 따로 실행할 때 생기는 실수를 줄일 수 있다. 교차 검증에서도 전처리를 포함한 파이프라인 전체를 넣어야 각 분할의 학습 부분에서 기준을 구하게 된다. [전처리와 파이프라인](https://scikit-learn.org/stable/getting_started.html#pipelines-chaining-pre-processors-and-estimators)

파이프라인으로 묶어도 정답에서 계산한 열이 X에 들어 있거나 같은 사건이 학습과 테스트에 중복돼 있으면 Data leakage(데이터 누수)가 생길 수 있다. 입력에 어떤 정보가 들어가는지는 별도로 확인한다.

## 10. 오류가 나거나 점수가 이상할 때 확인할 것

처음에는 모델 종류보다 데이터 모양이나 순서 때문에 막히는 경우를 먼저 점검해두면 좋겠다.

- **2차원 배열이 필요하다는 오류**: X가 `(표본 수, 특성 수)`인지 확인한다. 특성 하나와 표본 하나를 구분한다.
- **표본 수가 다르다는 오류**: X의 행 수와 y의 길이를 비교하고, 정렬·삭제 후 같은 표본끼리 대응하는지 본다.
- **학습 전에 예측하는 오류**: 해당 모델 객체에 `fit()`을 했는지 확인한다. 중간에 모델을 새로 만들면 학습 상태도 새로 시작한다.
- **점수가 실행할 때마다 크게 변함**: 분할과 모델의 난수 설정, 데이터 크기를 확인한다.
- **학습 점수만 높음**: 테스트 점수와 비교해 Overfitting(과적합)을 살펴본다. 학습·테스트가 모두 낮다면 Underfitting(과소적합), 특성이나 정답의 문제도 검토한다.
- **너무 좋은 점수**: 학습과 평가에 같은 데이터를 썼는지, 정답 관련 열이 들어갔는지, 전체 데이터로 전처리했는지 점검한다. 100%라는 숫자만으로 누수라고 확정할 수는 없다.
- **새 범주 때문에 변환 실패**: 인코더가 처음 보는 범주를 어떻게 처리하는지 확인한다.

비교할 때는 분할과 지표를 고정한다. 여러 모델을 고르는 과정에서는 검증 데이터나 교차 검증을 쓰고, 마지막 테스트 결과와 구분해 적어둔다.

X와 y의 의미를 확인하고 데이터를 먼저 나눈다. `fit()`으로 학습한 뒤 `predict()` 결과를 정답과 비교한다. 점수가 높게 나왔을 때도 어떤 데이터를 대상으로 계산했는지 설명할 수 있어야겠다.

---

AIFFEL 수업을 공부하며 만든 개인 노트를 바탕으로 썼다. 회귀 데이터와 공개 예제 코드는 설명용으로 작성했고, Wine은 scikit-learn 내장 데이터다. 위 음성은 수업 복습용 TTS(음성 합성)이며 글의 코드와 실행 결과는 별도로 준비했다.

예제 실행 환경: Python 3.12.9, NumPy 2.5.2, scikit-learn 1.9.0. 2026년 9월 28일에 실행했으며, 버전과 실행 환경에 따라 결과가 조금 달라질 수 있다.
