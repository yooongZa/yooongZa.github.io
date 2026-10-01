---
layout: post
title: "05. 선형회귀를 직접 구현하며 정리한 경사하강법"
date: 2026-09-28 14:03:00 +0900
permalink: /blog/ai-study/09-linear-regression/
description: "선형회귀의 예측과 MSE, 기울기와 가중치 갱신을 손으로 계산하고 NumPy 구현을 scikit-learn과 비교한 개인 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 12분 38초</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="05. 선형회귀를 직접 구현하며 정리한 경사하강법 복습 음성">
<source src="/blog/assets/audio/09-linear-regression.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/09-linear-regression.mp3">음성 파일 듣기</a>
</audio>
</div>

숫자를 예측하는 Linear regression(선형회귀)에서는 `fit()` 안에서 무슨 계산을 할까. 선형회귀의 학습 과정을 작은 계산으로 풀어봤다.

직선 하나를 정하고 예측과 정답의 차이를 계산한 뒤 그 차이를 줄이는 방향으로 직선을 조금씩 바꾼다. 특성이 여러 개면 행렬 곱으로 계산한다. 직접 구현한 결과는 scikit-learn과 비교했다.

## 1. 입력 X와 정답 y가 무엇인지부터 보기

중고 노트북의 사용 연수로 가격을 예측한다고 생각해보자. 사용 연수가 입력 `X`, 가격이 정답 `y`다. 한 행은 노트북 한 대에 해당하고, 각 행의 입력과 가격이 같은 대상을 가리켜야 한다.

이때 사용 연수는 Feature(특성), 가격은 Target(예측 대상)이다. 분류가 정해진 클래스 중 하나를 고른다면 회귀는 연속적인 수치를 예측한다.

학습 전에 Scatter plot(산점도)을 그리면 입력과 정답이 어떤 모양으로 연결돼 있는지 볼 수 있다. 사용 연수가 늘수록 가격이 대체로 낮아지는지, 특정 구간에서 휘는지, 유난히 떨어진 점이 있는지 확인한다.

Pearson correlation(피어슨 상관계수)은 선형적인 관련성을 -1부터 1 사이의 숫자로 요약한다. 직선의 기울기와는 다르다. 상관계수는 단위가 없지만 회귀계수는 입력과 출력의 단위에 영향을 받는다. 상관계수가 0에 가깝더라도 U자 모양 같은 비선형 관계가 있을 수 있고, 상관관계만으로 원인과 결과가 증명되는 것도 아니다.

아래 코드에서는 실제 중고 가격 자료 대신 관계를 직접 정한 합성 데이터를 사용한다. 작은 계산의 흐름을 확인하기 위한 예제다.

## 2. 직선에서 학습하는 것은 w와 b다

입력이 하나일 때 예측식은 다음과 같다.

```text
예측값 y_hat = w × x + b
```

`w`는 Weight(가중치)이자 직선의 기울기다. 입력이 1 늘어났을 때 예측값이 얼마나 변하는지 나타낸다. `b`는 Bias(편향)이자 절편으로, 입력이 0일 때의 예측값이다.

예를 들어 `y_hat = -10x + 100`이면 x가 1일 때 90, 2일 때 80을 예측한다. 이 식의 -10과 100을 데이터에 맞게 정하는 과정이 학습이다. 학습 중 입력 데이터 자체를 움직이는 것은 아니다.

이렇게 모델이 학습하는 값을 Parameter(매개변수)라고 한다. 뒤에서 정할 학습률과 반복 횟수는 학습 방법을 정하는 Hyperparameter(하이퍼파라미터)다. 둘을 구분해두면 무엇을 데이터로 구했고 무엇을 미리 정했는지 설명하기 쉽다.

특성이 여러 개면 입력별 가중치를 더한다.

```text
y_hat = w1 × x1 + w2 × x2 + ... + b
```

이 모델에서 한 회귀계수는 다른 특성들을 고정했을 때 해당 특성이 1 증가하면서 달라지는 예측값을 뜻한다. 현실에서 그 특성만 바꿨을 때 실제 결과가 그대로 바뀐다고 보장하는 인과 효과는 아니다.

## 3. 오차를 하나의 손실로 모으기

예측이 정답과 얼마나 다른지 먼저 샘플별로 계산한다. 이번 구현에서는 `error = prediction - y`로 둔다. 부호는 어느 쪽으로 빗나갔는지 알려준다.

오차를 그냥 더하면 +3과 -3이 서로 지워진다. Mean squared error(MSE, 평균제곱오차)는 오차를 제곱해서 평균낸다.

```text
MSE = 오차 제곱의 합 / 샘플 수
    = mean((prediction - y)²)
```

제곱하므로 큰 오차가 더 큰 영향을 준다. 이상치가 있거나 일부 샘플에서 크게 틀리면 평균 손실이 커질 수 있다. 손실 숫자와 실제 오답을 같이 살펴볼 이유다.

MSE의 단위는 정답 단위의 제곱이다. 가격 단위가 만 원이라면 MSE는 만 원의 제곱이어서 바로 가격 차이처럼 읽기 어렵다. Root mean squared error(RMSE, 평균제곱근오차)는 MSE에 제곱근을 취해 정답과 같은 단위로 돌린다. RMSE와 절대 오차의 평균인 MAE는 서로 다른 계산이다.

MSE를 쓴다고 모든 점을 정확히 통과하는 직선을 찾는 것은 아니다. 잡음이 섞인 자료에서는 전체 제곱오차가 작아지는 직선을 찾는다.

## 4. 손실의 기울기와 직선의 기울기를 구분하기

직선의 기울기 `w`와, 손실을 `w`로 미분한 값은 이름 때문에 헷갈리기 쉽다. 전자는 예측식의 계수이고 후자는 **현재 w를 조금 바꿨을 때 손실이 변하는 방향과 크기**다.

Derivative(미분)는 아주 작은 변화에 따른 함수값의 변화를 본다. 변수 하나를 움직일 때의 기울기가 양수라면 그 변수를 조금 줄이는 방향이 손실을 줄이는 쪽이다. 음수라면 조금 늘리는 방향을 생각할 수 있다. 여러 매개변수에 대한 이런 기울기를 모은 것이 Gradient(기울기 벡터)다.

Numerical differentiation(수치미분)은 값을 조금씩 바꿔 직접 함수값을 계산한다. 중앙차분은 다음처럼 쓴다.

```text
f'(w) ≈ [f(w + h) - f(w - h)] / (2h)
```

h가 작을수록 항상 더 정확한 것은 아니다. 너무 작으면 부동소수점 계산 오차의 영향을 받는다. 매개변수가 많으면 매번 양쪽의 손실을 계산하는 비용도 커진다. 이번에는 식으로 구한 기울기를 코드로 구현하고, 수치미분은 계산이 맞는지 검산하는 용도로 썼다.

샘플 수가 n이고 오차가 `w*x + b - y`이면 MSE의 기울기는 다음과 같다.

```text
dw = (2 / n) × sum(x × error)
db = (2 / n) × sum(error)
```

제곱을 미분하면 `2 × error`가 나오고, 그 안의 예측식을 w로 미분하면 x가 나온다. b로 미분하면 1이므로 db에는 x가 붙지 않는다. 합을 n으로 나누는 부분은 MSE의 평균에서 온다.

## 5. 가중치를 한 번만 손으로 바꿔보기

입력을 `[1, 2, 3]`, 정답을 `[3, 5, 7]`로 정한다. 정답 관계는 `y = 2x + 1`이다. w와 b를 모두 0에서 시작하면 세 예측은 전부 0이다.

```text
error = [0-3, 0-5, 0-7] = [-3, -5, -7]
MSE = (9 + 25 + 49) / 3 = 27.6667

dw = (2/3) × (1×(-3) + 2×(-5) + 3×(-7))
   = -22.6667
db = (2/3) × (-3 - 5 - 7) = -10
```

Gradient descent(경사하강법)는 기울기의 반대 방향으로 매개변수를 갱신한다. Learning rate(학습률)를 0.1로 정하면 아래처럼 된다.

```text
w_new = w - 학습률 × dw = 0 - 0.1×(-22.6667) ≈ 2.2667
b_new = b - 학습률 × db = 0 - 0.1×(-10) = 1.0
```

두 기울기는 갱신 전의 같은 w와 b에서 계산했다. w를 먼저 바꾸고, 바뀐 w로 db를 다시 구하면 같은 동시 갱신 계산이 아니다.

새 예측값은 약 `[3.2667, 5.5333, 7.8]`이다. 반올림 전 값으로 다시 계산한 MSE는 약 `0.3319`로 줄어든다. 정답의 기울기 2보다 조금 큰 2.2667까지 움직였지만 이전보다 손실은 작아졌다. 이 과정을 반복하면서 낮은 손실 쪽으로 간다.

## 6. 학습률과 반복 횟수는 함께 본다

학습률이 너무 작으면 매번 조금만 움직여서 학습이 오래 걸린다. 너무 크면 낮은 손실 지점을 지나쳐 진동하거나 손실이 커질 수 있다. 기울기가 가리키는 방향이 맞더라도 큰 걸음을 똑같이 적용하면 손실이 줄어든다는 보장은 없다.

선형회귀의 MSE는 매개변수에 대해 Convex(볼록)한 함수다. 다만 학습률, 데이터의 크기, 특성 간 관계에 따라 경사하강법이 얼마나 빨리 안정되는지는 달라진다. “볼록하니 아무 학습률로 해도 된다”는 뜻은 아니다.

Iteration(반복)은 보통 매개변수를 한 번 갱신하는 단위다. Epoch(에포크)는 학습 자료 전체를 한 번 훑는 단위다. 이번 코드는 매번 학습 데이터 전체를 쓰는 Full-batch(전체 배치) 방식이어서 한 번의 갱신이 한 에포크에 해당한다. Mini-batch(미니배치)로 나누면 한 에포크 안에 여러 번 갱신한다.

이번 예제는 학습률 0.05와 반복 2,000회를 미리 고정했다. 테스트 점수를 보며 후보를 고르는 실험은 하지 않았다. 실제로 값을 비교해 선택하려면 검증 세트나 교차 검증을 따로 둔다.

## 7. 특성이 늘어나면 배열 모양을 먼저 확인한다

샘플이 N개, 특성이 D개라면 X는 `(N, D)`다. 이번에는 특성 두 개를 사용한다.

| 값 | 형태 | 의미 |
|---|---|---|
| X | `(N, D)` | 행은 샘플, 열은 특성 |
| W | `(D,)` | 특성마다 하나씩 있는 계수 |
| b | 스칼라 | 모든 샘플에 더하는 절편 |
| y, prediction | `(N,)` | 샘플별 정답과 예측 |

NumPy의 `@`는 여기서 Matrix multiplication(행렬 곱)을 한다. 각 행의 특성과 가중치를 곱해 더하므로 `X @ W + b`의 결과는 `(N,)`이다.

```text
prediction = X @ W + b
error = prediction - y

dW = (2 / N) × X.T @ error
db = 2 × mean(error)
```

`X.T`는 `(D, N)`이다. 여기에 `(N,)`인 오차를 곱하면 각 특성의 기울기가 모인 `(D,)`가 나온다. 반복문으로 샘플을 하나씩 계산하던 식을 한 번의 배열 연산으로 표현한 것이다.

특히 y가 `(N, 1)`이고 예측이 `(N,)`이면 조심해야 한다. 빼기가 Broadcast(브로드캐스트)되면서 `(N, N)`으로 늘어날 수 있다. 에러 없이 계산됐다는 사실만으로 모양이 맞다고 판단하지 않고, 예측과 정답이 같은 형태인지 먼저 확인한다.

## 8. 스케일링은 분할한 뒤 학습 데이터에서 맞춘다

한 특성은 몇 년 단위이고 다른 특성은 수백만 단위라면 계수마다 손실의 변화 규모가 크게 다를 수 있다. 경사하강법에서는 이런 차이가 학습을 어렵게 만들 수 있다.

StandardScaler는 학습 데이터에서 각 열의 평균과 표준편차를 구한다. [01편](/blog/ai-study/04-data-preprocessing/)에서 정리한 순서대로 먼저 데이터를 나누고, 학습 데이터에만 `fit_transform()`을 쓴다. 테스트에는 학습 때 구한 기준으로 `transform()`만 적용한다.

```text
자료 분할
→ 학습 데이터로 평균·표준편차 계산
→ 같은 기준으로 학습·테스트 변환
→ 모델 학습
→ 테스트 예측과 평가
```

평가 자료까지 섞어 스케일 기준을 만들면 그 정보가 학습 과정에 들어간다. 전처리가 필요한 모델들을 비교할 때는 같은 분할과 같은 변환을 적용해야 비교 기준도 맞는다.

## 9. 두 특성으로 전체 학습 과정을 실행하기

설명용 관계를 `y = 5 + 3*x1 - 2*x2 + 작은 잡음`으로 정하고 80개 샘플을 만들었다. 데이터 생성과 분할의 난수 시드는 42다. 64개로 학습하고 16개로 평가한다.

가중치는 0으로 시작한다. 선형회귀의 이 예제에서는 이 초기화로 계산할 수 있다. 이 선택을 신경망의 모든 층을 0으로 초기화해도 된다는 규칙으로 옮겨가지는 않는다.

```python
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
```

실행 결과:

```text
Train / Test: (64, 2) (16, 2)
W / prediction: (2,) (16,)
학습 MSE (1회 갱신 후 -> 2000회): 50.2107 -> 0.0729
원래 단위의 계수: [2.978, -1.964]
원래 단위의 절편: 5.018
직접 구현 Test MSE: 0.0688
직접 구현 Test RMSE: 0.2623
직접 구현 Test R²: 0.9985
sklearn Test MSE: 0.0688
두 구현의 최대 예측 차이: 8.88e-15
```

학습 MSE는 첫 갱신 후 `50.2107`에서 마지막 `0.0729`로 줄었다. 테스트 MSE는 `0.0688`, RMSE는 `0.2623`이다. 이 값들은 서로 다른 세트에서 계산했으므로 테스트가 조금 더 낮다고 곧바로 이상한 결과는 아니다. 작게 나눈 합성 데이터의 잡음과 구성에 영향을 받는다.

R²(결정계수)는 이 테스트 정답의 평균만 예측하는 기준과 비교한 제곱오차 개선 정도다. 정답에 변동이 있는 일반적인 경우 `1 - 오차 제곱합 / 정답의 평균 주위 제곱합`으로 계산하며, 기준보다 못하면 음수가 될 수 있다. 여기의 `0.9985`를 분류 정확도 99.85%로 읽으면 안 된다.

처음부터 선형 관계와 작은 잡음으로 만든 자료이므로 높은 R²가 자연스럽다. 이 결과를 실제 가격 예측 성능으로 해석할 수는 없다.

## 10. scikit-learn과 비교하고 계수의 단위를 돌리기

직접 구현한 경사하강법과 `LinearRegression`에 같은 학습·테스트 분할과 같은 스케일링 결과를 줬다. 두 예측의 최대 차이는 실행 환경에서 약 `8.88e-15`였다. 이 자료에서는 반복 계산이 scikit-learn의 해와 거의 같은 예측에 도달한 것이다.

`LinearRegression`도 반드시 같은 경사하강법 반복문을 실행하는 것은 아니다. 일반적인 밀집 배열의 최소제곱 문제는 선형대수 풀이를 사용한다. 두 알고리즘의 내부 단계는 다르다. 이번에는 **같은 제곱오차 목적을 푼 결과가 일치하는지 확인했다**. [선형회귀와 최소제곱법](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

학습에 표준화한 X를 넣었으므로 W는 표준화된 입력에 대한 계수다. 원래 입력의 단위로 해석하려면 다시 변환해야 한다.

```text
W_original = W / scaler.scale_
b_original = b - scaler.mean_ @ W_original
```

표준화 식 `(X - 평균) / 표준편차`를 예측식에 대입해서 정리하면 위 관계가 나온다. 변환 후 계수는 약 `[2.978, -1.964]`, 절편은 `5.018`이다. 데이터를 만들 때 넣은 `[3, -2]`와 5에 가깝다. 잡음과 샘플 구성 때문에 정확히 일치할 필요는 없다.

표준화된 W와 원래 단위의 계수를 섞어 적으면 해석이 달라진다. 계수를 기록할 때는 어떤 입력 단위에서 구한 값인지 같이 남긴다.

## 11. 점수 다음에는 잔차와 적용 범위를 본다

Residual(잔차)은 보통 `정답 - 예측값`으로 정의한다. 이번 기울기 계산에서 쓴 `error = 예측값 - 정답`과 부호가 반대다. 제곱하면 같지만 그래프에서 어느 쪽으로 빗나갔는지 해석할 때는 정의를 확인한다.

예측값에 따라 잔차가 곡선 모양으로 남으면 선형식이 놓친 패턴이 있는지 본다. 값이 커질수록 잔차의 퍼짐이 넓어지는지, 일부 샘플이 큰 오차를 만드는지도 확인할 수 있다. 이번 코드에서는 MSE·RMSE·R²와 구현 일치를 확인했고, 실제 자료의 잔차 분석까지 진행한 것은 아니다.

특성끼리 매우 비슷하게 움직이는 Multicollinearity(다중공선성)가 강하면 계수가 불안정해질 수 있다. 예측이 비슷해도 각 특성에 배분된 계수는 달라질 수 있으므로 계수 크기만으로 중요도를 단정하지 않는다.

학습 때 본 범위 밖을 예측하는 Extrapolation(외삽)도 주의해서 읽는다. 사용 연수 1~5년 자료로 구한 직선을 30년까지 늘리는 계산은 가능하지만, 실제 가격 관계가 그대로 이어진다는 근거는 따로 필요하다.

직접 구현한 학습은 `예측 → 오차 → 손실 → 기울기 → 매개변수 갱신`을 반복했다. 배열 형태와 전처리 순서를 맞추고, 학습에 쓰지 않은 자료에서 결과를 확인한다. 이 계산을 따라가면 `fit()` 안에서 구하는 값도 조금 더 분명해진다.

---

AIFFEL 수업과 개인 복습 노트를 참고했다. 위 음성은 복습용 TTS(음성 합성)다. 음성 속 예제·강의 순서와 본문의 합성 데이터·블로그 회차는 다를 수 있다.

예제 실행 환경: Python 3.12.9, NumPy 2.5.2, scikit-learn 1.9.0. 2026년 9월 28일에 합성 데이터 학습, 손계산 값, 중앙차분 기울기, scikit-learn 예측과의 일치를 확인했다. 실제 중고 가격 자료를 내려받거나 학습한 결과는 아니다. 버전과 환경에 따라 마지막 자릿수는 달라질 수 있다.

<nav class="article-links" aria-label="관련 파일과 글 목록">
<p><a href="/blog/ai-study/08-classification-metrics/">← 04. 분류 모델 평가, 정밀도·재현율과 임계값</a></p>
<p><a href="/blog/ai-study/11-regularization-l1-l2/">06. 정칙화와 정규화, L1·L2가 헷갈렸던 이유 →</a></p>
<a href="/blog/">글 목록</a> · <a href="linear_regression_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
