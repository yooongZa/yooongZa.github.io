---
layout: post
title: "12. PyTorch와 Tensor, 숫자에서 학습까지"
date: 2026-09-28 14:39:25 +0900
permalink: /blog/ai-study/12-pytorch-tensor-basics/
description: "텐서의 모양·자료형·장치와 자동 미분, DataLoader부터 학습 반복문까지 연결한 개인 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 12분 09초</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="12. PyTorch와 Tensor, 숫자에서 학습까지 복습 음성">
<source src="/blog/assets/audio/12-pytorch-tensor-basics.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/12-pytorch-tensor-basics.mp3">음성 파일 듣기</a>
</audio>
</div>

scikit-learn에서는 `fit()`을 호출하면 학습 과정이 내부에서 진행됐다. PyTorch에서는 입력을 만들고, 예측과 손실을 구하고, 기울기로 가중치를 바꾸는 순서를 직접 적는다.

그 출발점이 Tensor(텐서)다. 이번에는 텐서의 모양과 연산부터 작은 분류 모델의 학습까지 연결해 정리했다. 복습 음성은 텐서 기초에 비중을 두고 있고, 본문에는 학습 반복문까지 이어지는 별도 예제를 넣었다.

## 1. 텐서를 볼 때는 값과 함께 세 가지를 본다

텐서는 여러 차원의 숫자를 담는 자료 구조다. 숫자 하나는 Scalar(스칼라), 한 줄은 Vector(벡터), 행과 열이 있으면 Matrix(행렬)로 볼 수 있고, 이미지 묶음처럼 축이 더 많은 데이터도 표현한다.

```python
import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

`shape`는 축마다 원소가 몇 개 있는지, `dtype`은 숫자의 자료형, `device`는 값이 저장되고 계산되는 장치다. 같은 숫자가 들어 있어도 이 셋이 맞지 않으면 원하는 계산을 할 수 없다.

`(2, 3)`은 2개의 샘플에 각각 3개의 특성이 있는 입력으로 해석할 수 있다. 축의 의미는 데이터가 정한다. 텐서 자체가 어느 축이 사람 수이고 어느 축이 키·몸무게인지 알고 있지는 않다.

이미지 묶음에서는 보통 `NCHW`를 쓴다. `N`은 Batch(배치) 크기, `C`는 채널 수, `H`와 `W`는 높이와 너비다. `(32, 1, 28, 28)`이면 한 채널의 28×28 이미지가 32장이라는 뜻이다.

초기화 방법도 용도를 구분한다. `zeros()`는 0, `ones()`는 1로 채우고, `randn()`은 표준정규분포에서 난수를 만든다. `arange()`는 일정 간격의 수열, `linspace()`는 양 끝을 포함해 지정한 개수의 값을 만든다. 난수 실습에는 seed를 남겨 결과를 다시 비교할 수 있게 한다.

## 2. 원소별 연산과 행렬 곱을 구분하기

`x * y`는 대응하는 원소끼리 곱한다. `x @ y` 또는 `torch.matmul(x, y)`는 행렬 곱 규칙을 따른다. 2차원끼리라면 `(N, D) @ (D, K)`의 결과가 `(N, K)`가 된다. 가운데 `D`가 맞아야 한다.

`sum()`이나 `mean()`에 주는 `dim`은 연산으로 줄일 축이다. 위의 `(2, 3)` 텐서에서 다음처럼 계산된다.

```text
mean(dim=0): 두 행을 평균 → [2.5, 3.5, 4.5]
sum(dim=1): 각 행의 세 값을 합 → [6, 15]
```

`dim=0`을 무조건 행별 연산이라고 외우면 혼란스럽다. **0번 축을 줄여서 무엇이 남는지** 생각하는 편이 낫다. `keepdim=True`를 붙이면 줄인 축을 크기 1로 남겨 다음 연산의 모양을 맞출 수 있다.

인덱싱과 슬라이싱도 자주 쓴다. `x[0]`은 첫 행, `x[:, 1]`은 모든 행의 두 번째 열이다. `x[x > 3]`처럼 조건으로 고른 값은 원래 행렬 모양을 그대로 유지하지 않을 수 있으므로 결과 shape도 확인한다.

Broadcasting(브로드캐스팅)은 서로 다른 모양의 텐서를 연산할 때 크기가 1인 축 등을 확장하는 규칙이다. 뒤쪽 축부터 비교해서 크기가 같거나 한쪽이 1이면 맞출 수 있다.

특히 회귀에서 예측이 `(N, 1)`, 정답이 `(N,)`이면 뺄셈 결과가 `(N, N)`으로 커질 수 있다. 오류 없이 계산됐다는 것만으로 올바른 손실은 아니다. 예측과 정답을 같은 모양으로 맞춘다.

## 3. reshape, permute, cat, stack

| 연산 | 하는 일 | 작은 예 |
|---|---|---|
| `reshape` | 원소 수를 유지하며 모양 변경 | `(2, 3)` → `(3, 2)` |
| `permute` | 축의 순서 변경 | `(N, H, W, C)` → `(N, C, H, W)` |
| `unsqueeze` | 크기 1인 축 추가 | `(3,)` → `(1, 3)` |
| `squeeze` | 크기 1인 축 제거 | `(2, 1, 3)` → `(2, 3)` |
| `cat` | 기존 축을 따라 연결 | 두 `(2, 3)`을 0번 축으로 → `(4, 3)` |
| `stack` | 새 축을 만들어 쌓기 | 두 `(2, 3)`을 쌓아 → `(2, 2, 3)` |

`reshape(3, 2)`와 `permute(1, 0)`은 결과 모양이 같아도 원소 배치가 다르다. 이미지의 채널 위치를 바꿀 때는 축의 의미를 옮기는 `permute()`를 쓴다. `reshape()`로 모양만 맞추면 픽셀과 채널이 엉뚱하게 섞일 수 있다.

`view()`는 기존 메모리 배치가 허용하는 모양으로만 바꿀 수 있다. 축을 바꾼 뒤에는 배치 조건 때문에 실패할 수 있다. `reshape()`는 가능한 경우 기존 저장 공간을 공유하고, 필요하면 복사한다. 따라서 reshape는 항상 복사한다거나 항상 공유한다고 단정하지 않는다.

`squeeze()`를 인자 없이 쓰면 크기 1인 축을 모두 없앤다. 배치가 한 개일 때 배치 축까지 사라질 수 있으니, 특정 축만 없애려면 `squeeze(dim=...)`으로 대상을 지정한다.

## 4. NumPy와 메모리를 공유할 수 있다

NumPy 배열을 텐서로 바꿀 때 `torch.from_numpy()`는 해당 배열과 저장 공간을 공유한다. 한쪽을 수정하면 다른 쪽 값도 바뀐다. `torch.tensor(array)`로 만들면 데이터를 복사한다.

이 차이는 변환 후 원본을 수정할 때 중요하다. 원본 배열을 보관해둔 줄 알았는데 전처리 과정에서 텐서까지 함께 바뀔 수 있기 때문이다. 아래 예제에서는 직접 만든 작은 배열만 수정해 차이를 확인한다.

`clone()`과 `detach()`도 역할이 다르다. `clone()`은 새 저장 공간으로 복사하지만 기울기 연결은 유지할 수 있다. `detach()`는 기울기 계산 연결을 끊지만 기존 텐서와 저장 공간을 공유한다. 기울기 연결과 저장 공간을 모두 분리하려면 `detach().clone()`을 쓴다.

학습 중인 텐서를 NumPy로 가져올 때는 보통 `tensor.detach().cpu().numpy()`처럼 쓴다. CPU에 있는 텐서와 결과 배열이 공유될 수 있으므로 독립적인 배열이 필요하면 마지막에 `.copy()`를 붙인다. 이런 변환은 모델 내부의 학습 경로와 기록·분석 용도를 구분해서 사용한다.

## 5. Autograd는 기울기를 계산한다

Autograd(자동 미분)는 연산의 연결을 따라 미분값을 구한다. `requires_grad=True`인 텐서로 계산한 손실에 `backward()`를 호출하면, 일반적인 학습 매개변수의 `.grad`에 기울기가 쌓인다.

예를 들어 `w=2`이고 손실이 `w²`이면 기울기는 `2w=4`다. `backward()`를 한 번 호출해도 `w`는 여전히 2다. **기울기 계산과 가중치 갱신은 별도 단계**다.

또 기울기는 기본적으로 누적된다. 새로 `w²`를 계산해 backward를 한 번 더 하면 `.grad`가 8이 된다. 이전 기울기를 남길 의도가 없다면 새 학습 단계 전에 비워야 한다. 이미 사용한 같은 계산 그래프를 그대로 두 번 역전파하는 문제와도 구분한다. 아래 코드는 매번 새로운 forward 계산을 한다.

이번에는 이어서 손실을 `(3w - 4)²`로 바꿨다. `w=2`에서 손실은 4이고, 미분값은 `2 × (3w - 4) × 3 = 12`다. 학습률 0.1로 한 번 갱신하면 `2 - 0.1 × 12 = 0.8`이 된다.

기본 연산과 이 미분 계산을 함께 실행하는 코드다.

```python
import numpy as np
import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("입력:", tuple(x.shape), x.dtype, x.device)
print("열별 평균:", x.mean(dim=0).tolist())
print("행별 합:", x.sum(dim=1).tolist())
print("행렬 곱:", (x @ torch.tensor([1., 0., -1.])).tolist())
print("축 순서 변경:", tuple(x.permute(1, 0).shape))
print("새 축으로 쌓기:", tuple(torch.stack([x, x]).shape))
print("기존 축으로 연결:", tuple(torch.cat([x, x], dim=0).shape))

# 직접 만든 배열로 공유와 복사의 차이만 관찰한다.
array = np.array([1., 2.], dtype=np.float32)
shared = torch.from_numpy(array)
copied = torch.tensor(array)
array[0] = 9
print("NumPy 수정 후 공유 / 복사:", shared.tolist(), copied.tolist())

w = torch.tensor(2.0, requires_grad=True)
(w ** 2).backward()
print("첫 backward의 w / grad:", w.item(), w.grad.item())
(w ** 2).backward()  # 새 forward를 계산해도 기존 grad에는 더해진다.
print("두 번째 backward의 grad:", w.grad.item())
w.grad = None
loss = (w * 3 - 4) ** 2
loss.backward()
print("새 손실 / 기울기:", loss.item(), w.grad.item())
with torch.no_grad():
    w -= 0.1 * w.grad
print("수동 갱신 후 w:", round(w.item(), 4))
```

```text
입력: (2, 3) torch.float32 cpu
열별 평균: [2.5, 3.5, 4.5]
행별 합: [6.0, 15.0]
행렬 곱: [-2.0, -2.0]
축 순서 변경: (3, 2)
새 축으로 쌓기: (2, 2, 3)
기존 축으로 연결: (4, 3)
NumPy 수정 후 공유 / 복사: [9.0, 2.0] [1.0, 2.0]
첫 backward의 w / grad: 2.0 4.0
두 번째 backward의 grad: 8.0
새 손실 / 기울기: 4.0 12.0
수동 갱신 후 w: 0.8
```

이 예제의 `w`처럼 직접 만든 Leaf Tensor(리프 텐서)의 grad와, 연산 도중 만들어진 중간 텐서의 grad는 보관 방식이 다를 수 있다. 중간 텐서의 `.grad`가 `None`이라고 해서 자동으로 학습 실패는 아니다. `zero_grad(set_to_none=True)` 직후에도 grad가 None인 것이 정상이다. 누적과 계산 그래프에 대한 동작은 [PyTorch 자동 미분 안내](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)를 참고했다.

## 6. 모델·손실·옵티마이저의 역할

PyTorch에서는 보통 `nn.Module`로 모델을 정의한다. `__init__()`에서 레이어를 등록하고 `forward()`에 입력이 지나가는 순서를 적는다. 기본 클래스의 초기화인 `super().__init__()`도 필요하다. 사용할 때는 `model(x)`로 호출한다.

간단히 순서대로 이어지는 모델은 `nn.Sequential`로 표현할 수 있다.

```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
```

두 특성을 받아 중간 값 8개로 바꾸고, 마지막에는 두 클래스에 대한 점수 두 개를 낸다. ReLU는 음수를 0으로 바꾸는 Activation Function(활성화 함수)다. 선형 층만 여러 개 이어 붙이면 전체가 다시 하나의 선형·아핀 변환으로 합쳐지므로, 이런 비선형 계산을 사이에 둔다.

Loss Function(손실 함수)은 출력과 정답의 차이를 계산한다. Optimizer(최적화 알고리즘)는 계산된 기울기로 매개변수를 바꾼다. `optimizer = Adam(model.parameters(), ...)`에서 전달하는 것은 모델의 학습 매개변수다.

손실에 맞는 입력 형식도 중요하다.

| 손실 함수 | 일반적인 출력 | 정답 |
|---|---|---|
| `CrossEntropyLoss` | 클래스별 원점수, `(N, C)` | 이 예제에서는 클래스 번호 `(N,)`, `long` |
| `BCEWithLogitsLoss` | 이진 분류 원점수 | 출력과 같은 모양의 실수형 0·1 값 |
| `MSELoss` | 회귀 예측값 | 예측과 같은 모양의 실수형 값 |

Logits(로짓)는 Softmax나 Sigmoid를 거치기 전의 점수다. 이번 CrossEntropyLoss에는 logits를 그대로 넣는다. 손실 계산용 출력에 Softmax를 미리 적용하지 않는다. 예측 클래스를 고르는 데는 `argmax(dim=1)`을 쓴다. CrossEntropyLoss는 확률 분포 형태의 정답을 받는 모드도 있지만, 여기서는 정수 클래스 번호를 사용하는 경우만 다룬다.

## 7. Dataset과 DataLoader로 배치 만들기

Dataset은 개별 샘플을 제공하고, DataLoader는 샘플을 배치로 묶어 순서대로 꺼내준다. 이번에는 이미 준비한 텐서를 `TensorDataset(X_train, y_train)`으로 묶었다.

학습 자료는 96개, `batch_size=16`이므로 한 번 전체를 보는 데 배치 6개가 필요하다. 이 전체 한 바퀴가 Epoch(에포크)다. 배치마다 한 번 가중치를 갱신한다면 100에포크는 600번의 Step(갱신 단계)이다.

`shuffle=True`는 학습 샘플의 순서를 섞는다. 입력과 정답을 한 Dataset에 같이 넣었으므로 서로 대응하는 관계가 유지된다. 입력과 정답을 따로 무작위로 섞으면 안 된다.

배치 크기가 나누어떨어지지 않는 데이터에서 에포크 평균 손실을 구할 때는 주의한다. 기본 평균 손실에 실제 배치의 샘플 수를 곱해 누적하고 전체 샘플 수로 나누면 된다. 작은 마지막 배치와 큰 배치를 같은 비중으로 평균내면 샘플 기준 평균과 달라진다.

## 8. 한 번의 학습은 이 순서로 진행된다

```text
이전 기울기 비우기 → 예측 → 손실 계산 → 역전파 → 가중치 갱신
```

`optimizer.zero_grad()`가 이전 기울기를 비우고, `model(xb)`가 Forward Pass(순전파)를 한다. `loss.backward()`는 Backward Pass(역전파)로 기울기를 계산한다. 실제 매개변수 변경은 `optimizer.step()`에서 일어난다.

손실 값만 화면에 기록할 때는 `loss.item()`으로 Python 숫자를 꺼낸다. 꺼낸 숫자는 기울기 계산의 연결을 가지지 않는다. 역전파에는 원래 loss 텐서를 사용한다.

학습에서는 `model.train()`, 검증·테스트에서는 `model.eval()`을 쓴다. 이 둘은 Dropout과 BatchNorm 같은 레이어의 행동을 바꾼다. **eval만 호출해도 기울기 기록이 자동으로 꺼지는 것은 아니다.**

반대로 `torch.no_grad()`는 기울기 기록을 끄지만 레이어를 eval 모드로 바꾸지 않는다. 순수 평가에서는 `model.eval()`과 `torch.inference_mode()`를 함께 사용할 수 있다. inference_mode에서 만든 텐서를 나중에 기울기 계산에 다시 쓰려는 경우에는 제약이 있으므로, 이번처럼 결과 확인만 하는 구간에 사용했다.

## 9. 작은 분류 모델을 끝까지 학습해보기

입력은 두 개의 실수다. `첫 번째 값 - 0.5 × 두 번째 값 > 0`이면 클래스 1, 아니면 0으로 정했다. 128개를 만들어 학습 96개와 테스트 32개로 나눴다.

이번 목적은 학습 흐름을 확인하는 것이다. 구조·학습률·에포크 수를 미리 정했고, 테스트 점수로 설정을 고르는 탐색은 하지 않았다. 실제 모델을 조정할 때는 검증 데이터를 따로 사용해야 한다.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(1)
# 작은 CPU 실습. 이 규칙으로 만든 데이터의 점수이며 실제 문제 성능은 아니다.
X = torch.randn(128, 2, dtype=torch.float32)
y = (X[:, 0] - 0.5 * X[:, 1] > 0).long()
order = torch.randperm(len(X))
train_idx, test_idx = order[:96], order[96:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(42),
)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():
    initial_loss = loss_fn(model(X_train), y_train).item()
for epoch in range(100):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.inference_mode():
    train_loss = loss_fn(model(X_train), y_train).item()
    test_logits = model(X_test)
    test_loss = loss_fn(test_logits, y_test).item()
    test_accuracy = (test_logits.argmax(dim=1) == y_test).float().mean().item()
print("Train / Test:", tuple(X_train.shape), tuple(X_test.shape))
print("입력 / 정답 dtype:", X.dtype, y.dtype)
print("에포크당 배치 / 전체 step:", len(loader), 100 * len(loader))
print("출력 logits:", tuple(test_logits.shape))
print(f"학습 Loss: {initial_loss:.4f} -> {train_loss:.4f}")
print(f"테스트 Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}")
```

```text
Train / Test: (96, 2) (32, 2)
입력 / 정답 dtype: torch.float32 torch.int64
에포크당 배치 / 전체 step: 6 600
출력 logits: (32, 2)
학습 Loss: 0.6875 -> 0.1230
테스트 Loss=0.1016, Accuracy=1.0000
```

학습 손실은 0.6875에서 0.1230으로 내려갔다. 테스트 32개에서는 모두 맞아 Accuracy가 1.0000으로 나왔다. 직접 정한 단순한 규칙의 작은 데이터에서 확인한 결과이므로 일반적인 분류 문제에서도 같은 성능을 낸다는 뜻은 아니다.

마지막 출력은 `(32, 2)`다. 32개 샘플 각각에 클래스 점수 두 개가 나온 것이다. 입력은 `float32`, 클래스 번호는 `int64`, 즉 `long`이었다. 무조건 모든 텐서를 같은 dtype으로 바꾸는 것보다 각 텐서의 역할에 맞추는 것이 중요하다.

## 10. 장치와 저장, 오류 확인까지

CPU, CUDA, MPS처럼 연산 장치를 바꿀 때는 모델과 입력이 같은 장치에 있어야 한다. 손실을 계산하는 정답도 맞춰준다. 텐서의 `x.to(device)`는 반환값을 `x = x.to(device)`처럼 받아 사용한다. 모델의 `model.to(device)`는 내부 매개변수와 버퍼를 이동한다.

이번 예제는 모두 CPU에서 실행했다. 작은 계산에서는 장치 이동 비용도 있어서 GPU가 항상 더 빠르다고 가정하지 않는다.

모델을 보관할 때 쓰는 `state_dict()`에는 학습 매개변수와 BatchNorm의 이동 통계 같은 등록된 버퍼가 들어간다. 다시 읽을 때는 같은 구조의 모델을 만든 뒤 `load_state_dict()`로 값을 넣는다. 학습을 이어가려면 옵티마이저 상태 등도 필요하다. 최적 시점의 상태를 메모리에 따로 보관하려면 단순 대입으로 참조만 남기는 것과 실제 복사를 구분한다. 이번 예제는 파일 저장·불러오기를 수행하지 않았다.

오류가 나면 다음 순서로 좁혀본다.

- 행렬 곱 오류: 입력 마지막 차원과 `Linear`의 `in_features`가 맞는지 본다.
- `float`·`double` 오류: NumPy에서 넘어온 입력 dtype과 모델 dtype을 비교한다.
- 분류 손실 오류: logits의 클래스 축, 정답의 shape·dtype·클래스 번호 범위를 본다.
- 장치 오류: 모델·입력·정답의 device를 확인한다.
- 예상과 다른 손실: 원치 않는 broadcasting이나 입력·정답 대응 오류를 확인한다.

텐서의 **shape·dtype·device를 먼저 보고**, 그다음 `예측 → 손실 → 기울기 → 갱신`이 이어지는지 확인하는 순서로 기억해둔다.

실행 환경은 Python 3.12.9, NumPy 2.5.2, scikit-learn 1.9.0, PyTorch 2.13.0 (CPU)이며 2026년 9월 28일에 두 예제를 실행했다. 강의 복습 노트를 바탕으로 텐서 연산과 CPU의 작은 학습 과정을 정리했다. 대규모 데이터 학습과 GPU 성능 비교는 수행하지 않았다.

<nav class="article-links" aria-label="관련 파일과 글 목록">
<p><a href="/blog/ai-study/11-regularization-l1-l2/">← 11. 정칙화와 정규화, L1·L2가 헷갈렸던 이유</a></p>
<p><a href="/blog/ai-study/14-dropout-batch-normalization/">14. Dropout과 Batch Normalization의 학습·평가 모드 →</a></p>
<a href="/blog/">글 목록</a> · <a href="pytorch_training_example.py" download>학습 예제</a> · <a href="article.md">Markdown</a> · <a href="tensor_basics_example.py" download>텐서 예제</a>
</nav>
