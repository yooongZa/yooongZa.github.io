---
layout: post
title: "08. Dropout과 Batch Normalization의 학습·평가 모드"
date: 2026-09-28 14:39:26 +0900
permalink: /blog/ai-study/14-dropout-batch-normalization/
description: "Dropout의 확률과 배율, BatchNorm의 감마·베타 및 이동 통계, train·eval 차이를 확인한 개인 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 12분 40초</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="08. Dropout과 Batch Normalization의 학습·평가 모드 복습 음성">
<source src="/blog/assets/audio/14-dropout-batch-normalization.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/14-dropout-batch-normalization.mp3">음성 파일 듣기</a>
</audio>
</div>

PyTorch 학습 코드에서 `model.train()`과 `model.eval()`을 봤다. 이름만 보면 학습을 시작하고 끝내는 명령처럼 보이지만, 실제로는 특정 레이어의 동작 방식을 바꾼다. 대표적인 것이 Dropout(드롭아웃)과 Batch Normalization(배치 정규화)이다.

이번에는 두 방법의 목적과 계산을 정리하고, 같은 입력이 학습 모드와 평가 모드에서 어떻게 달라지는지 확인했다. 위 음성에는 기존 강의 실습의 정확도 비교가 포함돼 있다. 아래에서는 작은 텐서의 동작을 확인하며, 그 강의의 모델 성능 실험을 다시 수행한 것은 아니다.

## 1. 두 방법을 쓰는 이유부터 구분하기

Dropout은 학습 중 일부 출력을 무작위로 0으로 만든다. 특정 특징이나 유닛에 지나치게 의존하는 경향을 줄여 일반화를 돕는 정칙화 방법이다.

Batch Normalization, 줄여서 BatchNorm은 신경망 중간 값의 크기를 배치 통계로 조정한다. 이후 학습 가능한 배율과 이동값을 적용한다. 최적화가 안정적으로 진행되는 데 도움이 될 수 있고, 배치 통계의 변동이 정칙화 효과를 주기도 한다.

| 항목 | Dropout | BatchNorm |
|---|---|---|
| 기본 동작 | 무작위로 일부 출력을 0으로 만듦 | 특성·채널별 통계로 값을 조정 |
| 학습되는 값 | 기본 Dropout 자체에는 없음 | 기본 설정에서 감마·베타 |
| 학습 중 저장되는 통계 | 없음 | 기본 설정에서 이동 평균·분산 |
| 평가 모드 | 입력을 그대로 통과 | 기본 설정에서 저장된 통계 사용 |

과적합을 볼 때는 학습 정확도 하나보다 학습·검증 손실의 흐름을 함께 확인한다. 이미 학습 데이터도 잘 맞추지 못하는데 Dropout을 강하게 넣으면 더 어려워질 수 있다. 두 방법 모두 넣으면 무조건 좋아지는 옵션으로 생각하지 않는다.

## 2. Dropout의 p는 버릴 확률

`nn.Dropout(p=0.5)`의 `p`는 각 원소를 0으로 만들 확률이다. 남길 확률은 `1-p`다. 한 번 실행할 때 전체의 정확히 절반을 골라 지운다는 뜻은 아니다. 원소가 적으면 0의 비율이 0.5와 꽤 다를 수 있다.

남은 값에는 학습 중 `1 / (1-p)`를 곱한다. `p=0.5`이면 남은 값이 두 배가 된다. 이를 Inverted Dropout(역 드롭아웃) 방식이라고 부른다.

```text
학습: 입력 x를 확률 p로 0으로 만듦
      남은 경우에는 x / (1-p)
평가: x를 그대로 통과
```

값이 1인 원소를 생각하면 절반의 확률로 0, 나머지 절반의 확률로 2가 나온다. 기대값은 `0 × 0.5 + 2 × 0.5 = 1`이다. 이 보정 덕분에 평가 시 별도로 반을 곱하지 않아도 된다. 한 번 뽑힌 작은 텐서의 평균까지 반드시 원본과 같아지는 것은 아니다. [PyTorch Dropout 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)에서 이 배율과 평가 동작을 확인했다.

여기서 사라지는 것은 이번 forward의 출력이다. 모델에서 뉴런이나 연결이 영구적으로 삭제되는 것은 아니다. 다음 학습 계산에서는 다른 마스크가 적용될 수 있다.

일반 `nn.Dropout`은 원소 단위로 동작한다. 이미지 특징 맵에서 채널 단위로 끄는 `Dropout2d`와는 구분한다. 어떤 단위의 정보를 없애고 있는지 보고 레이어를 선택해야 한다.

## 3. p를 크게 하면 정칙화도 무조건 좋을까

p가 커질수록 한 번의 계산에서 사용할 정보가 줄어든다. 과적합을 줄이는 데 도움이 될 수 있지만 학습 자체가 어려워지거나 과소적합이 생길 수도 있다. 데이터 크기, 모델 구조, 다른 정칙화의 유무에 따라 적절한 값이 달라진다.

Dropout을 켠 채 기록한 학습 정확도가 평가 정확도보다 낮게 나오는 경우도 있다. 학습 중에는 일부 출력을 제거하고, 평가에서는 전체를 사용하기 때문이다. 학습·평가 때 조건이 다르다는 점을 먼저 생각해야 한다.

또 학습 중 모은 정확도는 배치마다 가중치가 바뀌는 도중의 점수다. 마지막 가중치로 전체 학습 데이터를 다시 평가한 값과도 다를 수 있다. 두 값을 비교하려면 측정 시점과 모드를 맞추는 편이 낫다.

Dropout 효과를 비교하는 실험이라면 데이터 분할, 모델의 나머지 구조, 초기 가중치, 학습률, 에포크 수 같은 조건을 가능한 한 맞춘다. 여러 seed에서 결과가 안정적인지도 확인할 수 있다. 작은 검증 세트에서 나온 한 번의 차이는 우연에 크게 흔들릴 수 있다. 이번 글의 코드는 이런 성능 비교까지 수행하지 않는다.

## 4. BatchNorm은 평균·분산 다음에 감마·베타를 쓴다

한 특성에 대해 배치 평균을 `μ`, 분산을 `v`라고 하면 기본 계산은 다음과 같다.

```text
정규화한 값 = (x - μ) / sqrt(v + ε)
최종 출력   = γ × 정규화한 값 + β
```

Epsilon(엡실론) `ε`는 분모가 너무 작아지는 문제를 완화하는 작은 값이다. `γ`는 Gamma(감마), `β`는 Beta(베타)다. 감마와 베타는 역전파로 학습하는 매개변수다. 기본 초기값은 각각 1과 0이다.

평균과 분산으로 조정한 뒤 다시 크기와 위치를 바꾸는 이유는 모델이 필요한 표현을 학습할 수 있게 하기 위해서다. 따라서 최종 출력의 평균과 분산이 항상 0과 1로 고정되는 것은 아니다. 엡실론과 학습된 감마·베타도 영향을 준다. 정규분포 모양으로 바꾸는 변환이라는 뜻도 아니다.

입력 전처리의 StandardScaler와도 적용 위치가 다르다. 전처리는 입력 자료의 학습 통계를 구해 사용한다. BatchNorm은 네트워크 내부에서 현재 가중치가 만든 활성값에 적용되며, 학습 중 그 값들이 계속 바뀐다.

## 5. 어느 축을 묶어 통계를 구할까

`BatchNorm1d(2)`의 2는 샘플 수가 아니라 특성 또는 채널 수다. 입력 모양에 따라 평균·분산을 구하는 축을 구분한다.

| 입력 | 레이어 | 통계를 구할 때 묶는 축 |
|---|---|---|
| `(N, C)` | `BatchNorm1d(C)` | 각 C에 대해 N |
| `(N, C, L)` | `BatchNorm1d(C)` | 각 C에 대해 N과 L |
| `(N, C, H, W)` | `BatchNorm2d(C)` | 각 C에 대해 N·H·W |

예를 들어 `(4, 2)` 입력이면 특성 두 개에 대해 평균과 분산을 각각 하나씩 구한다. 감마와 베타도 각 특성에 하나씩 있어 총 네 개의 학습 매개변수가 생긴다.

BatchNorm2d는 이미지 한 장의 전체 색을 한꺼번에 평균내는 방식이 아니다. 채널별로 배치와 공간 위치를 모아 통계를 구한다. 채널 축을 잘못 넣으면 통계의 의미도 달라진다.

## 6. train과 eval에서 쓰는 통계가 다르다

기본값인 `track_running_stats=True`에서는 학습할 때 현재 배치의 평균·분산으로 출력값을 계산하고, 평가용 이동 통계도 갱신한다. 평가 모드에서는 저장해둔 `running_mean`과 `running_var`를 사용한다.

기본 `momentum=0.1`일 때 이동 통계의 갱신은 다음처럼 해석한다.

```text
새 이동 통계 = 0.9 × 기존 이동 통계 + 0.1 × 이번 배치 통계
```

여기서 momentum은 옵티마이저의 모멘텀과 다른 설정이다. 같은 이름이어도 무엇을 누적하는지 확인해야 한다.

PyTorch에서는 분산 계산에도 구분이 있다. 학습 출력의 정규화에는 `correction=0`인 분산을 사용하고, `running_var`의 갱신에는 `correction=1`인 분산을 사용한다. [BatchNorm1d 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)의 정의와 아래 실행 결과를 같이 확인했다.

`running_mean`과 `running_var`는 옵티마이저가 기울기로 학습하는 감마·베타와 다르다. Buffer(버퍼)로 등록돼 forward 과정에서 갱신되고, 모델의 `state_dict()`에는 함께 저장된다.

`track_running_stats=False`로 바꾸면 이 이동 통계를 유지하지 않으며 평가에서도 배치 통계를 사용한다. 그러므로 “BatchNorm은 eval에서 언제나 저장된 평균을 쓴다”는 설명에는 기본 설정이라는 조건이 붙는다.

## 7. 작은 텐서로 두 모드 비교하기

Dropout에는 1이 열두 개 들어 있는 텐서를 넣는다. BatchNorm에는 특성 두 개를 가진 샘플 네 개를 넣는다.

```text
첫 번째 특성: 2, 4, 6, 8   → 평균 5, 분산 5
두 번째 특성: 4, 8, 12, 16 → 평균 10, 분산 20
위 분산은 correction=0 기준
```

BatchNorm의 초기 이동 평균은 0, 이동 분산은 1이다. 이 상태에서 한 배치만 통과시키고 평가 모드로 바꿔본다. 충분히 학습된 이동 통계가 어떤 성능을 내는지 보는 실험과는 구분한다.

```python
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)
values = torch.ones(12)
dropout = nn.Dropout(p=0.5)
dropout.train()
with torch.no_grad():
    dropped = dropout(values)
dropout.eval()
with torch.inference_mode():
    passed = dropout(values)
print("Dropout train:", dropped.tolist())
print("Dropout eval:", passed.tolist())
print("Dropout 학습 매개변수 수:", sum(p.numel() for p in dropout.parameters()))

x = torch.tensor([[2., 4.], [4., 8.], [6., 12.], [8., 16.]])
bn = nn.BatchNorm1d(2)  # eps=1e-5, momentum=0.1, affine=True
bn.train()
with torch.no_grad():
    train_output = bn(x)  # no_grad 안에서도 train 모드의 통계는 갱신된다.
print("BatchNorm train 출력:\n", train_output)
print("배치 평균:", x.mean(dim=0))
print("배치 분산(correction=0):", x.var(dim=0, correction=0))
print("running_mean:", bn.running_mean)
print("running_var:", bn.running_var)
bn.eval()
with torch.inference_mode():
    eval_output = bn(x)
print("BatchNorm eval 첫 행:", eval_output[0])
print("학습 매개변수:", [name for name, _ in bn.named_parameters()])
print("eval 후 관찰한 배치 수:", bn.num_batches_tracked.item())
```

실행 결과를 확인해봤다. Dropout에서 어느 위치가 0이 되는지는 난수와 실행 환경의 영향을 받는다.

```text
Dropout train: [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
Dropout eval: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Dropout 학습 매개변수 수: 0
BatchNorm train 출력:
 tensor([[-1.3416, -1.3416],
        [-0.4472, -0.4472],
        [ 0.4472,  0.4472],
        [ 1.3416,  1.3416]])
배치 평균: tensor([ 5., 10.])
배치 분산(correction=0): tensor([ 5., 20.])
running_mean: tensor([0.5000, 1.0000])
running_var: tensor([1.5667, 3.5667])
BatchNorm eval 첫 행: tensor([1.1984, 1.5885])
학습 매개변수: ['weight', 'bias']
eval 후 관찰한 배치 수: 1
```

Dropout 학습 출력은 0 또는 2이고, 평가 출력은 모두 1이다. 이번에는 열두 개 중 세 개가 0이었다. p가 0.5라고 정확히 여섯 개가 지워질 필요는 없다는 것도 볼 수 있다. Dropout 자체의 학습 매개변수 수는 0이다.

BatchNorm 학습 출력의 첫 행은 두 특성 모두 약 -1.3416이다. 첫 번째 특성으로 계산하면 `(2 - 5) / sqrt(5 + 0.00001)`이다. 두 번째 특성은 값과 퍼짐이 함께 커져서 비슷한 정규화 결과가 나온다.

## 8. 이동 통계를 직접 계산해보기

배치 평균이 `[5, 10]`이므로 이동 평균은 다음과 같다.

```text
0.9 × [0, 0] + 0.1 × [5, 10] = [0.5, 1.0]
```

이동 분산에는 샘플 수 4에 대한 불편분산이 들어간다. 이 예제에서는 앞에서 구한 분산에 `4/3`을 곱한 값이다.

```text
이번 배치의 불편분산 = [6.6667, 26.6667]
새 running_var ≈ 0.9 × [1, 1] + 0.1 × [6.6667, 26.6667]
               ≈ [1.5667, 3.5667]
```

평가에서는 이 이동 통계를 사용한다. 그래서 첫 행 `[2, 4]`의 결과가 `[1.1984, 1.5885]`로 달라졌다. 같은 입력을 넣었어도 학습과 평가가 사용하는 평균·분산이 다르기 때문이다.

한 배치로 이동 통계를 한 번 갱신했을 뿐이라 원래 배치의 평균·분산과 차이가 크다. 이 숫자를 보고 BatchNorm이 예측을 개선했다거나 악화했다고 판단할 수는 없다. 여기서는 모드별 계산만 확인했다.

`named_parameters()`에는 `weight`, `bias`가 나온다. 각각 감마와 베타에 해당한다. 평가 후에도 `num_batches_tracked`는 1이다. 평가 입력으로 이동 통계를 다시 갱신하지 않았다는 것도 확인했다.

## 9. eval과 no_grad는 함께 구분해두기

예제에서 학습 모드의 forward를 `torch.no_grad()` 안에서 실행했다. 그래도 Dropout은 출력을 지웠고 BatchNorm의 이동 통계는 바뀌었다.

`no_grad()`는 기울기 기록을 끄는 기능이다. 레이어의 학습·평가 동작을 선택하는 것은 `train()`과 `eval()`이다. 또 `train()`을 호출했다고 가중치가 저절로 바뀌는 것도 아니다. 일반적인 가중치 갱신에는 손실 계산, backward, optimizer step이 필요하다.

| 설정 | 기울기 기록 | 드롭아웃 | 기본 배치 정규화 |
|---|---|---|---|
| `train()`만 | 조건에 따라 기록 | 활성 | 배치 통계 사용·이동 통계 갱신 |
| `train()` + `no_grad()` | 끔 | 활성 | 배치 통계 사용·이동 통계 갱신 |
| `eval()`만 | 조건에 따라 기록 | 통과 | 이동 통계 사용 |
| `eval()` + `inference_mode()` | 끔 | 통과 | 이동 통계 사용 |

검증을 마친 뒤 다시 학습할 때는 `model.train()`을 호출하는 것도 잊지 않는다. 반대로 예측 전에 eval 전환을 빠뜨리면 같은 입력의 출력이 Dropout 때문에 달라지거나, 평가 데이터로 BatchNorm 통계가 바뀔 수 있다.

## 10. 배치 크기와 레이어 순서

BatchNorm은 통계를 구할 값이 충분해야 한다. `(1, C)` 입력의 BatchNorm1d를 학습 모드로 실행하면 채널마다 값이 하나뿐이라 오류가 날 수 있다. 다만 배치가 1이라도 `(1, C, L)`이나 `(1, C, H, W)`에서 공간·길이 축에 값이 여러 개 있으면 상황이 다르다. 통계에 실제로 몇 개의 값이 들어가는지 봐야 한다.

작은 마지막 배치만 문제인 경우 `drop_last=True`를 고려할 수 있지만, 그 샘플들이 해당 에포크에서 빠진다는 점을 알아야 한다. 배치 크기를 늘리기 어렵다면 구조에 따라 LayerNorm이나 GroupNorm 같은 다른 정규화 방법을 검토할 수 있다. 기울기를 여러 번 누적해 큰 배치를 흉내 내더라도 BatchNorm의 통계는 각 forward 배치에서 계산된다.

레이어 순서로는 `Linear 또는 Conv → BatchNorm → 활성화 함수`가 흔히 쓰인다. Dropout까지 쓴다면 활성화 뒤에 배치하는 구성을 생각할 수 있다. 강의 예제처럼 활성화 뒤에 BatchNorm이 오는 구성도 있으므로 모든 모델의 정답 순서로 외우지는 않는다. Dropout을 BatchNorm 앞에 놓으면 지워지고 확대된 값으로 통계를 계산하게 된다는 점을 고려한다.

BatchNorm이 들어가면 학습이 안정돼 목표 성능에 더 적은 갱신으로 도달할 수 있다. 하지만 층의 계산 자체는 추가되므로 실제 시간이 언제나 짧아진다는 뜻은 아니다. 초기 설명에 나오는 Internal Covariate Shift(내부 공변량 변화)만으로 효과의 모든 이유가 확정됐다고 받아들이기보다, 적용한 모델의 손실과 평가 결과를 함께 확인한다.

이번에 기억할 것은 `p는 버릴 확률`, `감마·베타는 학습값`, `이동 통계는 버퍼`, 그리고 `eval과 기울기 기록은 별개`라는 네 가지다.

실행 환경은 Python 3.12.9, NumPy 2.5.2, scikit-learn 1.9.0, PyTorch 2.13.0 (CPU)이며 2026년 9월 28일에 작은 CPU 예제를 실행했다. 강의 복습 노트를 바탕으로 정리했고, Dropout·BatchNorm의 계산과 모드 차이를 확인했다. 이미지 분류 모델의 재학습이나 정확도 비교는 수행하지 않았다.

<nav class="article-links" aria-label="관련 파일과 글 목록">
<p><a href="/blog/ai-study/12-pytorch-tensor-basics/">← 07. PyTorch와 Tensor, 숫자에서 학습까지</a></p>
<p><a href="/blog/ai-study/15-linear-and-convolution-layers/">09. Linear에서 CNN까지, 레이어가 바꾸는 것 →</a></p>
<a href="/blog/">글 목록</a> · <a href="dropout_batchnorm_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
