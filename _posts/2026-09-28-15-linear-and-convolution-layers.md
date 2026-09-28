---
layout: post
title: "09. Linear에서 CNN까지, 레이어가 바꾸는 것"
date: 2026-09-28 14:39:27 +0900
permalink: /blog/ai-study/15-linear-and-convolution-layers/
description: "Linear와 합성곱의 손계산, stride·padding·pooling과 출력 모양 및 매개변수 수를 정리한 개인 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 12분 33초</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="09. Linear에서 CNN까지, 레이어가 바꾸는 것 복습 음성">
<source src="/blog/assets/audio/15-linear-and-convolution-layers.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/15-linear-and-convolution-layers.mp3">음성 파일 듣기</a>
</audio>
</div>

모델 구조를 보면 `Linear`, `Conv2d`, `ReLU`, `MaxPool2d` 같은 이름이 이어진다. 이번에는 각 레이어에서 무슨 계산을 하고, 입력과 출력의 모양이 어떻게 바뀌는지 정리한다. 작은 숫자로 손계산을 해보고 마지막에 짧은 CNN을 통과시켜봤다.

위 음성은 **「딥러닝 레이어의 이해 1과 2」를 합친 최신 생성본**이다. 뒤쪽에는 VGG와 ResNet 이야기도 포함돼 있다. 이번 본문은 Linear·Convolution·Pooling의 기초까지 다룬다.

## 1. 레이어는 입력을 받아 다른 표현으로 바꾼다

Layer(층)는 입력에 정해진 계산을 적용해 출력을 만드는 단위다. Neural Network(신경망)는 여러 층을 연결해 입력에서 최종 예측까지 계산한다.

학습되는 Parameter(매개변수)가 있는 층도 있고 없는 층도 있다. Linear와 Convolution은 가중치를 학습한다. 기본 ReLU나 MaxPool은 학습할 가중치 없이 정해진 규칙으로 값을 바꾼다. 레이어 수와 학습 매개변수 수는 같은 개념이 아니다.

이미지를 분류하는 모델을 예로 들면 입력은 픽셀 값이고 마지막 출력은 클래스별 점수일 수 있다. 중간에서는 입력을 여러 방식으로 변환한 표현이 만들어진다. 어떤 표현이 유용한지는 손실을 줄이는 학습 과정에서 결정된다.

CNN을 만들었다고 처음부터 경계나 사물의 모양을 알아보는 것은 아니다. 무작위 초기 가중치에서 시작한다면 그에 따른 계산 결과가 먼저 나온다. 학습 후 어떤 특징이 나타나는지는 데이터와 모델을 보고 해석해야 한다.

## 2. Linear는 입력에 가중치를 곱하고 편향을 더한다

Fully Connected Layer(완전 연결층), Dense Layer(밀집층)라는 표현을 자주 본다. PyTorch에서는 `nn.Linear`로 이 계산을 한다. 편향이 포함되면 정확히는 Affine Transformation(아핀 변환)이다.

샘플 한 개의 입력이 `[x1, x2, x3]`이고 출력이 두 개라면, 출력마다 세 입력을 조합하는 가중치가 필요하다.

```text
y1 = w11×x1 + w12×x2 + w13×x3 + b1
y2 = w21×x1 + w22×x2 + w23×x3 + b2
```

PyTorch의 `Linear(3, 2)`는 입력 특성 3개를 출력 특성 2개로 바꾼다. weight의 모양은 `(2, 3)`, bias는 `(2,)`다. 샘플을 행으로 둔 입력 `X`에 대해서는 `X @ W.T + b`로 계산한다.

```text
입력 X: (N, 3)
가중치 W: (2, 3)
출력: (N, 2)
매개변수 수: 3×2 + 2 = 8
```

기본 설정에서는 bias가 있지만 `bias=False`로 만들면 편향이 빠진다. 이때 매개변수 수도 달라진다.

Linear는 입력의 **마지막 축**에 작용한다. `(N, T, 3)`을 넣으면 `(N, T, 2)`가 된다. 앞의 축들을 자동으로 전부 펼쳐서 샘플 하나의 특성으로 만드는 것은 아니다. 입력·출력과 weight의 모양은 [PyTorch Linear 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)에서도 확인할 수 있다.

## 3. 손으로 정한 Linear의 출력

계산을 보기 위해 가중치와 편향을 직접 정했다.

```text
가중치: [[1, 0, -1], [0.5, 0.5, 0.5]]
편향:   [0.5, -1]
첫 입력: [1, 2, 3]
```

첫 번째 출력은 `1×1 + 0×2 - 1×3 + 0.5 = -1.5`다. 두 번째 출력은 `0.5×1 + 0.5×2 + 0.5×3 - 1 = 2.0`이다. 아래 실행 코드에서도 `[-1.5, 2.0]`이 나온다.

출력을 두 개 만들었다고 반드시 두 클래스의 확률인 것은 아니다. 레이어는 두 개의 숫자를 계산했을 뿐이다. 어떤 의미로 학습하고 어떤 손실을 쓰는지가 따로 필요하다.

Linear 사이에 ReLU 같은 비선형 활성화를 넣는 이유도 여기서 연결된다. 선형·아핀 변환만 여러 번 합성하면 다시 하나의 아핀 변환으로 표현할 수 있다. ReLU는 `max(0, x)`를 적용해 음수를 0으로 만들고, 여러 층의 모델이 비선형 관계를 표현할 수 있도록 한다.

여러 Linear와 활성화 함수를 연결한 모델을 MLP, Multilayer Perceptron(다층 퍼셉트론)이라고 부른다. FC는 층의 연결 방식이고 MLP는 그런 층들을 연결한 모델이라는 정도로 구분해둔다.

## 4. 이미지를 그대로 펼치면 어떤 점이 달라질까

RGB 8×8 이미지는 픽셀 값이 `3×8×8=192`개다. 이를 펼쳐 Linear에 넣을 수도 있다. 다만 Linear는 인접한 픽셀끼리 먼저 계산하거나, 같은 작은 패턴을 여러 위치에서 같은 가중치로 찾도록 구조가 정해져 있지는 않다.

이미지가 커지면 입력 특성 수도 빠르게 늘어난다. 192개 입력을 100개 출력에 연결하면 편향을 포함해 `192×100+100=19,300`개의 매개변수가 필요하다.

Convolutional Neural Network(합성곱 신경망), CNN은 작은 주변 영역을 보는 계산과 Weight Sharing(가중치 공유)을 활용한다. 같은 가중치를 이미지의 여러 위치에 적용해서 위치마다 새 가중치를 전부 만들지 않는다.

예제의 이미지 텐서는 `(2, 3, 8, 8)`이다. 이미지 2장, RGB 채널 3개, 높이와 너비가 각각 8이다. 이미지 라이브러리에서 받은 값이 `(H, W, C)`라면 PyTorch 입력에 맞춰 축을 옮긴다. 이때는 `permute()`로 축의 의미를 이동해야 한다.

이 작은 텐서는 원소 384개이고 float32라면 데이터 자체가 1,536바이트다. 실제 학습에는 중간 출력, 기울기, 모델과 옵티마이저 상태도 필요하다. 입력 크기만으로 학습 메모리 전체를 계산할 수는 없다.

## 5. Convolution은 작은 창에서 곱하고 더한다

Kernel(커널) 또는 Filter(필터)를 이미지 위로 움직이면서 겹치는 위치의 값끼리 곱하고 더한다. 다음은 한 채널 이미지와 2×2 커널이다.

```text
입력             커널
1 2 3            1  0
4 5 6            0 -1
7 8 9
```

왼쪽 위 창을 계산하면 `1×1 + 2×0 + 4×0 + 5×(-1) = -4`다. 오른쪽으로 한 칸 옮기면 `2×1 + 3×0 + 5×0 + 6×(-1) = -4`다. 아래쪽 두 위치도 -4가 나와 최종 결과는 2×2가 된다.

```text
-4 -4
-4 -4
```

PyTorch의 Conv2d는 엄밀히는 커널을 뒤집지 않는 Cross-correlation(교차상관)을 계산한다. 딥러닝에서는 관례적으로 이를 Convolution(합성곱) 레이어라고 부른다. 여기서도 커널을 뒤집지 않고 계산했다. [Conv2d 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)의 연산 정의를 참고했다.

RGB처럼 입력 채널이 여러 개면 출력 채널 하나를 만들 때 입력의 각 채널에 대한 계산을 합친다. 기본 `groups=1`에서 필터 하나의 가중치가 `(입력 채널 수, 커널 높이, 커널 너비)`를 갖는 이유다. 서로 다른 필터를 여러 개 두면 출력 채널도 여러 개가 된다.

Feature Map(특징 맵)은 이렇게 위치별 계산으로 얻은 출력이다. 출력 채널 수는 사용할 필터 수와 연결되고, 출력의 높이·너비는 커널 크기·이동 간격·패딩 등에 따라 정해진다.

## 6. stride와 padding으로 출력 크기 계산하기

Stride(스트라이드)는 커널이 이동하는 간격이고, Padding(패딩)은 가장자리에 덧붙이는 영역이다. 기본 zero padding이면 0으로 채운다. Dilation(팽창률)은 커널 안에서 참조할 원소 사이의 간격을 조절한다.

dilation이 1일 때 한 축의 출력 크기는 다음처럼 구한다.

```text
출력 크기 = floor((입력 크기 + 2×padding - kernel_size) / stride) + 1
```

`floor`는 소수 부분을 버리고 아래 정수를 취하는 계산이다. 높이와 너비에 각각 적용한다. dilation까지 포함하면 커널이 차지하는 범위가 `dilation × (kernel_size - 1) + 1`이 되므로 다음과 같이 쓴다.

```text
출력 크기 = floor((입력 크기 + 2P - D×(K-1) - 1) / S) + 1
```

이번 CNN에서는 입력 높이가 8이고 `kernel_size=3`, `padding=1`, `stride=1`, `dilation=1`이다. `(8 + 2 - 3) / 1 + 1 = 8`이므로 높이가 유지된다. 너비도 같다.

같은 입력에 padding 없이 3×3 커널을 쓰면 출력은 6×6이 된다. stride를 2로 바꾸고 padding을 1로 두면 `floor(7/2)+1=4`가 되어 4×4가 된다. 단순히 “합성곱은 이미지 크기를 줄인다”라고 외우기보다 설정을 식에 넣어보면 된다.

커널이 지나가는 작은 범위를 Local Receptive Field(국소 수용 영역)라고 생각할 수 있다. stride 1의 3×3 합성곱 두 층을 쌓으면 내부 위치의 출력 하나가 원래 입력의 5×5 범위에 영향을 받는다. 층을 거치며 더 넓은 영역의 정보를 조합할 수 있다는 뜻이다. 실제 범위는 stride·dilation·pooling도 함께 고려한다.

## 7. 합성곱 매개변수는 얼마나 될까

기본 `groups=1`, 편향을 포함하는 Conv2d의 매개변수 수는 다음과 같다.

```text
출력 채널 × 입력 채널 × 커널 높이 × 커널 너비 + 출력 채널
```

`Conv2d(3, 4, kernel_size=3, padding=1)`이면 `4×3×3×3+4=112`개다. weight의 모양은 `(4, 3, 3, 3)`, bias의 모양은 `(4,)`다.

같은 레이어에 더 큰 이미지를 넣어도 가중치 수는 그대로다. 위치가 바뀌어도 같은 가중치를 쓰기 때문이다. 대신 출력 위치가 많아져 연산량과 중간 텐서의 메모리는 늘어난다. 매개변수가 적다는 것과 계산이 항상 가볍다는 것은 따로 봐야 한다.

`groups`를 바꾸는 grouped convolution이나 depthwise convolution에서는 연결되는 입력 채널 수가 달라진다. 이번에는 기본 합성곱만 사용하므로 위 식에 전체 입력 채널 수를 넣었다.

## 8. Pooling과 Flatten은 무엇을 바꿀까

Pooling(풀링)은 주변 영역을 대표값으로 줄인다. Max Pooling은 최댓값, Average Pooling은 평균을 사용한다. 기본 풀링에는 학습되는 가중치가 없다.

예를 들어 다음 2×2 창의 최대값은 4, 평균은 2.5다.

```text
1 3
2 4
```

`MaxPool2d(2)`는 기본 stride도 2여서 이번 8×8 특징 맵을 4×4로 줄인다. 채널 수는 그대로다. 공간 해상도와 이후 계산량을 줄일 수 있지만 세부 위치 정보도 일부 사라진다. 어떤 움직임에도 완전히 같은 출력을 보장하는 기능으로 이해하지 않는다.

Flatten(평탄화)은 여러 축을 한 축으로 펼친다. `Flatten(start_dim=1)`이면 배치 축은 남기고 그 뒤를 합친다.

```text
(2, 4, 4, 4) → (2, 64)
```

샘플 하나마다 `4×4×4=64`개의 특성을 가지게 되고, 이를 `Linear(64, 3)`에 넣으면 샘플마다 점수 세 개가 나온다. Flatten에는 학습 매개변수가 없고, 마지막 Linear에는 `64×3+3=195`개가 있다.

## 9. 전체 코드를 실행해보기

앞의 Linear 손계산, 2×2 커널 계산, 작은 CNN의 shape 확인을 한 파일에 넣었다. 손계산 예제의 가중치를 넣을 때는 `no_grad()` 안에서 복사했다. 마지막 CNN은 무작위 가중치 상태로 모양만 확인한다.

```python
import torch
from torch import nn

torch.manual_seed(42)
# 손으로 정한 가중치로 Linear 계산을 확인한다.
x = torch.tensor([[1., 2., 3.], [0., 1., -1.]])
linear = nn.Linear(3, 2)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[1., 0., -1.], [.5, .5, .5]]))
    linear.bias.copy_(torch.tensor([.5, -1.]))
    linear_output = linear(x)
print("Linear 출력:", linear_output.tolist())
print("Linear weight:", tuple(linear.weight.shape))
print("Linear 매개변수 수:", sum(p.numel() for p in linear.parameters()))

# 작은 창에서 곱하고 더하는 Conv2d 계산. 커널을 뒤집지 않는다.
image = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)
conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[1., 0.], [0., -1.]]]]))
    convolution_output = conv(image)
print("작은 Conv 출력:", convolution_output[0, 0].tolist())

# 이미지 모양의 텐서를 통과시키는 예제이며 분류 학습은 하지 않는다.
model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(start_dim=1),
    nn.Linear(4 * 4 * 4, 3),
)
model.eval()
features = torch.randn(2, 3, 8, 8)
print("CNN 입력:", tuple(features.shape))
with torch.inference_mode():
    for layer in model:
        features = layer(features)
        print(type(layer).__name__, "->", tuple(features.shape))
print("CNN 매개변수 수:", sum(p.numel() for p in model.parameters()))
```

실행 결과다.

```text
Linear 출력: [[-1.5, 2.0], [1.5, -1.0]]
Linear weight: (2, 3)
Linear 매개변수 수: 8
작은 Conv 출력: [[-4.0, -4.0], [-4.0, -4.0]]
CNN 입력: (2, 3, 8, 8)
Conv2d -> (2, 4, 8, 8)
ReLU -> (2, 4, 8, 8)
MaxPool2d -> (2, 4, 4, 4)
Flatten -> (2, 64)
Linear -> (2, 3)
CNN 매개변수 수: 307
```

Linear와 작은 Conv의 값이 손계산과 일치했다. CNN에서 shape가 바뀌는 순서는 다음처럼 정리할 수 있다.

| 단계 | 샘플 하나의 모양 | 학습 매개변수 수 |
|---|---|---:|
| 입력 | `3×8×8` | 0 |
| Conv2d | `4×8×8` | 112 |
| ReLU | `4×8×8` | 0 |
| MaxPool2d | `4×4×4` | 0 |
| Flatten | `64` | 0 |
| Linear | `3` | 195 |
| 합계 |  | 307 |

배치 크기 2는 끝까지 유지된다. 최종 `(2, 3)`은 두 샘플에 점수 세 개씩 나온 결과다. 이 출력에 분류 의미를 부여하려면 데이터의 정답과 손실 함수를 정하고 학습해야 한다. 여기서는 이미지처럼 생긴 난수 텐서를 넣었고, 이미지 분류 학습이나 정확도 측정은 하지 않았다.

## 10. 모델을 연결할 때 확인할 것

`Linear`에서 행렬 크기 오류가 나면 바로 앞의 Flatten 결과부터 본다. 예를 들어 입력 이미지를 키우면 Conv의 매개변수 수는 같아도 Flatten 뒤의 특성 수는 달라질 수 있다. 마지막 Linear가 기대하는 64와 실제 특성 수가 맞는지 확인해야 한다.

Conv2d의 `in_channels`는 실제 채널 축과 맞아야 한다. `(N, H, W, C)`를 그대로 넣어서 높이를 채널 수로 읽게 만들지 않았는지도 확인한다. 각 레이어를 통과할 때 shape를 한 줄씩 찍어보면 어디서 생각과 달라졌는지 찾기 쉽다.

모델이 커지면 표현할 수 있는 관계도 늘어나지만 연산량과 메모리 부담, 과적합 가능성도 함께 고려해야 한다. 좋은 데이터와 정확한 정답, 적절한 분할 없이 층만 늘린다고 성능이 보장되지는 않는다.

이번에는 **Linear는 마지막 특성 축을 바꾸고, Conv는 주변 영역을 같은 가중치로 훑으며, Pooling은 공간 크기를 줄인다**는 흐름을 기억해둔다. 다음 구조를 볼 때도 이름부터 외우기보다 입력·출력 shape와 매개변수 수부터 계산해볼 생각이다.

실행 환경은 Python 3.12.9, NumPy 2.5.2, scikit-learn 1.9.0, PyTorch 2.13.0 (CPU)이며 2026년 9월 28일에 CPU에서 위의 작은 예제를 실행했다. 강의 복습 노트를 바탕으로 레이어의 계산을 다시 설명했고, 손계산·출력 모양·매개변수 수를 대조했다. 대용량 이미지나 사전 학습 모델은 내려받지 않았다.

<nav class="article-links" aria-label="관련 파일과 글 목록">
<p><a href="/blog/ai-study/14-dropout-batch-normalization/">← 08. Dropout과 Batch Normalization의 학습·평가 모드</a></p>
<a href="/blog/">글 목록</a> · <a href="layers_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
