지난 [09편](/blog/ai-study/15-linear-and-convolution-layers/)에서는 Linear와 Convolution이 입력을 어떻게 바꾸는지 봤다. 그런 층을 많이 쌓으면 어떤 문제가 생길까. ResNet의 `F(x) + x` 연결을 입력 모양과 활성화 함수까지 함께 보며 살펴본다.

위 복습 음성은 **「딥러닝 레이어의 이해 1과 2」**다. 09편과 같은 통합 음성이며 Linear·CNN부터 VGG·ResNet까지 포함한다. 본문에서는 뒷부분인 깊은 신경망과 잔차 연결을 다룬다.

## 1. 층을 깊게 쌓으면 표현을 여러 번 바꾼다

얕은 층의 출력을 다음 층이 받아 다시 가공한다. 이미지에서는 작은 경계나 무늬를 조합해 더 복잡한 특징을 만드는 식으로 설명할 수 있다. 실제로 어떤 특징을 배우는지는 데이터와 학습 결과에 달려 있다.

Depth(깊이)는 계산을 몇 단계 이어가는지, Width(너비)는 각 단계에 얼마나 많은 채널이나 유닛을 두는지와 관련 있다. 층 수가 같아도 채널 수와 커널 크기에 따라 Parameter(매개변수) 수와 계산량이 크게 달라진다.

깊이가 늘면 중간에 거쳐야 할 계산도 늘어난다. 학습 때는 중간 출력과 기울기를 보관할 메모리까지 필요하다. 깊은 모델의 표현력이 유용하려면 그 모델을 실제로 잘 학습시킬 수 있어야 한다.

## 2. AlexNet과 VGG를 보는 이유

수업에서는 ImageNet과 ILSVRC가 모델 구조의 변화와 함께 나왔다. ImageNet은 이미지 데이터셋이고 ILSVRC는 이를 활용한 인식 대회다. Top-1은 가장 높은 점수의 예측 하나에 정답이 있는지, Top-5는 상위 다섯 예측 안에 정답이 있는지 본다. 같은 숫자라도 어느 지표인지 먼저 확인해야 한다.

AlexNet은 합성곱층 5개와 완전 연결층 3개로 알려진 모델이다. ReLU, Dropout, GPU 학습이 함께 등장한다. 여기서는 세부 성적보다 이미지에서 특징을 추출하는 합성곱 부분과 마지막 분류 부분이 이어진다는 구조를 기억해둔다.

VGG는 작은 `3×3` 합성곱을 반복해 쌓는 구성이 특징이다. VGG16의 16은 보통 가중치를 가진 합성곱층 13개와 완전 연결층 3개를 센 수다. ReLU와 Pooling까지 세면 코드에 보이는 모듈 수는 더 많아진다.

모델 이름의 숫자와 코드 줄 수가 맞지 않아도 이상한 일이 아니다. 어느 종류의 층을 세었는지 기준이 다르다.

## 3. 작은 커널을 여러 번 쓰는 이유

Stride(이동 간격)가 1이고 Dilation(팽창률)이 1일 때 `3×3` 합성곱 두 개를 이어 붙이면, 두 번째 출력 하나가 처음 입력의 `5×5` 영역을 보게 된다. 세 개를 연결하면 `7×7`이다. 이 범위를 Receptive Field(수용 영역)라고 한다.

입력·중간·출력 채널을 모두 C로 같게 놓고 편향을 제외하면 가중치 수는 다음처럼 비교할 수 있다.

```text
5×5 합성곱 하나: 25 × C²
3×3 합성곱 둘:   18 × C²

7×7 합성곱 하나: 49 × C²
3×3 합성곱 셋:   27 × C²
```

같은 채널 조건에서는 작은 커널을 쌓는 쪽의 가중치가 적다. 중간에 ReLU도 넣을 수 있어서 비선형 변환을 여러 번 적용한다. 입력 영역의 크기가 같다고 두 구조가 똑같은 함수를 계산하는 것은 아니다.

실제 모델에서는 층마다 채널 수가 달라진다. 가중치 수를 셀 때는 각 층의 `커널 높이 × 커널 너비 × 입력 채널 × 출력 채널`을 따로 계산한다. 중간 출력이 추가되므로 가중치 수가 적다는 사실만으로 메모리와 실행 시간까지 단정하기도 어렵다.

## 4. 기울기는 여러 미분값을 거쳐 전달된다

Backpropagation(역전파)은 Chain Rule(연쇄 법칙)을 이용한다. 앞쪽 층으로 갈수록 여러 단계의 미분이 곱해진다. 구조를 단순화해서 각 단계가 같은 스칼라 값을 곱한다고 생각해보면 크기의 변화를 볼 수 있다.

```text
0.5를 10번 곱하면 약 0.000977
1.5를 10번 곱하면 약 57.67
```

작은 값이 반복되면 기울기가 매우 작아지는 Vanishing Gradient(기울기 소실)가, 큰 값이 반복되면 Exploding Gradient(기울기 폭주)가 생길 수 있다. 기울기가 너무 작으면 가중치 갱신이 잘 전달되지 않고, 너무 크면 학습이 불안정해질 수 있다.

실제 신경망에서는 행렬과 활성화 함수의 미분이 함께 작용한다. 가중치 원소 하나가 1보다 크다고 곧바로 폭주가 결정되는 것은 아니다. 위 숫자는 반복 곱의 직관을 위한 예시다.

초기화, 활성화 함수, Normalization(정규화), Learning Rate(학습률) 등이 이 문제에 영향을 준다. Gradient Clipping(기울기 자르기)은 너무 큰 기울기를 제한할 때 쓰지만, 사라진 기울기를 복구하는 방법으로 볼 수는 없다.

## 5. 깊은 모델의 학습 오차도 커질 수 있다

Overfitting(과적합)은 학습 데이터에는 잘 맞지만 새 데이터에서 성능이 떨어지는 문제다. ResNet을 공부할 때 나오는 Degradation(성능 저하)은 더 깊어진 모델의 **학습 오차 자체도** 나빠지는 현상을 가리킨다.

추가한 층이 입력을 그대로 통과시키는 Identity Mapping(항등 사상)을 배운다면, 더 깊은 모델도 얕은 모델과 같은 계산을 할 수 있을 것 같다. 하지만 일반적인 층을 쌓아놓고 최적화로 그 동작을 찾게 하는 일은 생각만큼 쉽지 않다.

ResNet은 층의 입력을 출력 쪽에 직접 더하는 경로를 두고, 중간 층들이 입력에 보탤 변화를 학습하게 만든다. 깊은 모델의 최적화를 돕는 것이 출발점이다. 이 문제 설정은 [ResNet 원 논문](https://arxiv.org/abs/1512.03385)의 도입과 연결된다.

## 6. 잔차는 입력에 더할 변화다

블록에서 원하는 전체 변환을 `H(x)`, 중간 층들의 계산을 `F(x)`라고 쓰면 다음과 같다.

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

Residual(잔차)은 여기서 입력과 원하는 변환 사이의 차이다. 회귀에서 정답과 예측값의 차이를 부를 때도 같은 단어를 쓰지만, 지금 식의 `x`는 블록의 입력이다.

예를 들어 입력이 `[2, 3]`이고 이 블록에서 더할 값이 `[0.1, -0.2]`라면 덧셈 결과는 `[2.1, 2.8]`이다. 학습은 `F(x)`를 항상 0으로 만들려는 것이 아니다. 전체 손실을 줄이는 데 필요한 변화를 찾는다.

Skip Connection(건너뛰기 연결) 또는 Shortcut(지름길)은 입력을 덧셈 지점까지 보내는 추가 경로다. 블록 안의 합성곱도 그대로 계산한다.

```text
x ── 합성곱·BN·ReLU·합성곱·BN ── F(x) ─┐
└────────────────────────────────── x ─┴─ 더하기 → ReLU
```

여기서는 같은 위치끼리 더한다. Concatenation(이어 붙이기)처럼 채널 축을 길게 늘리는 연산과 구분해둔다.

## 7. 더하려면 두 경로의 모양이 맞아야 한다

입력과 `F(x)`가 모두 `(N, C, H, W)`라면 입력을 그대로 더할 수 있다. 이를 Identity Shortcut(항등 지름길)이라고 한다.

채널 수를 늘리거나 이미지의 높이·너비를 줄이는 블록에서는 입력도 같은 모양으로 바꿔야 한다. 이때 `1×1` 합성곱 등을 지름길에 두는 Projection Shortcut(투영 지름길)을 쓴다.

```text
입력:          (2, 4, 8, 8)
주 경로 출력:  (2, 8, 4, 4)
지름길 출력:   (2, 8, 4, 4)
덧셈 결과:     (2, 8, 4, 4)
```

위 경우 지름길은 입력 채널 4를 출력 채널 8로 바꾸고 `stride=2`로 공간 크기를 줄인다. `1×1` 커널도 채널 사이의 정보를 섞을 수 있다.

PyTorch는 일부 서로 다른 모양을 Broadcasting(브로드캐스팅)으로 더해준다. 잔차 연결을 구현할 때는 연산이 성공했다는 사실에 더해, 의도한 배치·채널·높이·너비가 두 경로에서 일치하는지도 확인한다.

## 8. BasicBlock과 Bottleneck

기본적인 BasicBlock은 `3×3` 합성곱 두 개를 사용한다. 이번 코드의 순서는 다음과 같다.

```text
Conv → BatchNorm → ReLU → Conv → BatchNorm
                                            ↓
입력 또는 projection 출력 ──────────────── 더하기 → ReLU
```

Bottleneck(병목 블록)은 보통 `1×1 → 3×3 → 1×1`로 이어진다. 앞에서 채널을 줄이고 가운데 합성곱을 계산한 뒤 마지막에서 채널을 늘린다. 채널이 큰 상태에서 비싼 `3×3` 계산을 계속하는 부담을 줄이는 구성이다.

ResNet18·34는 BasicBlock, ResNet50은 Bottleneck을 사용하는 대표적인 경우다. ResNet50의 네 구간은 블록 `3, 4, 6, 3`개로 구성되며, 각 Bottleneck에 가중치 층 3개가 있다. 여기에 첫 합성곱과 마지막 분류층을 더해 `16×3+2=50`으로 센다.

실제 구현은 변형이 있다. TorchVision의 Bottleneck은 공간 크기를 줄이는 stride를 가운데 `3×3` 합성곱에 둔다. 같은 ResNet50이라도 구현을 볼 때 이 위치를 확인하면 좋다. 블록별 연산 순서는 [TorchVision ResNet 코드](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)에서 확인했다.

## 9. 작은 블록으로 모양과 기울기 확인하기

합성곱을 배우는 시간을 줄이기 위해 채널 수와 이미지 크기를 작게 잡았다. 아래 코드는 무작위 텐서로 구조를 확인하고, 마지막에 별도의 스칼라 식으로 기울기를 비교한다. 이미지 분류 학습은 하지 않는다.

```python
"""작은 잔차 블록: CPU 실행, 모델/데이터 다운로드 없음."""
import torch
from torch import nn


class SmallResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, 8)
    same = SmallResidualBlock(4, 4).eval()
    down = SmallResidualBlock(4, 8, stride=2).eval()
    with torch.inference_mode():
        print('입력:', tuple(x.shape))
        print('identity 출력:', tuple(same(x).shape))
        print('projection 출력:', tuple(down(x).shape))
        print('projection 경로:', tuple(down.shortcut(x).shape))
        # 마지막 BN의 gamma/beta를 0으로 두어 F(x)=0으로 만든다.
        same.main[-1].weight.zero_()
        same.main[-1].bias.zero_()
        print('F=0이면 ReLU(x):', torch.allclose(same(x), torch.relu(x)))
        print('양수 입력이면 x:', torch.allclose(same(x.abs()), x.abs()))

    # 덧셈 지점만 보는 별도의 스칼라 예제다.
    scalar = torch.tensor(2.0, requires_grad=True)
    plain = 0.1 * scalar
    residual = 0.1 * scalar + scalar
    print('plain 기울기:', round(torch.autograd.grad(plain, scalar)[0].item(), 1))
    print('residual 기울기:', round(torch.autograd.grad(residual, scalar)[0].item(), 1))


if __name__ == '__main__':
    main()
```

실행 출력이다.

```text
입력: (2, 4, 8, 8)
identity 출력: (2, 4, 8, 8)
projection 출력: (2, 8, 4, 4)
projection 경로: (2, 8, 4, 4)
F=0이면 ReLU(x): True
양수 입력이면 x: True
plain 기울기: 0.1
residual 기울기: 1.1
```

`eval()`은 BatchNorm을 평가 모드로 바꾼다. `inference_mode()`는 이 구간의 자동 미분 기록을 끈다. 스칼라 미분 예제는 그 구간 밖에서 실행했다.

마지막 BatchNorm의 gamma와 beta를 0으로 두면 이 예제의 주 경로는 `F(x)=0`이 된다. 그래도 블록 끝의 ReLU가 남아 있으므로 출력은 `ReLU(x)`다. 입력이 모두 0 이상인 경우에는 입력과 출력이 같다. 코드의 두 `True`는 이 차이를 확인한 결과다.

## 10. 지름길이 기울기에 주는 변화

덧셈 직전·직후만 보면 `y=F(x)+x`의 미분은 `F'(x)+1`이다. 벡터에서는 `1` 자리에 Identity Matrix(단위 행렬)가 들어간다.

스칼라 예제의 `F(x)=0.1x`는 미분값이 0.1이다. `x`를 더하면 1.1이 된다. 입력에서 덧셈 지점으로 이어지는 경로가 기울기에도 더해진다는 점을 이 작은 식으로 볼 수 있다.

전체 블록의 기울기에는 덧셈 뒤 ReLU도 영향을 준다. Projection을 쓴 경로는 항등 변환 대신 그 투영의 미분을 거친다. 따라서 지름길을 넣었다고 모든 입력에서 기울기 크기나 분류 정확도가 보장되는 것은 아니다.

## 11. 큰 모델을 사용할 때 연결할 것

ResNet 전체는 여러 블록 뒤에서 특징 맵을 모으고 분류층으로 연결한다. Global Average Pooling(전역 평균 풀링)은 채널마다 공간 위치의 값을 평균낸다. PyTorch의 `AdaptiveAvgPool2d((1, 1))`을 쓰면 공간 출력 크기를 `1×1`로 맞출 수 있다.

사전 학습 모델을 쓸 때는 입력 크기와 정규화 규칙도 해당 가중치에 맞춘다. 구조만 만들어 `weights=None`으로 사용했다면 가중치는 학습 전 상태다. 모델을 생성한 것, 데이터에 맞게 학습한 것, 평가한 것은 각각 확인할 일이 있다.

작은 예제로 **잔차 블록의 순전파, 두 지름길의 모양, `F=0`일 때의 출력, 단순 미분식**을 확인했다. VGG와 ResNet의 정확도·학습 시간 비교는 실행하지 않았다.

실행 환경: Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), SentencePiece 0.2.2, Gensim 4.4.0. 내려받은 파일은 `python resnet_example.py`로 실행한다.

다음에는 이미지에서 텍스트로 넘어가서, 문장을 토큰과 정수 ID로 바꾸는 과정을 정리한다.
