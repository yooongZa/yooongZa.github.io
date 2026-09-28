---
layout: post
title: "17. Attention에서 Transformer로"
date: 2026-09-28 15:45:10 +0900
permalink: /blog/ai-study/17-attention-to-transformer/
description: "위치 인코딩과 Q·K·V, 여러 헤드, 인과 마스크, FFN·잔차·LayerNorm을 작은 계산으로 정리한 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 30분 09초 · Transformer 개념·후속 모델 복습</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="17. Attention에서 Transformer로 복습 음성">
<source src="/blog/assets/audio/17-attention-to-transformer.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/17-attention-to-transformer.mp3">음성 파일 듣기</a>
</audio>
</div>

앞의 번역기와 요약기에서는 RNN 계열 모델이 문장을 순서대로 읽고, Attention으로 필요한 원문 위치를 다시 참고했다. Transformer는 이 흐름에서 순환 계산을 빼고 Attention을 중심에 놓는다.

이번에는 위치 정보가 필요한 이유부터 Q·K·V, 여러 헤드, 마스크, Encoder–Decoder 구조까지 정리한다. 위 음성은 뒤에 나온 모델들의 변화까지 포함한 Transformer 개념 복습용 생성 음성이다. 본문 코드는 작은 Self-Attention 계산과 마스크의 효과를 확인하도록 따로 만들었다.

## 1. 순환 계산이 없어지면 무엇이 달라질까

RNN은 앞 위치의 상태가 계산되어야 다음 위치의 상태를 계산할 수 있다. Attention을 추가한 RNN에서도 이 순환 경로는 남아 있었다. Transformer는 한 층 안에서 여러 위치의 표현을 함께 놓고 관계를 계산한다.

이 방식은 학습할 때 여러 위치를 병렬로 처리하기 좋다. 다만 모든 작업이 항상 한 번에 끝나는 것은 아니다. 번역문을 새로 만들 때는 아직 다음 입력 토큰을 모르므로, 앞서 예측한 토큰에 이어 다음 토큰을 생성한다.

문장 안의 모든 위치 쌍을 직접 비교하는 기본 Self-Attention(자기 어텐션)은 길이가 늘면 점수 행렬도 커진다. 순환 계산을 없앴다는 장점과 긴 입력의 계산·메모리 부담을 함께 봐야 한다. 아래 설명은 2017년 논문의 기본 Transformer 구조를 기준으로 한다. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 2. 임베딩에 위치 정보를 더한다

토큰 임베딩만 같은 집합으로 놓으면 토큰이 어느 위치에 있는지 별도로 알려 주지 못한다. `내가 너를`과 `너가 나를`처럼 관계가 순서에 따라 달라지는 문장을 처리하려면 위치 정보가 필요하다.

Positional Encoding(위치 인코딩)은 각 위치에 대응하는 벡터를 만들어 임베딩에 더한다. 기본 논문은 차원마다 다른 주기의 sin과 cos를 사용했다.

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

`pos`는 문장 안의 위치, `i`는 짝을 이루는 차원의 번호다. `pos=0`이면 `[0, 1, 0, 1, ...]` 형태가 된다. 단어 뜻을 나타내는 벡터와 위치 벡터를 같은 크기로 만들어 더하므로 최종 차원은 그대로다.

이 방식은 위치 번호로 값을 계산하므로 별도의 학습 파라미터가 필요 없다. 위치 임베딩 자체를 학습하는 방식도 있다. 계산식으로 더 긴 위치를 만들 수 있다는 사실과, 학습 때보다 긴 문장에서 모델이 잘 동작한다는 사실은 구분해야 한다.

## 3. Q와 K로 비중을 정하고 V를 모은다

Query(질의), Key(키), Value(값)는 Attention 계산에서 각각 다른 역할을 한다. 현재 위치가 필요한 정보를 찾는 표현이 Q, 비교할 상대의 표현이 K, 실제로 모을 정보가 V다.

Self-Attention에서는 같은 입력 `X`에 서로 다른 학습 가능한 행렬을 곱해 세 표현을 만든다.

```text
Q = X · W_Q
K = X · W_K
V = X · W_V
```

같은 문장에서 출발해도 변환 행렬이 달라 Q·K·V의 값은 다를 수 있다. `W_Q` 같은 가중치는 학습하면서 바뀌는 모델 파라미터다. Q·K·V는 그 가중치에 현재 입력을 넣어 계산한 값이다. Softmax 뒤의 Attention 비중은 입력 위치마다 참고하는 비율이다. 이 세 종류를 모두 같은 가중치라고 부르면 헷갈리기 쉽다.

Q와 K를 비교해 어느 위치에 얼마나 비중을 줄지 정한다. 그 비중으로 V를 더해 새로운 표현을 만든다. 점수를 구하는 역할과 정보를 가져오는 역할을 나누어 읽으면 행렬곱의 순서도 이해하기 쉽다.

## 4. Scaled Dot-Product Attention 계산 순서

Scaled Dot-Product Attention(크기를 조절한 내적 어텐션)은 다음 식을 사용한다.

```text
Attention(Q, K, V) = softmax(Q · Kᵀ / sqrt(d_k)) · V
```

Q의 각 행과 K의 각 행을 내적하면 위치별 점수가 나온다. `d_k`는 한 헤드에서 Q·K가 사용하는 차원 수다. 차원이 커지면서 점수의 크기가 커져 Softmax가 지나치게 한곳에 몰리는 경향을 줄이려고 `sqrt(d_k)`로 나눈다.

Softmax는 각 점수에 지수를 취하고 그 합으로 나눈다. 원래 점수들을 단순히 합으로 나누는 연산과 다르다. 점수가 `[2, 1, 0]`이면 비중은 대략 `[0.6652, 0.2447, 0.0900]`이 된다. V가 `[10, 20, 30]`이면 가중합은 약 `14.2479`다. 가장 큰 점수의 위치만 고르는 대신 여러 위치의 정보를 비율대로 모은다.

입력 길이를 `T`, 헤드 수를 `h`라 하면 Self-Attention 점수와 비중은 `(B, h, T, T)` 형태가 된다. 끝에서 두 번째 축은 정보를 찾는 query 위치, 마지막 축은 참고할 key 위치다. Softmax는 마지막 축으로 계산한다.

## 5. Self-Attention과 Cross-Attention의 입력

Self-Attention은 Q·K·V가 같은 시퀀스에서 나온다. 인코더에서는 원문 토큰끼리, 디코더에서는 지금까지의 출력 토큰끼리 관계를 계산한다.

Cross-Attention(교차 어텐션)은 Q를 만드는 시퀀스와 K·V를 만드는 시퀀스가 다르다. 번역 디코더의 현재 표현으로 Q를 만들고, 인코더의 원문 표현으로 K·V를 만든다. 그러면 출력의 각 위치가 번역에 필요한 원문 정보를 모을 수 있다.

```text
인코더 self attention: 원문 → Q, K, V
디코더 self attention: 출력 쪽 입력 → Q, K, V
교차 attention:       디코더 표현 → Q, 인코더 출력 → K, V
```

출력 길이가 `T`, 원문 길이가 `S`라면 Cross-Attention 비중의 마지막 두 축은 `(T, S)`다. 정사각형이라고 생각하고 읽으면 원문 길이와 출력 길이가 다른 상황에서 축을 잘못 해석하게 된다.

## 6. 마스크는 미래 정답과 빈자리를 가린다

학습할 때 디코더 입력은 정답을 한 칸 옮긴 `BOS + 이전 정답 토큰들`이다. 모든 위치를 함께 계산하더라도 뒤에 있는 정답 토큰을 미리 보면 안 된다. Causal Mask(인과 마스크)로 현재 위치의 오른쪽을 가린다.

```text
입력 위치      0  1  2  3
query 0        ○  ×  ×  ×
query 1        ○  ○  ×  ×
query 2        ○  ○  ○  ×
query 3        ○  ○  ○  ○
```

대각선은 볼 수 있다. 현재 위치에 들어 있는 것은 다음 토큰을 예측할 때 사용할 이전 토큰이기 때문이다. 점수의 가려진 위치를 `-inf`로 바꾼 뒤 Softmax를 계산하면 해당 비중은 0이 된다.

Padding Mask(패딩 마스크)는 빈자리를 가린다. 인과 마스크가 출력 위치 사이의 순서 제한이라면 패딩 마스크는 각 문장의 실제 길이를 반영한다. Cross-Attention에서도 원문 PAD 위치를 가려야 한다. 정답 PAD를 손실에서 제외하는 일은 별도다.

아래 자체 코드의 불리언 마스크는 `True`인 위치를 가린다. PyTorch에서도 API에 따라 마스크의 불리언 의미가 다를 수 있으므로, 다른 Attention 함수로 바꿀 때는 해당 함수의 설명을 확인해야 한다.

## 7. Multi-Head는 특징 차원을 여러 갈래로 나눈다

Multi-Head Attention(다중 헤드 어텐션)은 여러 Q·K·V 표현 공간에서 Attention을 계산하고 결과를 합친다. 문장 앞부분을 첫 헤드, 뒷부분을 둘째 헤드가 맡는 식으로 토큰을 나누는 것은 아니다. 각 헤드는 허용된 모든 위치를 대상으로 관계를 계산한다.

기본 모델의 `d_model=512`, 헤드 수 8이면 한 헤드의 Q·K 차원은 64다. 아래 작은 예제는 차원 8, 헤드 2를 사용해서 한 헤드의 차원이 4다.

```text
입력/각 Q,K,V       (B, T, 8)
헤드로 나눈 뒤      (B, 2, T, 4)
헤드별 비중         (B, 2, T, T)
헤드별 가중합       (B, 2, T, 4)
다시 합친 뒤        (B, T, 8)
최종 Linear         (B, T, 8)
```

헤드를 합친 뒤에도 출력 변환 행렬을 한 번 더 거친다. 각 헤드에 문법·인물·시점 같은 역할을 미리 지정하지는 않는다. 학습 결과를 보고 어떤 패턴이 나타나는지 분석할 수는 있지만, 모든 헤드가 반드시 서로 다른 의미를 담당한다고 단정하지 않는다.

## 8. FFN, 잔차 연결, LayerNorm이 뒤를 잇는다

Attention으로 위치 사이의 정보를 모았다면 Feed-Forward Network(순방향 신경망)가 각 위치의 표현을 변환한다. 기본 구조는 `Linear → ReLU → Linear`이며 모든 위치에 같은 가중치를 적용한다. 원 논문의 기본 설정에서는 512차원을 2048로 넓혔다가 512로 돌린다.

Attention은 다른 위치의 정보를 섞는다. 위치별 FFN은 그 결과의 특징 차원을 바꿔 처리한다. FFN 자체가 시간축을 따라 다른 위치로 정보를 보내는 연산은 아니다.

Residual Connection(잔차 연결)은 하위 층의 결과에 입력을 더한다. 10편의 ResNet에서 봤던 것처럼 정보와 기울기가 지나갈 경로를 만든다. 더하려는 두 텐서의 크기가 같아야 한다.

Layer Normalization(층 정규화)은 각 토큰의 특징 차원을 기준으로 정규화한다. 배치의 통계를 모으는 Batch Normalization과 기준 축이 다르다. 원 논문의 블록은 `LayerNorm(x + Dropout(하위 층 결과))` 순서다. 정규화를 하위 층 앞에 놓는 구현도 있으므로 코드에서 위치를 확인한다.

## 9. 작은 블록에서 미래 토큰을 바꿔보기

코드에는 위치 인코딩, 두 헤드의 Self-Attention, FFN, 잔차 연결, LayerNorm을 넣었다. 계산을 보기 쉽게 Dropout은 생략했다. Cross-Attention과 최종 어휘 출력 층은 이 예제의 범위에 포함하지 않았다.

인과 마스크를 적용한 출력과 마스크 없는 출력을 각각 구한다. 마지막 토큰 벡터의 한 성분을 바꿨을 때 앞 세 위치의 출력이 변하는지 비교한다.

```python
"""위치 인코딩, 여러 head의 self attention, FFN을 확인한다."""
import math
import torch
from torch import nn


def positional_encoding(length, dimension):
    positions = torch.arange(length).float().unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimension, 2).float()
                            * (-math.log(10000.0) / dimension))
    result = torch.zeros(length, dimension)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result


class AttentionBlock(nn.Module):
    def __init__(self, dimension=8, heads=2):
        super().__init__()
        self.heads = heads
        self.head_dim = dimension // heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, 16), nn.ReLU(), nn.Linear(16, dimension))
        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, inputs, blocked=None):
        batch, length, dimension = inputs.shape
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)

        def split_heads(x):
            return x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = [split_heads(x) for x in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if blocked is not None:
            scores = scores.masked_fill(blocked, float("-inf"))
        weights = scores.softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, dimension)
        hidden = self.norm1(inputs + self.output(context))
        outputs = self.norm2(hidden + self.ffn(hidden))
        return outputs, weights


def main():
    torch.manual_seed(42)
    embedding = nn.Embedding(10, 8)
    positions = positional_encoding(4, 8)
    inputs = embedding(torch.tensor([[1, 2, 3, 4]])) + positions
    # True인 곳을 가린다. 각 행은 query, 각 열은 key 위치다.
    blocked = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
    block = AttentionBlock().eval()
    with torch.no_grad():
        outputs, weights = block(inputs, blocked)
        changed = inputs.clone()
        changed[:, 3, 0] += 5.0
        changed_outputs, _ = block(changed, blocked)
        unmasked, _ = block(inputs)
        changed_unmasked, _ = block(changed)
    print("position 0:", positions[0].tolist())
    print("input / output:", tuple(inputs.shape), tuple(outputs.shape))
    print("attention:", tuple(weights.shape))
    print("row sums:", weights[0, 0].sum(-1).round(decimals=5).tolist())
    print("future attention:", weights[..., blocked].abs().max().item())
    print("earlier outputs unchanged (causal):", torch.allclose(outputs[:, :3], changed_outputs[:, :3]))
    print("earlier outputs changed (unmasked):", not torch.allclose(unmasked[:, :3], changed_unmasked[:, :3]))


if __name__ == "__main__":
    main()
```

실행 결과를 확인했다.

```text
position 0: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
input / output: (1, 4, 8) (1, 4, 8)
attention: (1, 2, 4, 4)
row sums: [1.0, 1.0, 1.0, 1.0]
future attention: 0.0
earlier outputs unchanged (causal): True
earlier outputs changed (unmasked): True
```

미래 위치의 비중은 0이다. 인과 마스크를 적용하면 마지막 토큰을 바꿔도 앞 세 위치의 출력이 유지된다. 마스크가 없으면 앞 위치도 마지막 토큰을 참고할 수 있어서 출력이 달라졌다.

이 비교에서는 같은 블록과 가중치를 그대로 사용했다. LayerNorm은 위치별 특징을, FFN도 각 위치를 따로 처리하므로 가린 미래 정보가 그 두 연산을 거쳐 앞쪽으로 넘어오지는 않는다. 학습 전 무작위 가중치로도 확인할 수 있는 구조적 성질이다.

## 10. 전체 Transformer에 연결해 보기

기본 인코더 한 층은 `Self-Attention → FFN`이다. 디코더 한 층은 `인과 Self-Attention → Cross-Attention → FFN`이다. 각각의 하위 층에 잔차 연결과 정규화를 적용하고, 여러 층을 쌓는다. 원 논문의 기본 모델은 인코더와 디코더를 각각 6층 사용했다.

디코더 마지막 표현을 어휘 수만큼의 점수로 바꾸면 다음 토큰을 예측할 수 있다. 학습에서는 한 칸 이동한 정답을 넣고 인과 마스크로 뒤를 가려 여러 위치의 손실을 함께 계산한다. 생성에서는 BOS부터 자신의 이전 예측을 붙이며 한 단계씩 진행한다.

원 논문에는 Warmup(준비 구간)을 포함한 학습률 조절도 있다. 처음 4,000 step까지 학습률을 높인 뒤 줄이는 설정이다. 학습 시작 시 무작위 상태의 큰 갱신을 조절하려는 학습 설정으로, Attention 점수를 `sqrt(d_k)`로 나누는 연산과는 역할이 다르다.

원 논문에서 임베딩에 `sqrt(d_model)`을 곱하는 설정, Attention 내적을 `sqrt(d_k)`로 나누는 설정도 서로 다른 위치에 적용된다. 아래 작은 코드에서는 위치와 마스크 계산에 집중하려고 임베딩 배율을 추가하지 않았다.

입력 임베딩과 출력 층의 가중치를 공유하는 Weight Tying(가중치 공유)도 볼 수 있다. 크기가 같은 것 외에 토큰 ID의 의미가 맞는지 확인해야 한다. 서로 다른 언어의 사전 크기만 우연히 같다고 같은 행을 공유할 수 있는 것은 아니다.

## 11. 뒤에 나오는 모델 이름은 바꾼 지점과 연결하기

Transformer 구조를 익힌 뒤에는 어떤 문맥을 보고 무엇을 예측하는지로 모델을 구분하면 정리가 쉽다. BERT는 인코더 쪽 양방향 문맥과 가려진 토큰 예측을, GPT 계열은 인과 마스크를 사용하는 다음 토큰 예측을 연결해서 볼 수 있다. 전형적인 GPT의 decoder-only 구조에는 번역 Transformer의 원문용 Cross-Attention이 없다.

Transformer-XL은 앞 구간의 표현을 이어 사용해 긴 문맥을 다루는 방향, Reformer는 Attention 계산과 메모리 부담을 줄이는 방향으로 이해할 수 있다. 음성에서는 이런 후속 흐름까지 더 넓게 다룬다. 이번 글에서는 기본 구조에서 달라지는 지점을 찾는 정도로 정리한다.

기억해둘 계산 순서는 `뜻과 위치를 입력에 담기 → Q·K로 참고 비중 구하기 → V를 모으기 → 특징 변환 → 다음 토큰 점수`다. 여기에 마스크가 어느 위치를 볼 수 있는지 정한다. 식이 길어 보여도 각 축과 역할을 나누면 앞의 Seq2Seq·Attention에서 이어지는 부분을 찾을 수 있다.

## 실행 환경과 참고

개인 Transformer 자습 노트와 Attention 논문 정리를 바탕으로 다시 썼다. 기본 구조·설정은 [원 논문](https://arxiv.org/abs/1706.03762)을 참고했다. 코드는 차원과 구성 요소를 줄인 자체 계산 예제다.

- 실행 환경: Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0.
- 실행 명령: `python transformer_example.py`.
- 확인 범위: 위치 벡터, 헤드 분리·합치기, Attention 비중의 합, 미래 위치 비중 0, 마스크 유무에 따른 미래 입력의 영향.
- 전체 Transformer 번역기 구현·학습과 번역 성능 측정은 이번에 실행하지 않았다.

<nav aria-label="관련 글">
<p><a href="/blog/ai-study/16-news-summarization/">← 16. 뉴스 요약봇 만들기</a></p>
<p><a href="/blog/ai-study/18-transformer-translator/">18. Transformer로 번역기 만들기 →</a></p>
<a href="/blog/">글 목록</a> · <a href="transformer_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
