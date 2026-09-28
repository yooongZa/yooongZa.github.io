---
layout: post
title: "19. GPT와 사전학습"
date: 2026-09-28 16:24:50 +0900
permalink: /blog/ai-study/19-gpt-pretraining/
description: "다음 토큰 예측, GPT-1 구조, 사전학습과 미세조정의 차이, 미래 정보 차단을 정리한 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 84분 27초 · GPT·BERT 통합 복습 · 파일에 1.1배속 적용</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="19. GPT와 사전학습 복습 음성">
<source src="/blog/assets/audio/19-modern-nlp-gpt-bert.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/19-modern-nlp-gpt-bert.mp3">음성 파일 듣기</a>
</audio>
</div>

번역 Transformer에서는 원문을 읽는 인코더와 번역문을 만드는 디코더가 함께 있었다. GPT를 볼 때는 앞선 토큰을 받아 다음 토큰을 예측하는 경로에 집중했다. 거기에 사전학습이 붙으면 무엇을 미리 배워두는지 정리해 봤다.

위 음성은 GPT와 BERT를 함께 다루는 modern NLP 통합 복습 음성이다. 20편에도 같은 파일을 연결했다. 파일 자체에 1.1배속이 반영돼 있으므로 플레이어는 1.0배속으로 두면 된다. 본문은 GPT 쪽에 집중한다.

## 1. 사전학습의 정답은 어디서 올까

Pre-training(사전학습)은 여러 작업에 활용할 바탕을 먼저 학습하는 과정이다. GPT의 언어 모델 학습에서는 문장 안의 다음 토큰이 정답이 된다. 사람이 문장마다 “긍정” 같은 별도 라벨을 붙이지 않아도 텍스트에서 입력과 목표를 만들 수 있다. 이런 방식을 Self-supervised Learning(자기지도학습)으로 읽었다.

가령 “나는 오늘 차를 마셨다”가 토큰 네 개라고 가정하면, “나는” 다음의 “오늘”, “나는 오늘” 다음의 “차를”를 맞히는 문제가 생긴다. 실제 토크나이저는 단어를 여러 부분으로 나눌 수 있지만 다음 위치를 예측한다는 원리는 같다.

이 과정에서 문장 구조와 표현의 관계를 배울 수 있다. 다음 토큰 예측을 오래 했다고 해서 모든 질문에 사실대로 답하거나 지시를 잘 따르는 능력이 자동으로 완성되는 것은 아니다. 뒤의 미세조정과 평가를 함께 봐야 하는 이유다.

## 2. GPT는 앞 문맥으로 다음 토큰을 예측한다

GPT-1은 Transformer의 인과적 디코더 구조를 사용했다. 번역 인코더와 그 출력을 읽는 Cross-Attention(교차 어텐션)은 없다. 입력 토큰끼리 관계를 계산하되 미래 위치를 가린다.

GPT-1 논문의 큰 흐름은 언어 모델 사전학습을 한 다음, 태스크에 맞는 입력과 출력층을 사용해 미세조정하는 두 단계다. 문장 분류에서는 마지막 위치의 표현을 분류층에 연결한다. 오늘날 대화형 모델처럼 모든 문제의 답을 자연어로 생성하는 형식만 떠올리면 초기 GPT의 학습 방식을 놓치기 쉽다. [GPT-1 논문](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

“디코더만 남았다”는 표현을 읽을 때도 번역 디코더를 그대로 복사해 사용한다고 생각하지 않기로 했다. 남는 것은 앞 문맥을 읽는 자기 어텐션과 그 뒤의 변환 경로이고, 다른 시퀀스를 읽는 교차 어텐션은 빠진다.

## 3. 토큰 표현과 위치 표현을 더한다

GPT-1은 Token Embedding(토큰 임베딩)과 학습 가능한 Position Embedding(위치 임베딩)을 더한다. 번역 예제에서 사용한 사인·코사인 위치 값과 방식이 다르다. 토큰 종류와 문장 안의 자리를 각각 벡터로 조회하는 식이다.

아래 작은 코드는 차원 16으로 두 표현을 만든다. `nn.Embedding(16, 16)`인 위치 임베딩은 0~15 위치에 대한 학습 가능한 표다. 이 예제에서 더 긴 입력을 넣으려면 구조와 학습 조건을 다시 다뤄야 한다.

블록 안에서는 자기 어텐션 뒤에 Residual Connection(잔차 연결)과 LayerNorm(층 정규화)을 적용하고, FFN(순방향 신경망) 뒤에도 같은 순서를 사용한다. 이번 코드의 Post-LN은 GPT-1을 참고했다. 중간 FFN은 16에서 64로 넓힌 뒤 16으로 돌리고 GELU 활성화 함수를 사용한다.

입력 임베딩과 어휘 출력층의 가중치는 공유한다. 마지막 표현 `(B, T, 16)`에 공유 행렬을 연결하면 어휘 크기 8에 대한 `(B, T, 8)` 점수가 된다. 단어 하나마다 정답 여부를 별도로 출력하는 구조가 아니라, 각 위치에서 다음 토큰 후보 전체의 점수를 계산한다.

## 4. 입력과 정답은 딱 한 칸 차이다

작은 배열을 적어보면 손실의 위치가 분명해진다.

```text
전체 토큰: [BOS, 4, 5, 6, EOS]
입력:      [BOS, 4, 5, 6]
정답:      [4,   5, 6, EOS]
```

현재 위치의 입력을 읽은 출력이 다음 위치의 토큰을 맞힌다. 그래서 인과 마스크의 대각선은 열고, 오른쪽 미래 위치만 가린다. 첫 입력인 BOS가 다음 위치의 정답 4까지 볼 수 있으면 훈련 문제가 쉬워지는 대신 실제 생성 조건과 어긋난다.

아래 자체 모델은 Logits(로짓)만 반환하므로 배열 이동을 밖에서 직접 한다. 라이브러리의 Causal LM(인과 언어 모델)이 `labels`를 받아 내부에서 한 칸 이동하는 경우에는 외부에서 다시 이동시키지 않아야 한다. 어느 클래스가 손실을 계산하는지부터 읽으면 이중 이동을 피할 수 있다.

패딩도 두 군데에서 본다. 자기 어텐션에서 PAD를 참고하지 않게 하고, 정답의 PAD는 교차엔트로피에서 제외한다. 짧은 문장을 긴 문장 길이에 맞췄다고 빈자리까지 학습 목표가 되는 것은 아니다.

## 5. 사전학습, 미세조정, 추론을 구분한다

| 과정 | 무엇을 주나 | 가중치가 바뀌나 |
|---|---|---|
| 사전학습 | 많은 텍스트와 다음 토큰 목표 | 바뀐다 |
| Fine-tuning(미세조정) | 이미 학습한 모델과 목적에 맞는 데이터 | 학습 대상으로 둔 부분이 바뀐다 |
| Inference(추론) | 새 입력 문맥 | 일반적인 추론에서는 고정된다 |

사전학습 본체를 고정하고 새 분류층만 학습하는 방법도 있고, 본체까지 함께 조정하는 방법도 있다. “모델을 활용했다”는 표현만으로 어느 파라미터가 바뀌었는지는 알 수 없다.

Few-shot(소수 예시) 프롬프트는 질문 앞에 예시를 넣어 입력 문맥을 바꾼다. 보통 이때마다 역전파로 가중치를 갱신하지 않는다. Instruction Tuning(지시 미세조정)은 지시와 응답 자료를 손실에 넣어 파라미터를 바꾼다. 둘 다 예시를 사용한다는 이유로 같은 학습이라고 묶지 않기로 했다.

GPT-2·GPT-3으로 이어지는 공부에서는 모델과 데이터 규모, 문맥 안의 예시 활용이 등장했다. 이번 코드는 그 모델들을 재현하는 것이 아니라 초기 GPT의 다음 토큰 계산을 작게 살펴보는 용도다.

## 6. 한 블록 GPT를 실행해 보기

직접 만든 토큰 배열 두 개를 넣는다. 토큰의 뜻을 외워 번역하는 실험이 아니라 Shape(형태), 손실, 인과 마스크를 확인하는 실험이다. 초기 가중치를 작게 만들고 CPU에서 한 번 갱신한다.

```python
"""GPT-1의 핵심 구조를 줄인 한 블록. 학습 성능 실험이 아니다."""
import torch
from torch import nn

PAD, BOS, EOS = 0, 1, 2


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(8, 16, padding_idx=PAD)
        self.position = nn.Embedding(16, 16)
        self.attention = nn.MultiheadAttention(16, 2, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(16)
        self.norm2 = nn.LayerNorm(16)
        self.ffn = nn.Sequential(nn.Linear(16, 64), nn.GELU(approximate="tanh"), nn.Linear(64, 16))
        self.head = nn.Linear(16, 8, bias=False)
        self.head.weight = self.token.weight
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.normal_(parameter, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        length = ids.size(1)
        x = self.token(ids) + self.position(torch.arange(length))
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(
            x, x, x, attn_mask=future, key_padding_mask=ids.eq(PAD), need_weights=False
        )
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return self.head(x)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    full = torch.tensor([[BOS, 4, 5, 6, EOS], [BOS, 4, 7, EOS, PAD]])
    inputs, targets = full[:, :-1], full[:, 1:]
    model = TinyGPT()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1), ignore_index=PAD)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    changed = inputs.clone()
    changed[:, -1] = 5
    with torch.no_grad():
        before = model(inputs)[:, :-1]
        after = model(changed)[:, :-1]
        delta = (before - after).abs().max().item()
    print("입력:", inputs.tolist())
    print("다음 토큰 정답:", targets.tolist())
    print("logits:", tuple(logits.shape))
    print(f"업데이트 전 loss: {loss.item():.4f}")
    print(f"마지막 입력을 바꾼 뒤 앞쪽 출력 최대 차이: {delta:.8f}")


if __name__ == "__main__":
    main()
```

실행 결과를 적어둔다.

```text
입력: [[1, 4, 5, 6], [1, 4, 7, 2]]
다음 토큰 정답: [[4, 5, 6, 2], [4, 7, 2, 0]]
logits: (2, 4, 8)
업데이트 전 loss: 2.1303
마지막 입력을 바꾼 뒤 앞쪽 출력 최대 차이: 0.00000000
```

확인한 환경은 Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0이다. GPT-1 전체 구조·데이터·학습량을 재현한 결과는 아니다. 모델 파일 다운로드와 대규모 사전학습은 하지 않았다.

## 7. 미래 토큰을 바꿔서 마스크를 확인했다

마지막 입력 위치를 다른 ID로 바꾸고, 그보다 앞 위치들의 출력을 비교했다. 최대 차이는 `0.00000000`이었다. 미래 위치의 입력이 앞 위치의 예측으로 새지 않는지 확인한 결과다.

반대로 맨 마지막 출력까지 같아야 하는 것은 아니다. 마지막 위치는 자기 자신을 볼 수 있으므로 그 입력을 바꾸면 해당 위치의 다음 토큰 예측도 달라질 수 있다. 검사에서는 의도적으로 마지막 출력을 제외했다.

이런 확인이 번거로운 전체 학습보다 먼저 필요한 것 같다. 손실이 내려가는 모습만 보면 미래 정답이 새어 들어간 경우를 알아차리기 어렵다. 입력을 조금 바꿨을 때 영향을 받을 수 있는 위치를 먼저 생각하면 마스크를 더 구체적으로 읽게 된다.

## 8. 학습 손실과 생성 문장은 다르게 본다

언어 모델에서는 토큰별 교차엔트로피와 Perplexity(퍼플렉서티)를 볼 수 있다. 자연로그로 계산한 평균 손실에 지수 함수를 적용한 값이 퍼플렉서티다. 다만 토크나이저와 평가 텍스트가 달라지면 토큰의 단위도 바뀌므로 숫자를 그대로 비교하기 어렵다.

배치마다 길이가 다르면 배치 손실을 단순 평균하는 것과 전체 유효 토큰의 손실을 평균하는 것이 다를 수 있다. PAD를 제외한 손실 합과 토큰 수를 모아 계산하는지 확인한다. 검증 문장도 훈련에 중복으로 들어가지 않아야 한다.

생성에서는 마지막 위치의 점수로 토큰을 고르고, 그 토큰을 문맥에 붙이는 과정을 반복한다. 최댓값만 고르는 방식과 Sampling(확률에 따라 뽑기)은 출력이 다르다. Temperature(온도), Top-k, Top-p 같은 설정을 바꾸면 같은 가중치에서도 문장이 달라진다. 모델끼리 비교할 때는 질문뿐 아니라 이 조건도 맞춰야 한다.

## 9. 이번에 기억할 것

GPT 공부에서 내가 먼저 붙잡을 것은 다음 토큰의 위치다. 입력을 어디까지 볼 수 있는지, 그 출력이 어느 정답과 비교되는지, 한 칸 이동을 누가 하는지 순서대로 읽는다. 사전학습을 설명할 때도 “큰 모델을 가져왔다”보다 어떤 목표로 가중치를 배웠는지를 말할 수 있어야겠다.

개인 modern NLP·GPT-1 자습 노트와 생성 음성을 바탕으로 정리했다. **Based-On: none** — 예제 색인에 대응하는 GPT 예제가 없어 작은 모델을 직접 작성했다. 이번에는 한 스텝 계산과 마스크의 영향 범위까지만 확인했다. 답변 품질이나 사전학습 성능을 측정한 기록은 없다.

<nav aria-label="관련 글">
<p><a href="/blog/ai-study/18-transformer-translator/">← 18. Transformer로 번역기 만들기</a></p>
<p><a href="/blog/ai-study/20-bert-understanding/">20. BERT의 문장 이해 →</a></p>
<a href="/blog/">글 목록</a> · <a href="gpt_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
