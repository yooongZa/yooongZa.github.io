---
layout: post
title: "18. Transformer로 번역기 만들기"
date: 2026-09-28 16:24:49 +0900
permalink: /blog/ai-study/18-transformer-translator/
description: "문장 쌍, 특수 토큰, 디코더의 한 칸 이동과 마스크를 작은 번역 모델로 확인한 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 29분 25초 · Transformer 번역기 프로젝트 복습</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="18. Transformer로 번역기 만들기 복습 음성">
<source src="/blog/assets/audio/18-transformer-translator.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/18-transformer-translator.mp3">음성 파일 듣기</a>
</audio>
</div>

앞에서는 Transformer 안에서 Attention이 어떻게 계산되는지 봤다. 이번에는 그 부품을 한국어에서 영어로 번역하는 흐름에 넣어봤다. 실제 입력을 만들 때는 시작 토큰, 한 칸 이동, 패딩 위치를 따로 확인해야 했다.

위 음성은 Transformer 번역기 프로젝트를 복습하는 생성 음성이다. 본문은 수업 내용을 내 말로 다시 정리했고, 코드는 작은 문장 두 개로 학습과 생성 경로를 확인하도록 따로 만들었다.

## 1. 번역 데이터는 문장 두 개가 한 묶음이다

번역에서는 Source(원문)와 Target(목표 문장)이 짝을 이룬다. 한국어 줄만 따로 정렬하거나 중복을 지우면 영어 줄과의 대응이 깨질 수 있다. 먼저 같은 인덱스의 문장을 묶고, 문장 쌍 단위로 빈 문장과 중복을 확인한다.

수업에서 읽은 전처리에는 문자 종류를 제한하는 정규식이 있었다. 이런 규칙을 그대로 쓰면 숫자까지 사라질 수 있다. “3시에 만나요”와 “5시에 만나요”의 차이는 번역에도 필요하다. 정규식 한 줄을 넣기 전에 무엇이 없어지는지 몇 문장으로 확인해야겠다고 적어뒀다.

실제 성능을 평가하려면 Train/Validation/Test(훈련·검증·평가) 분할도 필요하다. 같은 문장이나 거의 같은 문장이 여러 집합에 들어가지 않게 나눈다. 직접 Tokenizer(토크나이저)를 학습한다면 훈련 자료로 어휘를 만들고, 나머지에는 그 토크나이저를 적용한다. 아래 두 문장 예제는 입력 구조를 보는 용도라 성능 측정용 분할은 하지 않았다.

## 2. 한국어와 영어의 토큰 번호는 별개다

수업은 SentencePiece로 두 언어의 어휘를 준비한다. 한국어 ID 4와 영어 ID 4가 같은 뜻이라는 보장은 없다. 따라서 원문 임베딩과 목표 문장 임베딩도 따로 둔다.

작은 코드에서는 토큰화 과정을 생략하고 번호를 직접 정했다. 특수 토큰은 `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`이다. 일반 토큰은 한국어에서 `나는=4`, `차를=5`, `마신다=6`, `안녕=7`로 놓았다. 영어에서는 `i=4`, `drink=5`, `tea=6`, `hello=7`, `you=8`이다.

PAD(Padding, 길이를 맞추는 빈자리)는 문장 내용이 아니다. BOS(Beginning of Sequence, 시작 토큰)는 생성의 출발점이고, EOS(End of Sequence, 종료 토큰)는 문장의 끝을 알려 준다. UNK(Unknown, 미등록 토큰)는 어휘에 없는 입력을 표시한다. 번호 자체보다 모든 단계가 같은 약속을 쓰는지가 중요하다.

## 3. 디코더 입력과 정답을 한 칸 옮긴다

목표 문장 전체가 다음과 같다고 하자.

```text
전체:          BOS  i      drink  tea  EOS
디코더 입력:   BOS  i      drink  tea
맞힐 정답:     i    drink  tea    EOS
```

`full[:, :-1]`을 입력으로, `full[:, 1:]`을 정답으로 사용한다. BOS를 보고 첫 단어를, 마지막 단어를 보고 EOS를 예측한다. 이 코드에서는 손실을 직접 계산하므로 이동도 여기서 한 번 한다.

훈련 중에는 앞서 나온 실제 정답을 입력으로 주는 Teacher Forcing(교사 강요)을 사용한다. 생성할 때는 정답 문장이 없어서 모델이 방금 고른 토큰을 다음 입력에 붙인다. 훈련 손실이 줄어도 생성이 쉽게 무너질 수 있는 이유를 이 차이에서 이해했다.

## 4. 마스크는 어느 축을 가리는지부터 본다

원문 길이를 `S`, 목표 입력 길이를 `T`, 배치 크기를 `B`라고 두면 세 경로가 보인다.

| 적용 위치 | 마스크 | 가리는 대상 |
|---|---|---|
| 인코더 자기 어텐션 | 원문 패딩 `(B, S)` | 원문의 빈자리 |
| 디코더 자기 어텐션 | 미래 `(T, T)`와 목표 패딩 `(B, T)` | 아직 볼 수 없는 정답과 빈자리 |
| 디코더 교차 어텐션 | 원문 패딩 `(B, S)` | 인코더 출력의 빈자리 |

Causal Mask(인과 마스크)는 현재 위치보다 오른쪽을 가린다. 현재 토큰은 다음 토큰을 예측하는 조건이므로 대각선은 열어 둔다. 번역할 원문 전체는 이미 주어져 있으므로 교차 어텐션에 목표 문장의 삼각형 마스크를 그대로 넣지 않는다.

아래 `nn.Transformer` 경로의 불리언 마스크는 `True`를 차단 위치로 읽는다. 다른 Attention API에서는 의미가 달라질 수 있다. 그리고 입력의 PAD를 가리는 것과 정답 PAD를 Loss(손실)에서 제외하는 것은 별도 작업이다. [PyTorch Transformer 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

## 5. 수업 모델에서 무엇을 줄였나

수업 정리에는 차원 512, 헤드 8개, 인코더·디코더 각각 2개 층이 나온다. 여기서는 차원 16, 헤드 2개, 층 1개, FFN 중간 차원 32로 줄였다. Dropout(드롭아웃)은 0으로 두어 출력 비교를 쉽게 했다. 구조를 연결해 보는 예제라 번역 품질을 기대할 크기나 학습량은 아니다.

이번 프로젝트 노트가 사용하는 Pre-LN(정규화를 먼저 적용하는 구조)에 맞춰 `norm_first=True`를 썼다. 앞 글에서 설명한 기본 논문의 Post-LN과 위치가 다르다. 임베딩에는 `sqrt(16)=4`를 곱하고 사인·코사인 위치 인코딩을 더한다. 위치 값은 학습 파라미터가 아닌 Buffer(모델 상태에 함께 두는 값)로 등록한다.

영어 입력 임베딩과 마지막 어휘 출력층은 같은 가중치를 공유한다. 둘 다 영어 어휘 9개에 대응하기 때문이다. 한국어 임베딩까지 공유한 것은 아니다. 이 공유 때문에 PAD 행도 출력 후보 쪽 계산에서 기울기를 받을 수 있다. `padding_idx`가 있다는 이유로 공유 가중치의 그 행이 모든 경로에서 고정된다고 생각하면 안 된다.

## 6. 학습은 토큰별 점수와 정답을 비교한다

최종 Logits(로짓)는 `(B, T, V)`다. 각 문장의 각 위치마다 영어 어휘 `V`개에 대한 점수가 있다. 아래에서는 `(2, 4, 9)`가 나온다. 손실에 넣을 때 앞의 두 축을 합쳐 `(B*T, V)`로 만들고, 정답도 `(B*T,)`로 펼친다.

Cross-Entropy(교차엔트로피)에 들어가는 정답은 정수 ID다. 이 함수 앞에 Softmax를 따로 붙이지 않는다. `ignore_index=PAD`로 빈자리의 채점을 제외한다. 두 번째 문장이 짧아서 정답 8개 위치 중 실제 채점은 6개다.

학습률은 고정 `0.001`로 한 번만 갱신한다. 수업의 Warmup(워밍업) 스케줄은 모델 차원과 학습 스텝을 함께 사용한다. 스케줄을 떼고 기본 배율만 학습률로 남기면 전혀 다른 설정이 된다. 작은 예제에서는 고정 학습률로 바꿨다는 점을 명시해 두는 편이 이해하기 쉽다.

## 7. 두 문장으로 실행해 보기

필요한 패키지는 PyTorch다. 코드는 CPU에서 실행하며 파일이나 모델을 내려받지 않는다. 생성은 최대 5토큰으로 제한했다.

```python
"""두 문장으로 Transformer의 입력·마스크·손실을 확인한다. 사전학습 모델 없음."""
import math
import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class TinyTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_embed = nn.Embedding(8, 16, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(9, 16, padding_idx=PAD)
        position = torch.arange(32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, 16, 2) * (-math.log(10000) / 16))
        pe = torch.zeros(32, 16)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", pe)
        self.transformer = nn.Transformer(
            d_model=16, nhead=2, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=32,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.output = nn.Linear(16, 9, bias=False)
        self.output.weight = self.tgt_embed.weight

    def embed(self, ids, layer):
        return layer(ids) * 4 + self.position[:ids.size(1)]

    def encode(self, src):
        return self.transformer.encoder(
            self.embed(src, self.src_embed), src_key_padding_mask=src.eq(PAD)
        )

    def decode(self, tgt, memory, src_pad):
        length = tgt.size(1)
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(
            self.embed(tgt, self.tgt_embed), memory,
            tgt_mask=future, tgt_key_padding_mask=tgt.eq(PAD),
            memory_key_padding_mask=src_pad,
        )
        return self.output(hidden)

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src.eq(PAD))


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    # 한국어: 나는=4, 차를=5, 마신다=6, 안녕=7
    # 영어: i=4, drink=5, tea=6, hello=7, you=8
    src = torch.tensor([[4, 5, 6, EOS], [7, EOS, PAD, PAD]])
    full = torch.tensor([[BOS, 4, 5, 6, EOS], [BOS, 7, EOS, PAD, PAD]])
    decoder_input, labels = full[:, :-1], full[:, 1:]
    model = TinyTranslator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logits = model(src, decoder_input)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, 9), labels.reshape(-1), ignore_index=PAD
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    generated = torch.tensor([[BOS]])
    with torch.no_grad():
        memory = model.encode(src[:1])
        for _ in range(5):
            scores = model.decode(generated, memory, src[:1].eq(PAD))[:, -1].clone()
            scores[:, [PAD, BOS]] = -torch.inf
            next_id = scores.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if next_id.item() == EOS:
                break
    print("입력/정답:", decoder_input.tolist(), labels.tolist())
    print("logits:", tuple(logits.shape), "채점 토큰:", labels.ne(PAD).sum().item())
    print(f"업데이트 전 loss: {loss.item():.4f}")
    print("한 스텝 뒤 생성 ID:", generated[0].tolist())


if __name__ == "__main__":
    main()
```

실행 결과를 적어둔다.

```text
입력/정답: [[1, 4, 5, 6], [1, 7, 2, 0]] [[4, 5, 6, 2], [7, 2, 0, 0]]
logits: (2, 4, 9) 채점 토큰: 6
업데이트 전 loss: 17.7308
한 스텝 뒤 생성 ID: [1, 2]
```

확인한 환경은 Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0이다. 이 환경에서는 Pre-LN을 사용해 Nested Tensor 최적화가 적용되지 않는다는 PyTorch 경고도 나왔다. 위 작은 텐서의 학습·생성 계산은 완료됐다.

## 8. 생성에서는 인코더 출력을 재사용한다

생성 코드에서 원문은 바뀌지 않으므로 `encode()`를 한 번 호출해 `memory`로 보관한다. 디코더에는 BOS로 시작한 배열을 넣고 마지막 위치의 점수에서 다음 ID를 고른다. 이번에는 Greedy Decoding(최댓값을 고르는 생성)을 사용했다.

PAD와 BOS는 새로 생성할 후보에서 제외한다. EOS를 만나면 멈추고, EOS가 나오지 않더라도 최대 횟수에서 끝낸다. `model.eval()`은 층의 평가 모드를 정하고, `torch.no_grad()`는 기울기 기록을 생략한다. 두 호출의 역할이 다르다.

실행 결과는 `[1, 2]`, 즉 BOS 다음에 바로 EOS였다. 한 번 갱신한 모델이 문장 생성을 제대로 배우지 못한 상태다. 이 결과에서 확인한 것은 생성 루프와 종료 조건이 연결됐다는 점이다. 완성된 번역문이나 학습 성공 사례로 읽으면 안 된다.

## 9. 번역 결과와 Attention 그림을 읽을 때

실제 프로젝트에서는 훈련에 쓰지 않은 문장으로 의미, 숫자, 부정 표현, 이름이 보존되는지 읽어야 한다. BLEU 같은 지표를 함께 볼 수 있지만 단일 문장의 점수만으로 번역 품질을 모두 설명하기는 어렵다. 손실 평균도 PAD를 제외한 유효 토큰 수를 기준으로 비교해야 한다.

Attention 그림의 행은 디코더 입력 위치이고 열은 원문 위치다. BOS 행은 첫 단어를 예측하는 계산에 해당한다. 그림의 밝은 칸을 바로 단어 정렬 정답으로 해석하기보다, 실제 생성문과 함께 참고할 자료로 보는 편이 낫겠다.

학습을 이어갈 때는 모델 가중치 외에 옵티마이저와 스케줄러 상태, 진행 스텝, 토크나이저의 약속도 필요하다. “총 5에포크까지”와 “저장 시점부터 5에포크 더”도 다르다. 이번 실행에서는 저장이나 학습 재개까지 진행하지 않았다.

## 10. 다시 볼 때 기억할 것

내가 다시 확인할 순서는 문장 쌍 → 특수 토큰 → 디코더 입력과 정답 → 마스크 → 손실 → 생성이다. 모델 층을 늘리기 전에 두 문장짜리 입력을 손으로 읽을 수 있어야겠다. 특히 원문 PAD, 목표 PAD, 정답 PAD가 서로 다른 곳에서 처리된다는 점을 기억해 둔다.

자료는 개인 Transformer 자습 노트와 번역기 복습 음성을 참고했다. 공개 코드는 이 글을 위해 따로 작성한 축소 예제다. **Based-On: none** — 예제 색인에 직접 대응하는 번역 Transformer 예제가 없어 자체 코드를 사용했다. 실제 말뭉치 학습과 번역 품질 평가는 이번 글에서 실행하지 않았다.

<nav aria-label="관련 글">
<p><a href="/blog/ai-study/17-attention-to-transformer/">← 17. Attention에서 Transformer로</a></p>
<p><a href="/blog/ai-study/19-gpt-pretraining/">19. GPT와 사전학습 →</a></p>
<a href="/blog/">글 목록</a> · <a href="translation_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
