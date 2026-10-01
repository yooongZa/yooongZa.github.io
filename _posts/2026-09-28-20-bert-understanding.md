---
layout: post
title: "20. BERT의 문장 이해"
date: 2026-09-28 16:24:51 +0900
permalink: /blog/ai-study/20-bert-understanding/
description: "양방향 문맥, 입력 임베딩, MLM의 세 치환 방식과 문장 순서 손실을 작은 코드로 확인한 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 84분 27초 · GPT·BERT 통합 복습 · 파일에 1.1배속 적용</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="20. BERT의 문장 이해 복습 음성">
<source src="/blog/assets/audio/19-modern-nlp-gpt-bert.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/19-modern-nlp-gpt-bert.mp3">음성 파일 듣기</a>
</audio>
</div>

GPT는 앞 문맥으로 다음 토큰을 예측한다. BERT는 문장 양쪽 문맥을 함께 읽는다. 입력을 가리는 일과 손실을 계산할 위치를 고르는 일을 구분하고, `[MASK]`가 아닌 위치도 채점되는지 확인해봤다.

위 음성은 19편과 같은 modern NLP 통합 복습 파일이다. GPT부터 BERT까지 이어서 들을 수 있고, 파일에 이미 1.1배속이 적용돼 있다.

## 1. 양쪽 문맥을 보면서 빈자리를 맞힌다

BERT는 Transformer Encoder(인코더)를 사용한다. 예를 들어 “나는 오늘 `[MASK]`를 마셨다”에서 가려진 토큰을 맞힐 때 앞뒤 문맥을 함께 참고한다. GPT의 다음 토큰 학습처럼 오른쪽 전체를 가리는 인과 마스크를 기본으로 쓰지 않는다.

대신 정답 토큰을 일부 가리거나 바꿔 놓아 복원하는 문제를 만든다. 이를 MLM(Masked Language Modeling, 가려진 토큰 예측)이라고 한다. 양방향 문맥을 읽으면서도 그냥 입력을 그대로 복사하는 문제로 끝나지 않도록 입력을 가공하는 셈이다.

원래 BERT는 MLM과 NSP(Next Sentence Prediction, 다음 문장 예측)를 함께 사전학습했다. 이렇게 얻은 표현을 문장 분류, 문장 쌍 판단, 질의응답 같은 작업에 맞춰 미세조정한다. [BERT 논문](https://arxiv.org/abs/1810.04805)

## 2. 입력에는 토큰·위치·구간 정보가 들어간다

문장 쌍을 넣는 기본 모양은 아래와 같다.

```text
[CLS] 문장 A [SEP] 문장 B [SEP]
```

Token Embedding(토큰 임베딩)은 각 ID에 대응하는 표현이다. Position Embedding(위치 임베딩)은 문장 안의 자리, Segment Embedding(구간 임베딩)은 A와 B 같은 구간을 표시한다. 세 벡터를 같은 차원으로 만들어 더한다.

`token_type_ids`의 0과 1은 이 구간 표시다. 감성분석의 부정·긍정 라벨이나 문서 번호를 뜻하지 않는다. 문장 하나를 넣을 때는 한 구간을 사용하는 경우가 많다. 모델에 따라 구간 정보를 사용하지 않는 경우도 있어 모든 Transformer 입력에 꼭 있다고 생각하지 않는다.

`[CLS]`는 분류에 사용할 수 있는 특별한 위치다. 처음부터 문장 뜻을 담은 요약 벡터가 들어 있는 것은 아니다. 자기 어텐션과 학습을 거쳐 주변 문맥이 반영된다. 학습되지 않은 `[CLS]`의 출력을 뽑았다고 문장 이해가 완성된 것은 아니다.

## 3. 원래 토크나이저와 수업 토크나이저를 구분한다

원래 BERT는 WordPiece를 사용했다. 내가 읽은 수업 구현에서는 SentencePiece BPE로 어휘를 준비했다. 둘 다 부분 단어를 사용할 수 있지만 사전, 분할 규칙, 특수 토큰의 번호가 서로 같다며 섞어 쓰면 안 된다.

아래 작은 예제는 수업에서 쓰던 번호 약속에 맞춰 `PAD=0`, `UNK=1`, `BOS=2`, `EOS=3`, `SEP=4`, `CLS=5`, `MASK=6`으로 두었다. 실제 BERT 체크포인트의 ID를 가져온 것은 아니다. 일반 토큰은 7 이상을 사용한다.

어휘 수를 적을 때도 특수 토큰을 이미 포함한 숫자인지 확인한다. 토크나이저의 실제 크기가 8,000이라면 무조건 거기에 특수 토큰 수를 다시 더하는 것은 맞지 않는다. 입력의 가장 큰 ID와 모델 임베딩 행 수가 연결되는지 직접 봐야 한다.

## 4. 15%와 80·10·10은 기준이 다르다

원래 BERT의 MLM에서는 입력 토큰 중 약 15%를 예측 대상으로 고른다. 그 **선택된 위치 안에서** 80%는 `[MASK]`로, 10%는 무작위 토큰으로 바꾸고, 10%는 원래 토큰을 남긴다.

| 선택된 위치의 처리 | 모델에 보이는 입력 | 손실에서 맞힐 정답 |
|---|---|---|
| MASK로 교체 | `[MASK]` | 원래 토큰 |
| 무작위 토큰으로 교체 | 다른 토큰 | 원래 토큰 |
| 그대로 유지 | 원래 토큰 | 원래 토큰 |

그래서 입력에서 `[MASK]` 위치만 찾아 채점하면 나머지 두 종류를 놓친다. 예측 대상으로 골랐는지를 별도로 보관해야 한다. 반대로 고르지 않은 일반 토큰은 입력에 있지만 이번 MLM 손실에서는 제외한다.

비율은 무작위 선택 규칙이므로 작은 배치마다 정확히 80·10·10개로 나뉘는 것은 아니다. 아래 코드는 세 갈래를 확실히 보여주려고 문장마다 세 위치를 직접 골랐다. **15% 추출이나 비율을 재현하는 샘플러는 아니다.**

수업에는 Whole Word Masking(단어 단위 마스킹)도 있었다. 같은 단어에서 나온 부분 토큰들을 묶어 고르는 방식이다. SentencePiece의 단어 시작 표시를 이용할 수 있지만, 그룹 선택과 토큰 예산 때문에 실제 선택 수를 확인해야 한다. 짧은 문장에서 선택 수가 0이 되면 평균 MLM 손실을 계산할 대상도 사라진다.

## 5. 마스킹이라는 이름으로 세 일을 섞지 않는다

첫째는 입력을 `[MASK]`나 다른 토큰으로 바꾸는 일이다. 둘째는 Attention Mask(어텐션 마스크)로 PAD 위치를 참조하지 않게 하는 일이다. 셋째는 채점하지 않을 정답 위치를 `-100`으로 표시하는 일이다.

예제에서 `[MASK]`의 ID는 6이다. 이 위치는 실제 입력이므로 어텐션에서 볼 수 있어야 한다. PAD의 ID 0과는 다르다. Hugging Face BERT에 넘기는 `attention_mask`는 실제 입력 위치가 1, PAD가 0이다. 앞 번역 예제의 PyTorch 불리언 차단 마스크와 표현 방식이 다르다. [BERT 입력 문서](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert)

정답을 만들 때는 원래 입력을 따로 보관한 뒤 가공한 입력을 만든다. 원본 배열 자체를 바꿔버리면 무엇을 복원해야 하는지 잃어버릴 수 있다. 코드의 `clone()` 두 줄은 그 구분을 위해 있다.

## 6. 수업의 문장 순서 문제와 NSP는 다르다

원래 NSP는 두 번째 문장이 실제 다음 문장인지, 다른 문서에서 가져온 문장인지 판단하는 문제다. 내가 읽은 수업 코드는 같은 두 문장의 A-B 순서를 B-A로 뒤집는 예제를 만든다. 이 경우에는 문장 순서 판단에 가까운 문제로 설명하는 편이 정확하다.

아래에서도 `[7, 8]`을 A, `[9, 10]`을 B로 두고 두 순서를 만들었다. 라벨은 원래 순서 1, 뒤집은 순서 0으로 직접 정했다. 라이브러리의 NSP 라벨 의미를 가져온 것이 아니다.

모델 본체는 공유하고 토큰 예측용 출력층과 순서 분류용 출력층을 따로 붙인다. MLM 점수는 `(B, L, V)`, 순서 점수는 `(B, 2)`다. 두 손실을 더하면 같은 인코더가 두 학습 신호를 받는다. 이름이 비슷한 실습을 비교할수록 정답을 어떻게 만들었는지 먼저 봐야겠다.

## 7. 작은 입력으로 두 손실 계산하기

어휘 16개, 은닉 차원 16, 층 1개인 BERT를 설정에서 새로 만든다. 사전학습 가중치를 불러오지 않는다. 원래 BERT의 MLM 변환층과 풀링 처리는 생략하고, 토큰 표현과 CLS 표현에 선형 출력층만 연결했다.

```python
"""MLM의 세 치환 방식과 문장 순서 분류. 15% 무작위 추출은 생략한다."""
import torch
from torch import nn
from transformers import BertConfig, BertModel

PAD, UNK, BOS, EOS, SEP, CLS, MASK = range(7)


def make_batch():
    # A=[7,8], B=[9,10]. 두 번째 행은 B 다음에 A를 둔다.
    original = torch.tensor([[CLS, 7, 8, SEP, 9, 10, SEP, PAD],
                             [CLS, 9, 10, SEP, 7, 8, SEP, PAD]])
    corrupted = original.clone()
    labels = torch.full_like(original, -100)
    # 세 갈래를 하나씩 보여 주기 위해 위치를 직접 고른다.
    for position in [1, 2, 4]:
        labels[:, position] = original[:, position]
    corrupted[:, 1] = MASK
    corrupted[:, 2] = 11  # 원 토큰과 다른 일반 토큰
    # 위치 4는 원래 토큰을 유지하면서 채점한다.
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]]).repeat(2, 1)
    order_labels = torch.tensor([1, 0])  # 1=원래 순서, 0=뒤집은 순서
    return original, corrupted, labels, segments, order_labels


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=16, pad_token_id=PAD,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.encoder = BertModel(config, add_pooling_layer=False)
        # 원래 BERT의 MLM 변환층을 생략한 학습용 선형 헤드다.
        self.mlm = nn.Linear(16, 16)
        self.order = nn.Linear(16, 2)

    def forward(self, ids, segments):
        hidden = self.encoder(input_ids=ids, attention_mask=ids.ne(PAD),
                              token_type_ids=segments).last_hidden_state
        return self.mlm(hidden), self.order(hidden[:, 0])


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    original, corrupted, labels, segments, order_labels = make_batch()
    model = TinyBert()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mlm, order = model(corrupted, segments)
    mlm_loss = nn.functional.cross_entropy(mlm.reshape(-1, 16), labels.reshape(-1))
    order_loss = nn.functional.cross_entropy(order, order_labels)
    optimizer.zero_grad()
    (mlm_loss + order_loss).backward()
    optimizer.step()
    print("원래 입력:", original[0].tolist())
    print("가공한 입력:", corrupted[0].tolist())
    print("MLM 정답:", labels[0].tolist())
    print("MLM/순서 logits:", tuple(mlm.shape), tuple(order.shape))
    print("MLM 채점 위치 수:", labels.ne(-100).sum().item())
    print(f"MLM loss: {mlm_loss.item():.4f}, 순서 loss: {order_loss.item():.4f}")


if __name__ == "__main__":
    main()
```

실행 결과를 적어둔다.

```text
원래 입력: [5, 7, 8, 4, 9, 10, 4, 0]
가공한 입력: [5, 6, 11, 4, 9, 10, 4, 0]
MLM 정답: [-100, 7, 8, -100, 9, -100, -100, -100]
MLM/순서 logits: (2, 8, 16) (2, 2)
MLM 채점 위치 수: 6
MLM loss: 2.4901, 순서 loss: 0.7893
```

확인한 환경은 Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0, Transformers 5.17.0이다. 처음부터 작은 모델을 만들기 때문에 인터넷 연결이나 모델 다운로드가 필요 없다.

## 8. 출력에서 확인한 것

첫 문장의 1번 위치는 MASK로, 2번 위치는 일반 토큰 11로 바뀌었다. 4번 위치의 토큰 9는 그대로 남았다. 정답 배열에는 이 세 위치에만 원래 값 7·8·9가 있고, 나머지는 `-100`이다. 입력이 그대로인 4번 위치도 채점된다는 것을 여기서 확인했다.

두 문장에서 세 위치씩 골라 총 6개 토큰이 MLM 손실에 들어갔다. `[CLS]`, `[SEP]`, PAD를 포함한 전체 16개 위치를 분모로 삼으면 정확도의 뜻이 달라진다. MLM 정확도를 구할 때도 선택된 위치만 대상으로 계산해야 한다.

출력 크기와 손실이 계산되고 한 번 역전파가 된 것까지 확인했다. 이 수치로 한국어를 이해한다거나 문장 순서 판단 성능이 좋다고 말할 수는 없다. 그런 판단에는 별도 검증 자료와 충분한 학습이 필요하다.

## 9. 미세조정과 실행 크기까지 연결하기

문장 분류에서는 사전학습된 본체 위에 새 분류층을 연결하고 정답 라벨로 학습할 수 있다. 토큰 분류에서는 위치마다 답을 내고, 추출형 질의응답에서는 답의 시작·끝 위치를 예측할 수 있다. 본체가 같아도 출력과 정답 모양은 태스크에 맞게 달라진다.

직접 사전학습할 때는 층 수만 줄인다고 메모리가 모두 해결되지는 않는다. 어휘가 8,000개이고 차원이 128이면 토큰 임베딩에만 1,024,000개 값이 필요하다. 옵티마이저 상태와 중간 활성값도 추가된다. 큰 배열을 디스크에 뒀더라도 통째로 메모리에 복사하면 처음 의도한 절약 효과가 사라진다.

작은 입력에서 선택 위치와 손실이 어떻게 연결되는지 확인했다. 수업 코드를 읽을 때도 볼 수 있는 문맥, 복원할 토큰, 채점할 위치를 각각 짚어보면 된다.

개인 modern NLP·BERT 자습 노트를 참고했다. 입력과 축소 모델은 직접 만들었으며 말뭉치 사전학습과 분류 성능 평가는 실행하지 않았다.

<nav aria-label="관련 글">
<p><a href="/blog/ai-study/19-gpt-pretraining/">← 19. GPT와 사전학습</a></p>
<p><a href="/blog/ai-study/21-huggingface-basics/">21. Hugging Face 사용하기 →</a></p>
<a href="/blog/">글 목록</a> · <a href="bert_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
