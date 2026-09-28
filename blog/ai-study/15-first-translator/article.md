앞 글에서 본 Seq2Seq와 Attention으로 번역기를 만들려면 원문과 번역문의 짝부터 맞춰야 한다. 문장 쌍을 준비하고 다음 토큰을 예측해 손실을 구하는 과정에서, 데이터의 짝과 토큰 위치가 어떻게 연결되는지 살펴봤다.

위 음성은 Seq2Seq 번역기 프로젝트 복습용 생성 음성이다. 본문 예제는 직접 만든 문장 두 쌍으로 계산 한 번을 확인한다. 음성에서 다루는 프로젝트와는 규모가 다르다.

## 1. 번역기의 출발점은 문장 쌍이다

번역 학습 자료는 원문과 그 원문에 대응하는 번역문이 한 쌍이어야 한다. 빈 문장, 깨진 문자, 지나치게 긴 문장 등을 정리할 때도 쌍 단위로 처리한다. 한쪽만 삭제하거나 섞으면 입력과 정답의 연결이 어긋난다.

중복도 기준을 정해야 한다. 똑같은 원문·번역문 쌍은 중복으로 볼 수 있지만, 같은 원문에 자연스러운 번역이 여러 개 붙는 경우도 있다. 원문만 같다는 이유로 모두 지울지는 자료의 목적에 따라 결정한다. 평가 자료에 같은 원문이 걸쳐 들어가지 않도록 묶어서 분할하는 방법도 생각할 수 있다.

전체 자료의 중복과 빈 값을 먼저 점검하고, 학습·검증·테스트를 나눈다. Tokenizer(토크나이저)와 Vocabulary(어휘 사전)를 데이터로 학습한다면 학습 분할에서 기준을 정한다. 검증과 테스트에는 그 기준을 그대로 적용한다.

문자 정제도 번역할 정보를 보존해야 한다. 이름이나 숫자, 부정 표현을 지워 버리면 모델이 읽어야 할 뜻부터 달라진다. 문장 길이를 제한할 때 공백 단어 수와 부분단어 토큰 수도 구분한다. 공백 기준으로 짧은 문장이 토큰화 뒤에도 반드시 짧은 것은 아니다.

## 2. 두 언어의 ID 표와 특수 토큰을 정한다

한국어의 ID 4와 영어의 ID 4가 같은 뜻일 필요는 없다. 이번 예제는 입력용과 출력용 임베딩을 따로 둔다. 실제 프로젝트에서 두 언어가 사전을 공유하는지는 사용하는 토크나이저의 설계에 달려 있다.

| 토큰 | 예제의 ID | 역할 |
|---|---:|---|
| PAD | 0 | 배치 안에서 길이를 맞추는 빈자리 |
| BOS | 1 | 출력 생성을 시작하는 신호 |
| EOS | 2 | 출력 생성을 끝내는 신호 |
| UNK | 3 | 사전에 없는 토큰 |

이 숫자는 예제의 약속이다. 라이브러리나 저장된 토크나이저에서는 실제 ID를 확인해야 한다. BOS를 SOS라고 부르는 자료도 있으므로 이름보다 용도를 먼저 본다.

어휘 수를 늘리면 모르는 토큰을 줄이는 데 도움이 될 수 있다. 동시에 임베딩과 마지막 출력 층의 크기가 커지고 드문 토큰의 학습 자료는 적어진다. 어휘 수 하나만 크게 바꿔 번역이 좋아질 것이라고 예상하기는 어렵다.

## 3. 디코더 입력과 정답은 한 칸 차이가 난다

정답 번역이 `i read books`라면 학습 자료는 다음처럼 만든다.

```text
디코더 입력: BOS  i     read   books
예측할 정답: i    read  books  EOS
```

첫 위치에서 BOS를 보고 `i`를 예측한다. 다음 위치에서는 정답 토큰 `i`를 받고 `read`를 예측한다. 이런 Teacher Forcing(교사 강요)은 앞선 정답을 제공해 다음 토큰 예측을 학습시키는 방식이다.

입력과 정답을 따로 잘라 길이를 맞추면 마지막 EOS가 사라지거나 내용이 어긋날 수 있다. 최대 길이가 4라면 먼저 실제 내용 토큰을 3개까지 자르고, 같은 내용에서 `[BOS] + 내용`과 `내용 + [EOS]`를 만든다. 그다음 남은 자리에 PAD를 붙인다.

짧은 정답이 `read books`이면 아래처럼 된다.

```text
디코더 입력: BOS   read   books  PAD
예측할 정답: read  books  EOS    PAD
```

EOS도 모델이 맞혀야 할 정답이다. PAD는 손실에서 제외한다. 둘을 같은 빈자리로 취급하면 문장을 끝내는 방법을 배울 기회를 잃는다.

## 4. Encoder 출력과 Attention, Decoder를 연결한다

이번 코드는 GRU(게이트 순환 유닛)를 사용한다. 앞 글의 LSTM과 달리 셀 상태 `c`를 따로 넘기지 않고 은닉 상태 `h`를 전달한다. 각 계산의 순서는 다음과 같다.

```text
원문 ID → 원문 embedding → encoder GRU
                             ├─ 모든 위치의 상태
                             └─ decoder 초기 은닉 상태

이전 decoder 상태 + 원문의 모든 상태 → attention → context
현재 정답 토큰 embedding + 이전 상태 → decoder GRU → 현재 출력
현재 출력 + context → Linear → 다음 토큰별 점수
```

Attention은 이전 디코더 상태를 Query(질의)로 사용해 원문의 각 상태와 점수를 구한다. 현재 토큰의 임베딩은 GRU에 들어가고, 문맥 벡터는 현재 GRU 출력과 마지막에 이어 붙인다. Attention을 어디에 연결하는지는 구현마다 다르므로 실제 코드를 따라 읽어야 한다.

입력은 `나는 책을 읽는다`, `책을 읽어` 두 개다. 원문 최대 길이 `S=3`, 배치 크기 `B=2`, 정답 쪽 길이 `T=4`, 은닉 크기 `H=8`, 출력 어휘 수 `V=7`로 잡았다.

```text
원문 ID                  (S, B)       = (3, 2)
encoder 전체 상태        (B, S, H)    = (2, 3, 8)
한 단계 attention 비중   (B, S)       = (2, 3)
한 단계 context          (B, H)       = (2, 8)
현재 출력과 context 연결 (B, 2H)      = (2, 16)
전체 단계 logits         (T, B, V)    = (4, 2, 7)
```

14편의 예제는 `batch_first=True`였다. 이번 GRU는 시간축을 앞에 둔다. 변수의 이름만 보고 같은 형태라고 생각하지 말고, 어느 축이 배치와 시간인지 확인해야 한다.

## 5. PAD는 세 군데에서 따로 처리한다

첫 번째는 **원문을 읽는 RNN**이다. 짧은 문장 뒤의 PAD까지 계산하면 마지막 은닉 상태가 실제 마지막 단어 뒤에서도 바뀔 수 있다. 코드에서는 `pack_padded_sequence`에 실제 길이를 넘겨 유효한 위치만 순환 계산한다. 패딩은 오른쪽에만 붙이고, 길이는 PAD가 아닌 토큰 개수로 계산했다. [PyTorch의 packed sequence 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)

두 번째는 **Attention**이다. 원문의 PAD 위치에 해당하는 점수를 Softmax 전에 가린다. 그래야 빈자리로 비중이 분배되지 않는다. 빈 문장은 미리 제거해 적어도 하나의 유효한 위치가 있어야 한다.

세 번째는 **정답의 손실**이다. `ignore_index=PAD`로 정답 PAD 위치를 평균에서 제외한다. Attention 마스크는 원문의 길이를 기준으로, 손실 마스크는 정답의 길이를 기준으로 만든다. 서로 길이도 역할도 다르다.

개인 실습 기록에서는 손실의 PAD 제외를 먼저 다뤘다. 이 글의 작은 코드에는 인코더 길이 처리와 Attention 마스크까지 넣어 세 역할을 확인할 수 있게 했다. `padding_idx`를 임베딩에 지정하는 것만으로 이 세 처리가 모두 끝나지는 않는다.

## 6. 다음 토큰 점수에서 학습 한 번까지

출력 층은 단어마다 Logit(정규화 전 점수)을 만든다. `CrossEntropyLoss`에는 이 점수와 정답 ID를 전달한다. 먼저 Softmax를 적용할 필요가 없다. 코드에서는 `(T, B, V)`를 `(T×B, V)`로 펼치고 정답도 같은 순서로 펼쳤다. 정답의 자료형은 정수 `long`이다. [PyTorch CrossEntropyLoss 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

학습 계산은 `zero_grad → forward → loss → backward → clip → step` 순서다. 기울기를 구한 뒤 최대 norm을 1로 제한하고 Adam이 가중치를 갱신한다. 이 숫자는 작은 예제의 설정이며 데이터에 맞춘 최적값을 찾은 결과는 아니다.

```python
"""두 문장 쌍으로 학습 계산 한 번과 제한된 생성을 확인한다."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def target_pair(tokens, max_len):
    # 같은 내용에서 입력·정답을 만들어 EOS 자리를 남긴다.
    content = tokens[:max_len - 1]
    decoder = [BOS] + content
    labels = content + [EOS]
    missing = max_len - len(decoder)
    return decoder + [PAD] * missing, labels + [PAD] * missing


class TinyTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_embedding = nn.Embedding(8, 4, padding_idx=PAD)
        self.tgt_embedding = nn.Embedding(7, 4, padding_idx=PAD)
        self.encoder = nn.GRU(4, 8)
        self.decoder = nn.GRU(4, 8)
        self.query = nn.Linear(8, 8, bias=False)
        self.key = nn.Linear(8, 8, bias=False)
        self.score = nn.Linear(8, 1, bias=False)
        self.output = nn.Linear(16, 7)

    def encode(self, source):
        # source: (S, B), 오른쪽에만 PAD를 붙인 입력
        lengths = source.ne(PAD).sum(0).cpu()
        packed = pack_padded_sequence(
            self.src_embedding(source), lengths, enforce_sorted=False
        )
        encoded, hidden = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, total_length=source.size(0))
        return encoded.transpose(0, 1), hidden  # (B, S, H), (1, B, H)

    def step(self, token, previous, encoded, source_pad):
        scores = self.score(torch.tanh(
            self.query(previous[-1]).unsqueeze(1) + self.key(encoded)
        )).squeeze(-1)
        weights = scores.masked_fill(source_pad, float("-inf")).softmax(-1)
        context = (weights.unsqueeze(-1) * encoded).sum(1)
        current, hidden = self.decoder(self.tgt_embedding(token).unsqueeze(0), previous)
        logits = self.output(torch.cat([current[0], context], dim=-1))
        return logits, hidden, weights

    def forward(self, source, decoder_input):
        encoded, hidden = self.encode(source)
        source_pad = source.T.eq(PAD)
        logits, weights = [], []
        for token in decoder_input:
            step_logits, hidden, attention = self.step(token, hidden, encoded, source_pad)
            logits.append(step_logits)
            weights.append(attention)
        return torch.stack(logits), torch.stack(weights)

    @torch.no_grad()
    def generate(self, source, max_new_tokens=5):
        # 이 작은 예제의 생성 입력은 문장 한 개다.
        encoded, hidden = self.encode(source)
        token = torch.tensor([BOS])
        result = []
        for _ in range(max_new_tokens):
            logits, hidden, _ = self.step(token, hidden, encoded, source.T.eq(PAD))
            token = logits.argmax(-1)
            result.append(token.item())
            if token.item() == EOS:
                break
        return result


def main():
    torch.manual_seed(42)
    # source: 나는=4, 책을=5, 읽는다=6, 읽어=7
    # target: i=4, read=5, books=6. 언어별 ID 표는 별개다.
    source = torch.tensor([[4, 5, 6], [5, 7, PAD]]).T
    pairs = [target_pair(t, 4) for t in [[4, 5, 6], [5, 6]]]
    decoder_input = torch.tensor([p[0] for p in pairs]).T
    labels = torch.tensor([p[1] for p in pairs]).T
    model = TinyTranslator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    logits, attention = model(source, decoder_input)
    loss = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=PAD)
    before = model.output.weight.detach().clone()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("decoder input:", decoder_input.T.tolist())
    print("labels:", labels.T.tolist())
    print("logits / attention:", tuple(logits.shape), tuple(attention.shape))
    print("valid target tokens:", labels.ne(PAD).sum().item())
    print("loss:", round(loss.item(), 6))
    print("PAD attention:", attention[:, 1, 2].detach().tolist())
    print("weight changed:", not torch.equal(before, model.output.weight))
    model.eval()
    print("after one step, generated IDs:", model.generate(source[:, :1]))


if __name__ == "__main__":
    main()
```

실행 결과를 확인해봤다.

```text
decoder input: [[1, 4, 5, 6], [1, 5, 6, 0]]
labels: [[4, 5, 6, 2], [5, 6, 2, 0]]
logits / attention: (4, 2, 7) (4, 2, 3)
valid target tokens: 7
loss: 1.987353
PAD attention: [0.0, 0.0, 0.0, 0.0]
weight changed: True
after one step, generated IDs: [4, 4, 4, 4, 4]
```

정답 중 유효한 토큰은 EOS를 포함해 7개다. 둘째 원문의 마지막 위치는 PAD라서 모든 출력 단계에서 Attention 비중이 0이다. `weight changed: True`는 갱신 한 번이 일어났음을 보여준다.

이후 생성된 `[4, 4, 4, 4, 4]`는 `i`를 반복한 결과다. EOS도 만들지 못해 최대 생성 길이에서 끝났다. 문장 두 쌍으로 한 번 갱신한 상태이므로, 여기서 확인한 것은 데이터와 계산의 연결이다.

## 7. 실제 번역에서는 자신의 이전 예측을 넣는다

생성은 BOS에서 시작한다. 디코더가 낸 점수에서 하나를 고르고 그 토큰을 다음 입력으로 넣는다. Greedy Decoding(탐욕적 생성)은 매 단계 가장 높은 점수의 토큰을 고른다. EOS가 나오거나 최대 생성 길이에 도달하면 멈춘다.

학습 중 틀린 예측을 했더라도 다음 단계에서 정답 입력을 받았다면 다시 올바른 흐름으로 돌아올 기회가 있었다. 생성 중에는 자신의 잘못된 예측이 그대로 다음 입력이 된다. 이 입력 조건의 차이를 Exposure Bias(노출 편향)와 연결해 설명한다.

Beam Search(빔 탐색)는 여러 후보 경로를 남겨 비교하는 방법이다. 길이에 따른 점수 차이와 반복도 고려해야 하며, 후보를 더 많이 남긴다고 의미가 항상 정확해지는 것은 아니다.

검증 손실을 구할 때 `eval()`을 호출했더라도 정답의 이전 토큰을 계속 넣었다면 Teacher Forcing 조건이다. 자유 생성 품질을 보려면 정답 입력 없이 문장을 만들어 확인해야 한다. `eval()`과 `no_grad()`도 역할이 다르다. 전자는 Dropout 같은 층의 동작 모드, 후자는 기울기 기록 여부를 바꾼다.

## 8. 학습 기록에서 손실과 번역문을 같이 읽기

기존 개인 실습에는 어휘 수 10,000으로 7 epoch(전체 학습 자료를 한 번 도는 단위) 학습한 기록이 있다. 기록된 학습 손실은 첫 epoch의 5.3635에서 마지막 2.0055로 내려갔다. 그런데 출력 예시에는 단어 반복과 문장이 끊기는 현상이 남아 있었다. 이 숫자는 당시 기록을 옮긴 것이며 이번에 같은 학습을 재실행한 결과는 아니다.

그래서 결과 표에는 손실만 적기보다 원문과 생성문을 함께 놓는 편이 도움이 된다. 수량이 바뀌지는 않았는지, 누가 무엇을 했는지 남아 있는지, `없다` 같은 부정이 유지되는지 읽는다. 일부 문장이 자연스러워도 다른 문장에서 의미가 크게 바뀔 수 있다.

Embedding Size(임베딩 크기)는 토큰 표현의 차원, Hidden Size(은닉 크기)는 순환 상태의 차원이다. Batch Size(배치 크기), 어휘 수, 최대 길이, 학습률, epoch 수도 각각 다른 역할을 한다. 여러 값을 한꺼번에 바꾸면 어떤 변경이 결과에 영향을 주었는지 분리하기 어렵다. 같은 분할과 평가 문장을 유지하고 한 번에 바꾸는 조건을 줄이는 편이 비교하기 쉽다.

BLEU처럼 참조 번역과 겹치는 표현을 보는 지표도 사용할 수 있다. 다만 하나의 참조 번역으로 가능한 모든 자연스러운 번역을 대표하기는 어렵다. 점수와 실제 문장, 데이터 분할 조건을 함께 남겨야 한다. Attention 그림 역시 참고 위치를 살피는 보조 자료로 사용한다.

## 실행 환경과 참고

개인 Seq2Seq 번역기 복습 노트와 번역 실험 기록을 바탕으로 정리했다. 공개 코드는 문장 두 쌍으로 새로 작성했다. 첫 글에서 정리한 [Bahdanau Attention](https://arxiv.org/abs/1409.0473)의 덧셈 점수 방식을 작게 사용했다.

- 실행 환경: Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0.
- 실행 명령: `python translation_example.py`.
- 확인 범위: 입력·정답 정렬, 길이 제한 시 EOS 보존, PAD의 Attention·손실 제외, 짧은 문장의 단독/배치 인코딩 일치, 역전파와 갱신 한 번, 제한된 토큰 생성.
- 전체 말뭉치 학습과 검증·테스트 성능 측정은 이번에 실행하지 않았다. 출력 손실을 실제 번역기의 성능으로 해석하지 않는다.
