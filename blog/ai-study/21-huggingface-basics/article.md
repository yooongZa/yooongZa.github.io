Hugging Face의 Hub, Transformers, Datasets는 각각 맡은 역할이 다르다. 문장 하나가 모델 입력으로 바뀌는 과정을 각 클래스의 역할과 연결해봤다.

위 음성은 NLP Framework 수업을 복습하는 생성 음성이다. 1.1배속이 파일에 반영돼 있다. 본문 코드는 외부 모델을 내려받지 않고, 직접 만든 작은 어휘와 분류 모델로 입력 경로만 실행한다.

## 1. Hub, Transformers, Datasets부터 나눈다

Hugging Face Hub는 모델과 데이터 같은 저장소를 찾고 공유하는 공간이다. Transformers는 모델과 토크나이저 등을 코드에서 다루는 라이브러리이고, Datasets는 데이터를 불러오고 가공하는 라이브러리다. 셋을 모두 “Hugging Face”라고 부르면 어느 단계에서 문제가 생겼는지 설명하기 어렵다.

`pipeline`은 전처리, 모델 실행, 결과 정리를 묶어 호출하는 편리한 입구다. 감성분석용으로 학습된 모델에 문장을 주면 분류 결과를 받을 수 있다. 이 호출 자체는 주어진 문장으로 새 학습을 하는 과정이 아니다. 처음 불러올 때는 관련 모델 파일을 내려받을 수 있다.

예측 점수 `0.98`을 봐도 전체 정확도 98%로 읽지 않는다. 특정 입력에서 모델이 계산한 점수와, 정답이 있는 평가 자료 전체에서 측정한 정확도는 다르다. 모델 카드에서 언어와 태스크도 함께 확인해야 한다.

## 2. AutoModel의 Auto가 맡는 범위

`AutoModel`은 설정에 맞는 모델 본체를 선택한다. 사용자가 원하는 문제까지 알아서 정해 주지는 않는다. 문장 분류가 목적이라면 `AutoModelForSequenceClassification`처럼 목적에 맞는 출력 헤드를 선택한다.

| 구성 | 대표 출력 | 해석 |
|---|---|---|
| BERT 본체 | `(B, L, H)` | 각 토큰의 문맥 표현 |
| 문장 분류 모델 | `(B, C)` | 문장마다 범주 C개의 점수 |
| 토큰 예측 모델 | `(B, L, V)` | 위치마다 어휘 V개의 점수 |

Head(출력 헤드)는 본체의 표현을 내가 풀 문제의 답 모양으로 바꾼다. 기본 BERT 가중치를 문장 분류용으로 불러올 때 새 분류층이 초기화될 수 있다. 그 경우 본체를 가져왔더라도 분류 태스크에 맞춰 추가 학습해야 한다.

Config(설정)에는 어휘 수, 차원, 층 수 같은 구조 정보가 들어간다. 설정으로 모델을 만들면 구조에 맞는 새 가중치가 생긴다. `from_pretrained()`는 저장된 가중치를 불러오는 경로다. 설정만 읽어 무작위로 만든 모델과 학습된 모델을 불러온 상태를 구분한다. [Hugging Face 모델 문서](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/model)

## 3. 모델과 토크나이저는 한 세트로 본다

Tokenizer(토크나이저)는 텍스트를 Token ID(토큰 ID)로 바꾼다. 모델은 그 번호로 임베딩의 행을 조회한다. 학습할 때 ID 17이 `cat`이었는데 새 토크나이저에서 `dog`가 17이 되면 숫자 범위가 맞아도 의미의 대응이 어긋난다.

처음에는 모델과 토크나이저에 같은 체크포인트 ID를 사용하는 것이 이해하기 쉽다. 미세조정 모델이 기반 모델의 토크나이저를 그대로 쓰도록 안내하는 경우도 있다. 결국 확인할 것은 저장소 문자열 자체보다 어휘·특수 토큰·정규화 규칙의 호환성이다.

아래 예제는 WordLevel로 어휘를 직접 정했다. 띄어쓰기로 나눈 단어에 번호를 붙이는 단순한 구성이다. 실제 한국어 BERT의 토크나이저나 SentencePiece 성능을 재현하지 않는다. 모르는 단어는 `[UNK]`가 되므로 여기 적은 짧은 문장 밖으로 일반화할 수도 없다.

## 4. 모델로 들어가는 세 가지 배열

| 입력 이름 | 하는 일 |
|---|---|
| `input_ids` | 토큰 번호를 전달한다 |
| `attention_mask` | 실제 입력 위치와 패딩 위치를 구분한다 |
| `token_type_ids` | BERT에서 문장 A·B 같은 구간을 표시한다 |

토크나이저가 반환하는 것은 임베딩 벡터가 아닌 정수 ID다. 임베딩 조회는 모델 안에서 일어난다. 문장 분류의 정답은 별도 `labels`로 넘긴다. 이 예제의 라벨 1은 긍정, 0은 부정으로 내가 정했다.

길이를 제한하는 Truncation(잘라내기)과 길이를 맞추는 Padding(패딩)도 구분한다. 앞의 것은 긴 입력을 줄이고, 뒤의 것은 짧은 입력에 빈자리를 붙인다. 중요한 답변이나 문장의 끝이 잘리면 학습 문제 자체가 바뀔 수 있으므로 잘린 뒤의 입력도 몇 개 읽어본다.

## 5. 패딩은 배치를 만들 때 할 수 있다

길이가 5와 4인 두 문장을 묶을 때 Dynamic Padding(동적 패딩)을 사용하면 최대 길이 5에 맞춰 PAD 하나만 추가한다. 처음부터 둘 다 길이 12로 만들면 빈자리는 15개가 된다. 데이터는 같은데 계산할 위치 수가 달라진다.

`DataCollatorWithPadding`은 이미 토큰화한 예제들을 받아 배치로 묶으며 패딩한다. 토큰화와 배치 구성을 분리해서 읽으니 `map`과 Collator(배치 구성 도구)의 역할도 구분됐다. [Data collator 문서](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/data_collator)

길이가 비슷한 문장을 한 배치에 모으면 패딩을 더 줄일 수 있다. 예를 들어 `[10, 100]`, `[12, 102]`로 묶는 것보다 `[10, 12]`, `[100, 102]`로 묶는 편이 빈자리가 적다. 수업의 `group_by_length`는 이런 원리를 다뤘다. 이미 모든 문장을 같은 최대 길이로 채워놨다면 묶음만 바꾸는 효과는 제한된다.

## 6. Trainer 안에서도 기본 학습 순서는 같다

Trainer(학습 실행 도구)에는 모델, 학습 조건, 훈련·검증 자료, 입력 처리 도구와 배치 구성을 연결한다. 실제 학습 호출 안에서는 순전파 → 손실 → 역전파 → 가중치 갱신이 반복된다. 부품을 묶는 객체를 만들었다는 사실과 학습 루프를 돌렸다는 사실은 다르다.

`model.train()`은 모델을 학습 모드로 바꾼다. `trainer.train()`은 학습 루프를 실행한다. 마찬가지로 `model.eval()`은 평가 모드를 정하고, `trainer.evaluate()`는 평가 자료를 처리한다. 이름이 비슷해서 가장 먼저 구분해 두고 싶었다.

TrainingArguments(학습 조건)에서는 Epoch(데이터 전체를 도는 횟수), 배치 크기, 학습률, Warmup(워밍업), Weight Decay(가중치 감쇠), 평가·저장 주기를 본다. 기울기 누적과 여러 기기를 쓰면 한 배치를 처리한 횟수와 가중치를 갱신한 횟수도 달라진다.

수업에서 읽은 CoLA는 문장의 언어적 수용 가능성을 분류하는 자료다. 영화 리뷰 감성분석과 라벨 의미가 다르다. Accuracy(정확도)뿐 아니라 태스크에 맞는 지표를 정해야 하고, CoLA에서는 MCC(Matthews Correlation Coefficient, 매튜 상관계수)도 확인한다. 평가 데이터만 전달한다고 내가 원하는 지표가 자동으로 모두 출력되지는 않는다.

## 7. 다운로드 없이 입력부터 손실까지 실행하기

아래에서는 Transformers와 tokenizers, PyTorch를 사용한다. 작은 BERT를 설정에서 새로 만들고, 직접 작성한 문장 두 개로 한 번 갱신한다. Trainer를 실행하지 않고 내부 학습 순서를 직접 적어서 데이터 흐름을 봤다.

```python
"""직접 만든 WordLevel 어휘와 무작위 BERT. 다운로드·Trainer 실행 없음."""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig, BertForSequenceClassification,
    PreTrainedTokenizerFast, DataCollatorWithPadding,
)


def make_tokenizer():
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "영화가", "정말", "좋다", "별로다"]
    vocab = {word: i for i, word in enumerate(words)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=core,
                                  model_input_names=["input_ids", "token_type_ids", "attention_mask"],
                                  pad_token="[PAD]",
                                  unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")


def make_model(vocab_size):
    config = BertConfig(vocab_size=vocab_size, hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=16,
                        num_labels=2, pad_token_id=0,
                        hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    return BertForSequenceClassification(config)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    tokenizer = make_tokenizer()
    texts, labels = ["영화가 정말 좋다", "영화가 별로다"], [1, 0]
    features = []
    for text, label in zip(texts, labels):
        item = tokenizer(text, truncation=True, max_length=12)
        item["labels"] = label
        features.append(item)
    batch = DataCollatorWithPadding(tokenizer, return_tensors="pt")(features)
    model = make_model(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    output = model(**batch)
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
    lengths = [len(item["input_ids"]) for item in features]
    print("패딩 전 길이:", lengths)
    print("input_ids:", batch["input_ids"].tolist())
    print("attention_mask:", batch["attention_mask"].tolist())
    print("동적 패딩 수:", batch["input_ids"].eq(0).sum().item())
    print("고정 길이 12의 패딩 수:", 12 * len(texts) - sum(lengths))
    print("logits:", tuple(logits.shape), "한 스텝 후 예측:", logits.argmax(-1).tolist())
    print(f"업데이트 전 loss: {output.loss.item():.4f}")


if __name__ == "__main__":
    main()
```

실행 결과를 적어둔다.

```text
패딩 전 길이: [5, 4]
input_ids: [[2, 4, 5, 6, 3], [2, 4, 7, 3, 0]]
attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
동적 패딩 수: 1
고정 길이 12의 패딩 수: 15
logits: (2, 2) 한 스텝 후 예측: [0, 0]
업데이트 전 loss: 0.6932
```

확인한 환경은 Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0, Transformers 5.17.0, tokenizers 0.23.2다. 패키지와 모델을 새로 설치·다운로드하지 않았고, 준비된 로컬 환경에서 실행했다.

## 8. 결과를 어느 범위까지 읽을까

토큰화된 길이는 5와 4였고 배치에는 PAD 하나가 붙었다. 분류 출력은 `(2, 2)`다. 문장 두 개에 범주 두 개의 점수가 있으므로 이 모양이 맞다. `argmax`는 각 문장의 점수 중 더 큰 범주의 번호를 고른다.

한 스텝 뒤 두 문장이 모두 0으로 나왔다. 정답은 1과 0이지만 이 예제로 감성분석 성능을 배웠다고 할 수는 없다. 무작위 초기화한 모델에 입력과 손실이 연결되는지 확인한 결과다. 좋은 예측만 골라 실으면 초기 상태와 실행 범위를 오해하기 쉬워서 그대로 남겼다.

실제 작업에서는 훈련 데이터로 학습하고 검증 데이터로 조건을 고른 뒤, 따로 둔 평가 데이터로 최종 결과를 확인한다. 손실, 지표, 실제 오답 문장을 함께 읽어야 어떤 문제가 남았는지 보인다.

## 9. 저장할 때도 모델과 입력 규칙을 함께 둔다

`save_pretrained()`로 모델과 토크나이저를 같은 새 폴더에 저장하면 추론에 필요한 설정과 가중치, 입력 규칙을 함께 보관할 수 있다. 로컬 폴더를 `from_pretrained()`에 넘겨 다시 읽는 방식도 가능하다. 이번 첨부 예제 자체에는 저장 작업을 넣지 않았다.

학습을 정확히 이어가려면 옵티마이저·스케줄러·진행 상태를 담은 학습 체크포인트가 추가로 필요하다. 추론용 저장본을 갖고 있다는 것과 같은 스텝에서 학습을 재개할 수 있다는 것은 서로 다른 확인이다.

수업 노트의 API 기준과 현재 실행 환경은 버전이 다를 수 있다. 인자 오류가 나면 모델 이름을 바꾸기 전에 설치 버전과 함수 시그니처를 확인한다. 이 글에서는 실제로 실행한 버전을 위에 남겼다.

개인 NLP Framework 자습 노트와 생성 음성을 참고했다. 직접 만든 어휘·문장·축소 모델로 입력부터 손실 계산까지 확인했다. 사전학습 모델 미세조정, Trainer 전체 학습, 평가 데이터 성능 측정은 진행하지 않았다.
