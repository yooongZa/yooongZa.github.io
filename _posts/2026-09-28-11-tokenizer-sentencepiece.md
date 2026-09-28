---
layout: post
title: "11. 토큰화와 SentencePiece, 문장에서 ID까지"
date: 2026-09-28 15:14:11 +0900
permalink: /blog/ai-study/11-tokenizer-sentencepiece/
description: "전처리·부분단어·Unigram부터 특수 ID, 정규화, 패딩과 마스크까지 작은 말뭉치로 정리한 공부 기록."
---

<div class="audio-note">
<p>복습 음성 · 17분 08초 · 토크나이저와 SentencePiece</p>
<audio style="width: 100%;" controls preload="metadata" aria-label="11. 토큰화와 SentencePiece, 문장에서 ID까지 복습 음성">
<source src="/blog/assets/audio/11-tokenizer-sentencepiece.mp3" type="audio/mpeg">
<a href="/blog/assets/audio/11-tokenizer-sentencepiece.mp3">음성 파일 듣기</a>
</audio>
</div>

텍스트도 모델에 넣으려면 숫자로 바꿔야 한다. 문장을 어디서 나누고 어떤 조각에 번호를 붙일지부터 정해야 한다는 점이 이미지와 다르다. 이번에는 전처리에서 SentencePiece 학습, 정수 인코딩, 패딩까지 이어서 정리한다.

위 음성은 **「토크나이저와 SentencePiece」의 최신 생성본**이다. 본문 예제는 흐름을 바로 실행해볼 수 있도록 직접 만든 짧은 문장으로 구성했다.

## 1. 토큰, 사전, ID를 나눠서 보기

Corpus(말뭉치)는 문장이나 문서의 모음이다. Token(토큰)은 모델 입력에 쓸 한 단위이며 단어, 형태소, 단어의 일부, 문자 등이 될 수 있다. Vocabulary(어휘 사전)는 사용할 토큰과 각 토큰의 ID를 연결한다.

Tokenizer(토크나이저)는 입력을 정해진 방식으로 나누고 어휘 사전과 연결하는 도구다. 같은 문장을 서로 다른 토크나이저에 넣으면 조각의 수와 ID가 달라질 수 있다.

```text
문장 → 조각 → 정수 ID → 임베딩 벡터 → 모델
```

`고양이`가 ID 17이라고 해서 ID 18인 단어와 의미가 가까운 것은 아니다. ID는 사전에서 해당 항목을 찾을 번호다. 의미를 담는 벡터로 바꾸는 Embedding(임베딩)은 다음 단계에서 다룬다.

One-hot(원-핫)은 사전 크기만큼 긴 벡터에서 해당 ID의 자리만 1로 표시한다. 임베딩은 ID마다 더 작은 차원의 벡터를 연결한다. 토큰을 나누는 방식과 숫자 벡터로 표현하는 방식은 이렇게 구분해두면 편하다.

## 2. 정제 기준은 과제에 맞춰 정한다

전처리 예제에는 소문자화, 문장부호 분리, 특정 문자 제거가 자주 나온다. 하지만 `2.1`, 이메일 주소, 이모지, 코드의 밑줄 같은 문자까지 지우면 필요한 정보가 사라질 수 있다.

감정 분류에서는 `!!!`나 이모지가 감정의 강도를 드러낼 수 있다. 이름을 찾는 작업에서는 대문자가 단서가 될 수 있다. 무엇을 남길지는 실제 문장과 과제를 보고 결정한다.

Cleaning(정제)은 빈 행이나 불필요한 형식 등을 다루는 작업이고, Normalization(정규화)은 문자 표기를 일정한 규칙으로 맞추는 작업이다. 예를 들어 겉보기에 비슷한 전각·반각 문자를 통일할 수 있다. 공백을 합치는 규칙도 복원 결과에 영향을 준다.

학습과 추론에 서로 다른 전처리를 쓰면 같은 표현이 다른 토큰으로 바뀔 수 있다. 전처리 함수, 토크나이저 모델, 특수 토큰 설정을 함께 관리해야 한다.

## 3. 토큰화하기 전에 말뭉치부터 본다

문장 수를 세고 결측·빈 문자열·중복을 확인한다. 그다음 길이 분포와 유난히 짧거나 긴 문장을 살펴본다. 길이만 보고 바로 제거하기보다 실제 내용을 확인하는 편이 낫다.

중복을 없앨 때 `set`으로 바꾸면 순서가 바뀔 수 있다. 첫 등장 순서를 유지할 목적이라면 `dict.fromkeys()` 같은 방법을 사용할 수 있다. 번역용 병렬 말뭉치는 양쪽 언어의 문장 쌍도 함께 보존해야 한다.

문자 길이와 토큰 길이는 다르다. 한국어 20글자가 20토큰이 될 수도 있고 더 적게 나뉠 수도 있다. 실제 모델의 Maximum Length(최대 길이)를 정하려면 토큰화 뒤의 길이 분포를 다시 확인한다.

엄격한 평가에서는 학습·검증·테스트를 먼저 나눈 뒤 **학습 문장으로 토크나이저를 학습**한다. 정답 라벨이 없어도 테스트 문장을 어휘 학습에 사용하면 테스트 분포를 미리 본 셈이 된다. 중복 문장이 여러 분할에 걸치지 않도록 하는 기준도 함께 잡는다.

## 4. 공백, 형태소, 부분단어

`오늘도 공부한다`를 공백으로 나누면 `오늘도 / 공부한다`가 된다. 구현은 간단하지만 `오늘`, `오늘은`, `오늘도`를 각각 별개 항목으로 기억하기 쉽다.

Morpheme(형태소) 분석은 조사와 어미 같은 언어 단위를 구분한다. `오늘 / 도 / 공부 / 한다`처럼 생각해볼 수 있지만, 실제 결과는 분석기와 사전·버전에 따라 달라진다. 위 분할은 설명용이다.

Subword(부분단어)는 말뭉치에서 자주 쓰이는 조각을 배워 사용한다. 자주 등장하는 부분은 길게, 드문 표현은 더 작은 조각으로 처리할 수 있다. 한 단어 전체가 사전에 없어도 구성 조각이 있으면 표현할 여지가 생긴다.

부분단어가 항상 형태소와 일치하지는 않는다. 형태소 분석은 언어 구조를 분석하고, 부분단어 토큰화는 모델 입력에 사용할 조각을 학습한다. 한국어에서 조사·어미가 붙은 다양한 표면형을 다룬다는 공통점은 있지만 기준이 다르다.

## 5. BPE와 WordPiece

BPE, Byte-Pair Encoding(바이트 쌍 인코딩)의 기본 아이디어는 자주 붙어 나오는 두 조각을 합치는 것이다. 문자에서 시작하는 설명용 버전이라면 아래처럼 작은 단계를 반복한다.

```text
처음: c / a / t
c와 a가 자주 붙음 → ca / t
ca와 t도 자주 붙음 → cat
```

실제로 어떤 쌍을 먼저 합칠지는 전체 말뭉치의 빈도와 학습 설정에 따라 정해진다. 이 예시는 실행한 병합 결과가 아니다. BPE를 바이트 단위로 구현하는 경우도 있으므로 모든 구현이 같은 문자 목록에서 시작한다고 생각하지는 않는다.

WordPiece(워드피스)도 부분단어 사전을 만든다. 원래 설명에서는 새 조각이 학습 데이터의 우도를 높이는 방향을 고려한다. 단순히 가장 자주 등장한 쌍을 고르는 BPE와 학습 기준이 다르다.

BERT에서 보이는 `##`는 앞 조각에 이어지는 조각이라는 표기다. 그 표기 자체와 사전을 학습하는 원리를 구분해둔다. 토크나이저를 바꾸면 같은 문장에도 다른 경계 표시가 나올 수 있다.

soynlp처럼 말뭉치의 응집도나 분기 엔트로피로 한국어 경계를 찾는 방법도 있다. 수업에서는 여러 접근을 비교한 뒤 SentencePiece로 직접 사전을 만드는 흐름으로 이어졌다.

## 6. SentencePiece와 Unigram

SentencePiece는 원문 문장으로 토크나이저를 학습하고, 학습한 모델로 인코딩·디코딩하는 도구다. BPE와 Unigram을 지원한다. 언어별 형태소 분석을 먼저 거칠 필요가 없지만, 내부 정규화 과정은 적용된다.

Unigram(유니그램)은 후보 조각을 많이 준비한 뒤 각 조각의 확률을 추정하고, 제거했을 때 손실이 작은 조각을 줄여가는 방식으로 이해했다. BPE가 작은 조각을 합쳐가는 쪽이라면 Unigram은 후보를 추려가는 쪽이다.

SentencePiece는 공백을 `▁`로 표현한다. 예를 들어 `▁책을`은 단어 경계와 관련된 표시를 포함한 piece다. 문장 맨 앞에도 기본 설정의 dummy prefix 때문에 이 표시가 붙을 수 있다. 입력 맨 앞에 실제 공백이 있었다는 뜻으로만 읽으면 헷갈린다.

모델의 정규화·공백 처리와 특수 ID 기본값은 [SentencePiece 공식 설명](https://github.com/google/sentencepiece)에 정리돼 있다. 이번에는 기본 Unigram 흐름을 사용하고 작은 말뭉치에 맞게 어휘 수를 낮췄다.

## 7. 어휘 수와 문장 길이는 함께 바뀐다

Vocabulary Size(어휘 크기)를 늘리면 더 많은 조각을 기억할 수 있다. 긴 조각이 늘면 문장당 토큰 수가 줄기도 하지만, 모든 문장이 짧아지는 것은 아니다.

사전 크기 V와 임베딩 차원 D를 쓰는 임베딩 테이블의 가중치 수는 `V×D`다. 차원이 256일 때 V가 4,000이면 1,024,000개, 8,000이면 2,048,000개다. 짧은 토큰열의 이점과 큰 테이블의 부담을 함께 봐야 한다.

`vocab_size`를 너무 작게 잡으면 기본 문자와 특수 토큰조차 다 넣을 수 없어 학습이 실패할 수 있다. 반대로 아주 작은 말뭉치에 큰 값을 요구하면 그만큼의 조각을 만들지 못할 수 있다.

이번 예제는 `hard_vocab_limit=False`로 요청한 어휘 수보다 적게 만들어지는 것을 허용했다. 설정값 64와 실제 어휘 수를 구분해서 출력한다.

## 8. 특수 토큰은 ID를 직접 확인한다

Unknown(미등록 항목)인 `<unk>`는 어휘에 없는 입력을 처리한다. `<bos>`와 `<eos>`는 문장의 시작·끝을 표시할 때 쓰고, `<pad>`는 문장 길이를 맞추는 빈칸이다.

SentencePiece 기본값은 `unk=0`, `bos=1`, `eos=2`이며 pad는 기본적으로 비활성 상태인 `-1`이다. 예제에서는 앞의 세 ID를 유지하고 **pad를 3으로 따로 지정**했다.

```text
unk=0   bos=1   eos=2   pad=3
```

기본 unk가 0인데 패딩도 무심코 0으로 넣으면, 모르는 입력과 빈칸이 같은 ID가 된다. `ids != 0` 같은 마스크를 쓰면서 실제 미등록 입력까지 빈칸처럼 제외할 수도 있다.

ID를 따로 지정했다고 `encode()` 결과에 시작·끝 토큰이 자동으로 추가되지는 않는다. 필요한 모델이라면 `add_bos`, `add_eos` 등의 옵션을 명시한다. 이번 예제에서는 둘 다 추가하지 않는다.

## 9. 문장 몇 개로 전체 흐름 실행하기

직접 만든 학습 문장 12개를 사용했다. 변환을 확인할 두 문장은 학습 목록에 넣지 않았다. 다만 이 문장들에 쓰인 표현 대부분을 학습 문장에서 이미 보았으므로, 이 결과를 새 분야의 언어 처리 성능으로 해석할 수는 없다.

메모리 안의 `BytesIO`에 모델을 저장한다. 외부 말뭉치를 받거나 작업 폴더에 토크나이저 파일을 만들지 않아도 짧은 흐름을 실행할 수 있다. 이 사용법은 [SentencePiece Python 문서](https://github.com/google/sentencepiece/blob/master/python/README.md#training-without-a-local-filesystem)를 참고했다.

```python
"""직접 만든 문장으로 SentencePiece 학습·변환·패딩을 확인한다."""
import io
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence


def make_tokenizer():
    train_texts = [
        '오늘도 공부한다', '오늘은 자연어를 공부한다',
        '나는 문장을 읽는다', '나는 책을 읽는다',
        '우리는 함께 공부한다', '우리는 문장을 만든다',
        '오늘은 책을 읽는다', '자연어 처리를 공부한다',
        '토큰으로 문장을 나눈다', '문장을 숫자로 바꾼다',
        '책에는 여러 문장이 있다', '함께 읽고 함께 공부한다',
    ]
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(train_texts), model_writer=model,
        model_type='unigram', vocab_size=64, hard_vocab_limit=False,
        character_coverage=1.0, unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        num_threads=1, shuffle_input_sentence=False, minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


def main():
    sp = make_tokenizer()
    # 아래 문장들은 tokenizer 학습 목록에 넣지 않았다.
    texts = ['우리는 책을 읽는다', '읽는다']
    encoded = [sp.encode(text, out_type=int) for text in texts]
    batch = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in encoded],
        batch_first=True, padding_value=sp.pad_id(),
    )
    mask = batch.ne(sp.pad_id())
    print('특수 ID (unk, bos, eos, pad):',
          (sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()))
    print('실제 어휘 수:', sp.get_piece_size())
    print('첫 문장 pieces:', [sp.id_to_piece(i) for i in encoded[0]])
    print('첫 문장 IDs:', encoded[0])
    print('복원:', sp.decode(encoded[0]))
    print('배치:', batch.tolist())
    print('유효 길이:', mask.sum(dim=1).tolist())
    print('자료형:', batch.dtype)
    print('공백 정규화:', repr(sp.decode(sp.encode('  오늘도   읽는다  '))))
    print('미등록 문자에 unk 있음:', sp.unk_id() in sp.encode('책 🧬'))


if __name__ == '__main__':
    main()
```

실행 결과를 확인해봤다.

```text
특수 ID (unk, bos, eos, pad): (0, 1, 2, 3)
실제 어휘 수: 59
첫 문장 pieces: ['▁우리는', '▁책을', '▁읽는다']
첫 문장 IDs: [10, 15, 9]
복원: 우리는 책을 읽는다
배치: [[10, 15, 9], [9, 3, 3]]
유효 길이: [3, 1]
자료형: torch.int64
공백 정규화: '오늘도 읽는다'
미등록 문자에 unk 있음: True
```

어휘 수는 59개다. 첫 문장은 `▁우리는`, `▁책을`, `▁읽는다` 세 조각으로 나뉘었다. ID는 이때 학습한 모델의 사전 번호이므로 다른 말뭉치나 설정에서는 달라질 수 있다.

`torch.long`은 정수 ID를 임베딩에 넣을 때 사용하는 자료형이다. 여기서는 길이를 맞춘 배치와 마스크까지만 만들었다. 감정 라벨이나 분류 점수를 계산하지 않았다.

## 10. 패딩은 넣은 뒤에도 제외할 곳을 정해야 한다

`pad_sequence(..., batch_first=True)`는 `(배치, 최대 토큰 길이)` 모양으로 정렬한다. 짧은 문장의 뒤에는 pad ID 3이 들어간다. `mask.sum(dim=1)`을 보면 패딩을 제외한 실제 길이를 알 수 있다.

문장 벡터를 단어 벡터의 평균으로 만든다고 해보자. 실제 토큰 두 개에 패딩 세 개가 붙었다면 분모는 5가 아니라 실제 길이 2를 사용한다. 임베딩에서 패딩 벡터를 0으로 처리해도 평균의 분모까지 자동으로 바뀌지는 않는다.

Attention(어텐션)이나 토큰별 손실에서도 패딩을 제외할 위치를 정해야 한다. 사용하는 API에 따라 마스크의 `True`가 유효 위치를 뜻할 수도, 가릴 위치를 뜻할 수도 있으니 해당 규약을 확인한다. 여기서 만든 마스크는 `True=실제 토큰`이다.

매우 긴 문장을 자르는 Truncation(잘라내기)은 패딩과 별개다. 앞부분과 뒷부분 중 어디를 남길지, 잘라내는 비율이 얼마나 되는지도 과제에 맞춰 기록한다.

## 11. 디코딩 결과와 원문이 다른 이유

입력 → ID → 문자열의 Round Trip(왕복 변환)은 토크나이저를 확인하는 좋은 방법이다. 다만 정규화나 unknown 처리를 거쳤다면 원문의 모든 문자를 그대로 되돌릴 수는 없다.

예제의 여러 공백은 하나로 정리됐다. 학습 문자에 없는 `🧬`를 넣었을 때는 결과 ID에 unk가 포함됐다. `character_coverage=1.0`은 학습 말뭉치의 문자를 최대한 담겠다는 설정이지, 세상의 모든 문자를 넣는다는 뜻이 아니다.

조각을 문자열로 출력하면 미등록 문자가 그대로 보이는 경우도 있다. OOV, Out-of-Vocabulary(어휘 밖 항목)는 표시된 문자열만 보지 말고 **unk ID 포함 여부**로 확인한다.

Byte Fallback(바이트 대체)을 쓰면 미등록 문자를 바이트 조각으로 표현할 수 있다. 대신 관련 어휘와 토큰 길이가 늘 수 있다. OOV 비율만 줄었다고 최종 과제의 정확도도 올랐다고 결론 내리지는 않는다.

## 12. 감정 분류에 연결할 때 비교할 것

NSMC 같은 영화 리뷰 감정 분류에서는 다음 순서로 이어진다.

```text
분할 → 학습 문장으로 토크나이저 학습
     → 같은 모델로 train/validation/test 인코딩
     → 패딩·마스크 → 임베딩·분류기 학습
     → validation으로 설정 선택 → test 평가
```

공백 토큰화와 SentencePiece를 비교하려면 데이터 분할, 분류기, 임베딩 차원, 학습 횟수 같은 조건을 맞춘다. BPE와 Unigram을 비교할 때도 한 번에 바꿀 조건을 정해둔다.

기록할 것은 어휘 수, 평균·상위 구간 토큰 길이, unknown 비율, 잘린 문장 비율, 시간, 최종 분류 성능이다. 토큰 수가 적다는 사실만으로 어떤 방법이 항상 좋다고 말하기는 어렵다.

이번 글에서는 **작은 토크나이저 학습과 변환·복원·패딩·unknown 처리**를 실행했다. NSMC 다운로드와 분류기 학습은 하지 않았다. 다음 편에서는 만들어진 ID가 어떻게 벡터를 가리키고, 그 벡터를 어떻게 배우는지 정리한다.

실행 환경: Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), SentencePiece 0.2.2, Gensim 4.4.0. 필요한 패키지는 `sentencepiece`, `torch`이며 `python tokenizer_example.py`로 실행한다.

<nav aria-label="관련 글">
<p><a href="/blog/ai-study/10-resnet-skip-connection/">← 10. ResNet과 Skip Connection, 깊게 쌓는 방법</a></p>
<p><a href="/blog/ai-study/12-word-embedding/">12. 워드 임베딩, 단어를 벡터로 바꾸고 배우기 →</a></p>
<a href="/blog/">글 목록</a> · <a href="tokenizer_example.py" download>예제 코드</a> · <a href="article.md">Markdown</a>
</nav>
