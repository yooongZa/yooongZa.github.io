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
