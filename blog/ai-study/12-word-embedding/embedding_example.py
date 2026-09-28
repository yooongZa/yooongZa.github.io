"""설명용 임베딩 조회와 작은 CBOW 학습. 외부 corpus를 받지 않는다."""
import torch
from torch import nn
from torch.nn.functional import one_hot
from gensim.models import Word2Vec


def lookup_example():
    vocab = {'<pad>': 0, '<unk>': 1, '고양이': 2, '강아지': 3, '책': 4, '읽는다': 5}
    # 사람이 정한 설명용 숫자. 학습해서 얻은 값이 아니다.
    weights = torch.tensor([
        [0., 0., 0.], [0.1, 0.1, 0.1], [1., 0., 0.],
        [0.8, 0.6, 0.], [0., 1., 0.], [0., 0., 1.],
    ])
    embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
    ids = torch.tensor([[vocab['고양이'], vocab['책'], 0], [vocab['강아지'], 0, 0]])
    vectors = embedding(ids)
    via_one_hot = one_hot(ids, num_classes=len(vocab)).float() @ weights
    mask = ids.ne(vocab['<pad>']).unsqueeze(-1)
    # 예제의 각 문장에는 실제 토큰이 1개 이상 있다.
    pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1)
    print('ID 모양:', tuple(ids.shape))
    print('벡터 모양:', tuple(vectors.shape))
    print('one-hot 곱과 조회 일치:', torch.allclose(vectors, via_one_hot))
    print('패딩 제외 평균:', [[round(v, 3) for v in row] for row in pooled.tolist()])
    cos = torch.nn.functional.cosine_similarity(weights[2], weights[3], dim=0)
    print('가상 고양이·강아지 cosine:', round(cos.item(), 3))
    print('미등록 단어 ID:', vocab.get('새단어', vocab['<unk>']))
    return embedding, ids, pooled


def train_word2vec():
    texts = [
        '고양이 가 물 을 마신다', '강아지 가 물 을 마신다',
        '고양이 가 집 에 산다', '강아지 가 집 에 산다',
        '고양이 가 공 을 본다', '강아지 가 공 을 본다',
        '학생 이 책 을 읽는다', '친구 가 책 을 읽는다',
        '학생 이 글 을 쓴다', '친구 가 글 을 쓴다',
        '학생 이 도서관 에 간다', '친구 가 도서관 에 간다',
    ]
    sentences = [text.split() for text in texts] * 10
    model = Word2Vec(
        sentences=sentences, vector_size=100, window=5, min_count=5,
        sg=0, workers=1, seed=42, epochs=5,
    )
    print('Word2Vec 문장 수:', len(sentences))
    print('Word2Vec 어휘 수:', len(model.wv))
    print('고양이 벡터 모양:', model.wv['고양이'].shape)
    print('가까운 단어:', [(word, round(score, 4))
                         for word, score in model.wv.most_similar('고양이', topn=3)])
    print('새단어가 어휘에 있음:', '새단어' in model.wv)
    return model


if __name__ == '__main__':
    lookup_example()
    train_word2vec()
