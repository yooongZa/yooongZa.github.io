"""LSTM의 상태 형태와 Attention 가중합을 확인하는 자체 예제."""
import torch
from torch import nn


def main():
    torch.manual_seed(42)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    embedding = nn.Embedding(12, 4)
    lstm = nn.LSTM(4, 6, batch_first=True)
    vectors = embedding(ids)
    outputs, (hidden, cell) = lstm(vectors)
    print("embedding:", tuple(vectors.shape))
    print("outputs:", tuple(outputs.shape))
    print("hidden / cell:", tuple(hidden.shape), tuple(cell.shape))
    print("last output equals hidden:", torch.allclose(outputs[:, -1], hidden[-1]))

    # 학습된 점수 대신 원하는 비중의 log를 넣어 가중합만 확인한다.
    scores = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]))
    values = torch.tensor([[2.0, 0.0], [0.0, 4.0], [6.0, 2.0]])
    weights = scores.softmax(dim=-1)
    contexts = weights @ values
    print("weight sums:", [round(x, 4) for x in weights.sum(-1).tolist()])
    print("contexts:", [[round(x, 4) for x in row] for row in contexts.tolist()])


if __name__ == "__main__":
    main()
