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
