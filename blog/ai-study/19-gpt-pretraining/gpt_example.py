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
