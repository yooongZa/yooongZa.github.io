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
