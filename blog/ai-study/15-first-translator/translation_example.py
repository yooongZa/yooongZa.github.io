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
