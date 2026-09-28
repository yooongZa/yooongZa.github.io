"""MLM의 세 치환 방식과 문장 순서 분류. 15% 무작위 추출은 생략한다."""
import torch
from torch import nn
from transformers import BertConfig, BertModel

PAD, UNK, BOS, EOS, SEP, CLS, MASK = range(7)


def make_batch():
    # A=[7,8], B=[9,10]. 두 번째 행은 B 다음에 A를 둔다.
    original = torch.tensor([[CLS, 7, 8, SEP, 9, 10, SEP, PAD],
                             [CLS, 9, 10, SEP, 7, 8, SEP, PAD]])
    corrupted = original.clone()
    labels = torch.full_like(original, -100)
    # 세 갈래를 하나씩 보여 주기 위해 위치를 직접 고른다.
    for position in [1, 2, 4]:
        labels[:, position] = original[:, position]
    corrupted[:, 1] = MASK
    corrupted[:, 2] = 11  # 원 토큰과 다른 일반 토큰
    # 위치 4는 원래 토큰을 유지하면서 채점한다.
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]]).repeat(2, 1)
    order_labels = torch.tensor([1, 0])  # 1=원래 순서, 0=뒤집은 순서
    return original, corrupted, labels, segments, order_labels


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=16, pad_token_id=PAD,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.encoder = BertModel(config, add_pooling_layer=False)
        # 원래 BERT의 MLM 변환층을 생략한 학습용 선형 헤드다.
        self.mlm = nn.Linear(16, 16)
        self.order = nn.Linear(16, 2)

    def forward(self, ids, segments):
        hidden = self.encoder(input_ids=ids, attention_mask=ids.ne(PAD),
                              token_type_ids=segments).last_hidden_state
        return self.mlm(hidden), self.order(hidden[:, 0])


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    original, corrupted, labels, segments, order_labels = make_batch()
    model = TinyBert()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mlm, order = model(corrupted, segments)
    mlm_loss = nn.functional.cross_entropy(mlm.reshape(-1, 16), labels.reshape(-1))
    order_loss = nn.functional.cross_entropy(order, order_labels)
    optimizer.zero_grad()
    (mlm_loss + order_loss).backward()
    optimizer.step()
    print("원래 입력:", original[0].tolist())
    print("가공한 입력:", corrupted[0].tolist())
    print("MLM 정답:", labels[0].tolist())
    print("MLM/순서 logits:", tuple(mlm.shape), tuple(order.shape))
    print("MLM 채점 위치 수:", labels.ne(-100).sum().item())
    print(f"MLM loss: {mlm_loss.item():.4f}, 순서 loss: {order_loss.item():.4f}")


if __name__ == "__main__":
    main()
