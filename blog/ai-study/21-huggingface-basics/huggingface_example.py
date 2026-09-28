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
