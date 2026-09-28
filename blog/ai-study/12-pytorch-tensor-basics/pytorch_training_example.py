import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(1)
# 작은 CPU 실습. 이 규칙으로 만든 데이터의 점수이며 실제 문제 성능은 아니다.
X = torch.randn(128, 2, dtype=torch.float32)
y = (X[:, 0] - 0.5 * X[:, 1] > 0).long()
order = torch.randperm(len(X))
train_idx, test_idx = order[:96], order[96:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(42),
)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():
    initial_loss = loss_fn(model(X_train), y_train).item()
for epoch in range(100):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.inference_mode():
    train_loss = loss_fn(model(X_train), y_train).item()
    test_logits = model(X_test)
    test_loss = loss_fn(test_logits, y_test).item()
    test_accuracy = (test_logits.argmax(dim=1) == y_test).float().mean().item()
print("Train / Test:", tuple(X_train.shape), tuple(X_test.shape))
print("입력 / 정답 dtype:", X.dtype, y.dtype)
print("에포크당 배치 / 전체 step:", len(loader), 100 * len(loader))
print("출력 logits:", tuple(test_logits.shape))
print(f"학습 Loss: {initial_loss:.4f} -> {train_loss:.4f}")
print(f"테스트 Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}")
