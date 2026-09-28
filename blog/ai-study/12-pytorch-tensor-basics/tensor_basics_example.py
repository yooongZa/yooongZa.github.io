import numpy as np
import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("입력:", tuple(x.shape), x.dtype, x.device)
print("열별 평균:", x.mean(dim=0).tolist())
print("행별 합:", x.sum(dim=1).tolist())
print("행렬 곱:", (x @ torch.tensor([1., 0., -1.])).tolist())
print("축 순서 변경:", tuple(x.permute(1, 0).shape))
print("새 축으로 쌓기:", tuple(torch.stack([x, x]).shape))
print("기존 축으로 연결:", tuple(torch.cat([x, x], dim=0).shape))

# 직접 만든 배열로 공유와 복사의 차이만 관찰한다.
array = np.array([1., 2.], dtype=np.float32)
shared = torch.from_numpy(array)
copied = torch.tensor(array)
array[0] = 9
print("NumPy 수정 후 공유 / 복사:", shared.tolist(), copied.tolist())

w = torch.tensor(2.0, requires_grad=True)
(w ** 2).backward()
print("첫 backward의 w / grad:", w.item(), w.grad.item())
(w ** 2).backward()  # 새 forward를 계산해도 기존 grad에는 더해진다.
print("두 번째 backward의 grad:", w.grad.item())
w.grad = None
loss = (w * 3 - 4) ** 2
loss.backward()
print("새 손실 / 기울기:", loss.item(), w.grad.item())
with torch.no_grad():
    w -= 0.1 * w.grad
print("수동 갱신 후 w:", round(w.item(), 4))
