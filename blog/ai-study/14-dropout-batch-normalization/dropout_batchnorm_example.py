import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)
values = torch.ones(12)
dropout = nn.Dropout(p=0.5)
dropout.train()
with torch.no_grad():
    dropped = dropout(values)
dropout.eval()
with torch.inference_mode():
    passed = dropout(values)
print("Dropout train:", dropped.tolist())
print("Dropout eval:", passed.tolist())
print("Dropout 학습 매개변수 수:", sum(p.numel() for p in dropout.parameters()))

x = torch.tensor([[2., 4.], [4., 8.], [6., 12.], [8., 16.]])
bn = nn.BatchNorm1d(2)  # eps=1e-5, momentum=0.1, affine=True
bn.train()
with torch.no_grad():
    train_output = bn(x)  # no_grad 안에서도 train 모드의 통계는 갱신된다.
print("BatchNorm train 출력:\n", train_output)
print("배치 평균:", x.mean(dim=0))
print("배치 분산(correction=0):", x.var(dim=0, correction=0))
print("running_mean:", bn.running_mean)
print("running_var:", bn.running_var)
bn.eval()
with torch.inference_mode():
    eval_output = bn(x)
print("BatchNorm eval 첫 행:", eval_output[0])
print("학습 매개변수:", [name for name, _ in bn.named_parameters()])
print("eval 후 관찰한 배치 수:", bn.num_batches_tracked.item())
