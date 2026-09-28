import torch
from torch import nn

torch.manual_seed(42)
# 손으로 정한 가중치로 Linear 계산을 확인한다.
x = torch.tensor([[1., 2., 3.], [0., 1., -1.]])
linear = nn.Linear(3, 2)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[1., 0., -1.], [.5, .5, .5]]))
    linear.bias.copy_(torch.tensor([.5, -1.]))
    linear_output = linear(x)
print("Linear 출력:", linear_output.tolist())
print("Linear weight:", tuple(linear.weight.shape))
print("Linear 매개변수 수:", sum(p.numel() for p in linear.parameters()))

# 작은 창에서 곱하고 더하는 Conv2d 계산. 커널을 뒤집지 않는다.
image = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)
conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[1., 0.], [0., -1.]]]]))
    convolution_output = conv(image)
print("작은 Conv 출력:", convolution_output[0, 0].tolist())

# 이미지 모양의 텐서를 통과시키는 예제이며 분류 학습은 하지 않는다.
model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(start_dim=1),
    nn.Linear(4 * 4 * 4, 3),
)
model.eval()
features = torch.randn(2, 3, 8, 8)
print("CNN 입력:", tuple(features.shape))
with torch.inference_mode():
    for layer in model:
        features = layer(features)
        print(type(layer).__name__, "->", tuple(features.shape))
print("CNN 매개변수 수:", sum(p.numel() for p in model.parameters()))
