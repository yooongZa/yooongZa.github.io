"""작은 잔차 블록: CPU 실행, 모델/데이터 다운로드 없음."""
import torch
from torch import nn


class SmallResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, 8)
    same = SmallResidualBlock(4, 4).eval()
    down = SmallResidualBlock(4, 8, stride=2).eval()
    with torch.inference_mode():
        print('입력:', tuple(x.shape))
        print('identity 출력:', tuple(same(x).shape))
        print('projection 출력:', tuple(down(x).shape))
        print('projection 경로:', tuple(down.shortcut(x).shape))
        # 마지막 BN의 gamma/beta를 0으로 두어 F(x)=0으로 만든다.
        same.main[-1].weight.zero_()
        same.main[-1].bias.zero_()
        print('F=0이면 ReLU(x):', torch.allclose(same(x), torch.relu(x)))
        print('양수 입력이면 x:', torch.allclose(same(x.abs()), x.abs()))

    # 덧셈 지점만 보는 별도의 스칼라 예제다.
    scalar = torch.tensor(2.0, requires_grad=True)
    plain = 0.1 * scalar
    residual = 0.1 * scalar + scalar
    print('plain 기울기:', round(torch.autograd.grad(plain, scalar)[0].item(), 1))
    print('residual 기울기:', round(torch.autograd.grad(residual, scalar)[0].item(), 1))


if __name__ == '__main__':
    main()
