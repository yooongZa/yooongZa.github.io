"""SFT·RM·PPO의 손실 산술만 계산한다. 생성 모델 학습 루프는 없다."""
import torch
from torch.nn import functional as F


def sft_example():
    # BOS=1, 질문=3, 구분자=4, 답변=5, EOS=2, PAD=0
    ids = torch.tensor([[1, 3, 4, 5, 2, 0]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[ids == 0] = -100
    logits = torch.zeros(1, 6, 6, requires_grad=True)
    # 직접 CE를 계산하므로 여기서 한 번만 다음 토큰에 맞춘다.
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1))
    return labels, logits, loss


def ppo_objective(new_log_prob, old_log_prob, advantage, epsilon=0.2):
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    return ratio, torch.minimum(unclipped, clipped)


def main():
    labels, logits, sft_loss = sft_example()
    sft_loss.backward()
    chosen = torch.tensor([2.0, -1.0], requires_grad=True)
    rejected = torch.tensor([0.0, -3.0], requires_grad=True)
    rm_loss = -F.logsigmoid(chosen - rejected).mean()
    rm_loss.backward()
    old = torch.log(torch.tensor([0.2, 0.2, 0.2, 0.2]))
    new = torch.log(torch.tensor([0.3, 0.1, 0.3, 0.1]))
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    ratio, objective = ppo_objective(new, old, advantage)
    rounded = lambda x: [round(value, 3) for value in x.tolist()]
    print("SFT labels:", labels.tolist())
    print("채점 토큰 수:", labels[:, 1:].ne(-100).sum().item())
    print(f"SFT loss: {sft_loss.item():.4f}, RM loss: {rm_loss.item():.4f}")
    print("확률 비율:", rounded(ratio))
    print("advantage:", rounded(advantage))
    print("클리핑 목적함수:", rounded(objective))
    print(f"최소화할 policy loss: {-objective.mean().item():.4f}")


if __name__ == "__main__":
    main()
