SFT → RM → PPO는 GPT의 다음 토큰 예측을 바탕으로 질문에 맞는 답변을 학습하는 흐름이다. 각 단계에서 쓰는 데이터와 정답이 다르다.

위 음성은 KoChatGPT 실습 전체를 다룬다. 파일에 1.1배속이 적용돼 있다. 예제의 확인 범위는 작은 숫자로 계산한 세 학습 단계의 손실이다.

## 1. 같은 질문도 단계마다 다르게 쓰인다

가상 질문을 “도서관 이용 시간을 한 문장으로 알려 줘”로 정해보자. 다음 표의 답변과 점수는 원리를 설명하기 위한 예시다.

| 단계 | 준비할 자료 | 모델이 배우는 것 |
|---|---|---|
| SFT(Supervised Fine-tuning, 지도 미세조정) | 질문과 목표 답변 | 주어진 문맥에서 목표 답변을 이어 쓰기 |
| RM(Reward Model, 보상 모델) | 같은 질문의 답변들과 선호 순서 | 더 선호되는 답변에 높은 점수 주기 |
| PPO(Proximal Policy Optimization, 근접 정책 최적화) | 질문과 현재 모델이 생성한 답변, 보상 | 보상을 고려해 생성 정책 조정하기 |

SFT는 준비된 목표 답변을 사용한다. PPO에서는 현재 모델이 답변을 생성하고 평가받는다. RM은 그 사이에서 점수를 계산한다. 세 단계에서 사용하는 데이터와 학습 신호를 이렇게 구분했다.

이 흐름은 사람의 선호 정보를 활용하는 RLHF(Reinforcement Learning from Human Feedback, 사람 피드백 기반 강화학습)를 이해하는 한 가지 방식이다. SFT, 선호 기반 보상 모델, PPO의 연결은 InstructGPT 논문에서도 볼 수 있다. 이 글은 수업에서 다룬 흐름의 복습이며 모든 언어 모델 학습 방법을 나열한 글은 아니다. [InstructGPT 논문](https://arxiv.org/abs/2203.02155)

## 2. SFT에서 질문은 읽고 답변을 채점한다

질문과 목표 답변을 한 배열로 연결한다. 질문은 답변을 만들 문맥이 되고, 답변의 다음 토큰을 맞히도록 학습한다. 답변 끝의 EOS(종료 토큰)도 목표에 포함하면 언제 끝내야 하는지 학습 신호가 생긴다.

```text
입력:   BOS  질문  구분자  답변  EOS  PAD
labels: -100 -100  -100   답변  EOS  -100
```

여기서 `-100`은 PyTorch 교차엔트로피에서 기본으로 무시하는 정답 값이다. 질문 부분을 이 값으로 바꿔도 질문 토큰은 입력에 그대로 남는다. 질문을 참고해 답변을 쓰되 질문 자체를 이어 쓰는 오차는 직접 채점하지 않는 구성이다.

모든 SFT가 반드시 답변만 채점하는 것은 아니다. 학습 목적과 구현에 따라 전체 문자열을 채점할 수도 있다. 이번에는 수업의 답변 영역 중심 학습을 따라 설명한다. 긴 질문 때문에 답변이 전부 잘려나가면 채점할 토큰도 사라지므로, 최대 길이를 적용한 뒤 남은 답변을 확인해야 한다.

## 3. 한 칸 이동과 세 가지 마스크

Causal LM(인과 언어 모델)은 각 위치에서 다음 토큰을 예측한다. 아래 자체 계산에서는 `logits[:, :-1]`과 `labels[:, 1:]`을 비교한다. Hugging Face 모델이 손실을 내부에서 계산한다면 이런 이동을 내부에서 처리할 수 있으므로 외부에서 중복 적용하지 않는다.

구분자 위치의 출력은 첫 답변 토큰과 비교된다. 그래서 labels에서 질문과 구분자 위치를 제외해도 첫 답변을 배울 수 있다. 내가 실제로 보고 싶은 것은 배열에 `-100`이 몇 개 있는지보다 어느 출력이 어느 답변 정답과 연결되는지다.

Attention Mask(어텐션 마스크)는 PAD를 구분하고, Causal Mask(인과 마스크)는 미래 토큰을 가린다. labels의 `-100`은 손실에서 채점할 위치를 정한다. 질문 영역의 손실을 껐다고 질문을 어텐션에서도 전부 가리면 답변을 만들 조건까지 잃어버릴 수 있다.

## 4. RM은 두 답변의 점수 차이를 배운다

같은 질문에 대한 `chosen`(선호 답변)과 `rejected`(덜 선호한 답변)를 준비한다. RM은 각각을 읽고 숫자 하나를 출력한다. 두 점수를 `r_chosen`, `r_rejected`라고 하면 아래 같은 쌍 비교 손실을 사용할 수 있다.

```text
RM loss = -log(sigmoid(r_chosen - r_rejected))
```

선호 답변의 점수가 더 높아질수록 손실이 작아진다. 두 점수가 2와 0인 경우와 -1과 -3인 경우는 차이가 모두 2다. 이 손실은 두 점수의 차이로 계산한다. 0점을 품질의 절대 기준으로 읽지 않는다.

후보 세 개에 순위가 있으면 세 쌍을 만들 수 있다. 원자료가 작은 순위 값을 더 좋은 답으로 정했는지도 먼저 확인한다. 후보가 저장된 순서가 곧 선호 순서라고 가정하면 반대 방향으로 학습할 수 있다.

질문 하나에서 나온 여러 쌍은 같은 데이터 분할에 둬야 한다. 쌍을 만든 뒤 아무렇게나 나누면 같은 질문이 훈련과 검증에 겹칠 수 있다. 점수가 답변의 정확성보다 길이나 말투만 따라가는지도 새 질문으로 확인한다.

## 5. PPO에서 네 모델의 역할을 나눈다

| 역할 | 하는 일 | 이번 수업 흐름에서 업데이트 |
|---|---|---|
| Actor(액터, 생성 정책) | 질문을 받아 답변을 생성 | 한다 |
| Critic(크리틱, 가치 모델) | 앞으로 받을 보상을 예상 | 한다 |
| Reward Model | 질문·답변의 선호 점수를 계산 | 고정한다 |
| Reference Model(기준 모델) | SFT 출발점의 토큰 확률을 제공 | 고정한다 |

Actor가 답변을 만들면 RM이 평가한다. Critic의 예상과 실제로 얻은 보상 등을 이용해 Advantage(어드밴티지)를 계산하고, 어떤 선택에 더 높은 확률을 줄지 정한다. 실제 구현에서는 토큰별 보상, 할인, 가치 추정 등이 연결된다. 아래 산술 예제는 그 계산을 생략하고 advantage를 직접 준다.

RM의 보상 점수층과 Critic의 가치 출력층은 모양이 비슷할 수 있지만 목적과 학습 신호가 다르다. 모델을 저장할 때도 본체만 저장했는지, 필요한 점수층까지 포함됐는지 봐야 한다. 같은 입력의 저장 전후 출력이 유지되는지 확인하면 복원이 맞는지 판단하는 데 도움이 된다.

## 6. 이전 정책과 기준 모델은 서로 다른 비교 대상이다

PPO의 확률 비율은 답변을 생성하던 **이전 정책**과 업데이트 중인 정책을 비교한다.

```text
ratio = exp(현재 log probability - 생성 당시 log probability)
objective = min(ratio * advantage,
                clip(ratio, 1-epsilon, 1+epsilon) * advantage)
```

`epsilon=0.2`라면 클리핑 구간은 0.8~1.2다. 이 목적함수는 한 번에 크게 바꾸는 이득을 제한한다. 실제 비율이 반드시 구간 안에 남도록 강제로 묶는 규칙은 아니다. Advantage의 부호에 따라 어떤 항이 선택되는지도 달라진다. [PPO 논문](https://arxiv.org/abs/1707.06347)

KL Divergence(KL 발산) 벌점은 고정된 Reference Model과 비교해 지나친 변화를 억제하는 데 사용한다. 생성 당시 정책은 학습 중 갱신되는 기준이고, SFT 기준 모델은 고정된 출발점이므로 두 비교를 섞지 않는다.

한 샘플의 `log p - log q`는 음수가 나올 수 있다. 분포 전체의 기대값으로 정의한 KL이 음수가 아니라는 성질과 샘플 하나의 로그 확률 차이를 구분한다. 이번 코드는 KL·가치 손실까지 합친 전체 PPO 구현은 아니다.

## 7. 세 손실을 작은 숫자로 계산하기

SFT에서는 어휘 6개에 같은 점수 0을 주었다. RM에서는 점수 차이 2인 두 쌍을 넣었다. PPO에서는 확률 비율 1.5·0.5와 양수·음수 advantage를 조합했다. 좋은 방향으로 이미 크게 바뀐 경우와 나쁜 방향으로 바뀐 경우를 함께 보기 위한 숫자다.

```python
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
```

실행 결과를 적어둔다.

```text
SFT labels: [[-100, -100, -100, 5, 2, -100]]
채점 토큰 수: 2
SFT loss: 1.7918, RM loss: 0.1269
확률 비율: [1.5, 0.5, 1.5, 0.5]
advantage: [1.0, 1.0, -1.0, -1.0]
클리핑 목적함수: [1.2, 0.5, -1.5, -0.8]
최소화할 policy loss: 0.1500
```

확인한 환경은 Python 3.12.9, NumPy 2.5.2, PyTorch 2.13.0 (CPU), scikit-learn 1.9.0이다. SFT·RM의 역전파로 기울기를 계산했지만 생성 모델을 학습하는 반복 루프는 없다. PPO도 손실 산술만 계산했다.

## 8. 계산 결과를 해석해 보기

SFT는 답변 토큰과 EOS, 총 2개를 채점한다. 어휘 6개의 점수가 같아서 확률은 각각 `1/6`이고 손실은 `log(6)`인 약 1.7918이다. 질문과 PAD를 포함한 전체 길이 6으로 평균낸 값이 아니다.

RM의 두 쌍은 모두 차이가 2라 손실도 같다. 점수 둘을 함께 올리거나 내려도 차이는 유지된다. 선호 점수를 읽을 때 모델과 조건이 달라진 숫자를 절대 점수처럼 비교하면 곤란한 이유를 여기서 볼 수 있다.

PPO의 첫 항은 양의 advantage에 비율 1.5를 곱한 1.5 대신 1.2를 사용한다. 마지막 항은 음의 advantage에 비율 0.5를 곱한 -0.5 대신 -0.8을 사용한다. 나머지 두 경우는 0.5와 -1.5다. 이 네 값을 평균한 목적함수에 음수를 붙여 최소화할 policy loss 0.15를 얻었다.

## 9. 보상이 올랐을 때 실제 답변도 읽는다

기본 모델, SFT 모델, PPO 모델을 비교하려면 같은 평가 질문과 생성 조건을 사용한다. 학습에 쓴 질문을 다시 보여주는 것만으로 일반화 성능을 알 수는 없다. 관련성, 사실성, 지시 준수, 불필요한 반복을 실제 문장에서 확인한다.

보상 점수가 높아져도 반복되는 표현이나 길어진 답변이 보상 모델의 빈틈을 이용한 결과일 수 있다. RM 점수와 사람이 읽은 품질을 별도로 적어두는 편이 낫겠다. 학습 데이터, 시작 모델, 토크나이저를 동시에 바꾼 실험도 어느 한 변경의 효과라고 단정하지 않는다.

개인 LLM 학습 노트와 KoChatGPT 생성 음성을 참고했다. 손실의 원리를 확인하는 코드를 직접 작성했고, 대형 모델 학습·GPU 실행·실제 답변 품질 비교는 진행하지 않았다.
