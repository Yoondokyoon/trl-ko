<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_banner_dark.png">
</div>

# TRL - Transformer Reinforcement Learning
> 강화 학습을 적용한 전체 스택 트랜스포머 언어 모델.

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/trl/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/trl/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/trl/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/trl.svg">
    </a>
</p>

이것은 무엇인가?
<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/TRL-readme.png">
</div>


`trl`은 변압기 언어 모델과 안정적인 확산 모델을 강화 학습으로 훈련시키기 위한 전체 스택 라이브러리입니다. 이는 Supervised Fine-tuning 단계(SFT), Reward Modeling 단계(RM)에서 Proximal Policy Optimization (PPO) 단계까지 다양한 도구를 제공합니다. 이 라이브러리는 🤗 Hugging Face의 transformers 라이브러리를 기반으로 하므로, 사전 훈련된 언어 모델은 transformers를 통해 직접 로드할 수 있습니다. 현재 대부분의 디코더 아키텍처와 인코더-디코더 아키텍처가 지원됩니다. 예제 코드 조각과 이 도구들을 실행하는 방법은 문서나 `examples/` 폴더를 참조하십시오.


**Highlights:**

- [`SFTTrainer`]: 사용자 지정 데이터셋에서 언어 모델이나 어댑터를 쉽게 미세 조정할 수 있도록 하는 트랜스포머의 트레이너 주변의 가벼우며 친숙한 래퍼입니다.
- [`RewardTrainer`]: 인간의 선호도(보상 모델링)에 따라 언어 모델을 쉽게 미세 조정할 수 있도록 하는 트랜스포머의 트레이너 주변의 가벼운 래퍼입니다.
- [`PPOTrainer`]: 언어 모델을 위한 PPO 트레이너로, 언어 모델을 최적화하기 위해 (질문, 응답, 보상) 삼중항만 필요합니다.
- [`AutoModelForCausalLMWithValueHead & AutoModelForSeq2SeqLMWithValueHead`]: 강화 학습에서 값 함수로 사용할 수 있는 각 토큰에 대한 추가 스칼라 출력을 가진 트랜스포머 모델입니다.
- [예시]: BERT 감정 분류기로 긍정적인 영화 리뷰를 생성하기 위해 GPT2를 훈련시키기, 어댑터만을 사용한 전체 RLHF, GPT-j를 덜 독성 있게 훈련시키기, Stack-Llama 예시 등.

## PPO 작동 방식
언어 모델을 PPO를 통해 미세 조정하는 것은 대략 세 단계로 구성됩니다:

1. **롤아웃**: 언어 모델은 질문을 바탕으로 반응이나 연속을 생성합니다. 이는 문장의 시작일 수 있습니다.
2. **평가**: 질문과 반응은 함수, 모델, 인간의 피드백 또는 그들의 조합 등으로 평가됩니다. 중요한 것은 이 과정이 각 질문/반응 쌍에 대한 스칼라 값을 산출해야 한다는 것입니다.
3. **최적화**: 이것은 가장 복잡한 부분입니다. 최적화 단계에서 질문/반응 쌍은 시퀀스 내의 토큰의 로그-확률을 계산하는 데 사용됩니다. 이는 훈련 중인 모델과 보통 미세 조정 전의 사전 훈련된 모델인 참조 모델로 수행됩니다. 두 출력 사이의 KL-발산은 생성된 반응이 참조 언어 모델에서 너무 멀리 벗어나지 않도록 추가 보상 신호로 사용됩니다. 그런 다음 활성 언어 모델은 PPO로 훈련됩니다.

아래의 스케치에서 이 과정을 설명합니다:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_overview.png" width="800">
<p style="text-align: center;"> <b>Figure:</b> Sketch of the workflow. </p>
</div>

## Installation

### Python package
Install the library with pip:
```bash
pip install trl
```

### From source
If you want to run the examples in the repository a few additional libraries are required. Clone the repository and install it with pip:
```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install .
```

If you wish to develop TRL, you should install in editable mode:
```bash
pip install -e .
```

## How to use

### `SFTTrainer`

This is a basic example on how to use the `SFTTrainer` from the library. The `SFTTrainer` is a light wrapper around the `transformers` Trainer to easily fine-tune language models or adapters on a custom dataset.

```python
# imports
from datasets import load_dataset
from trl import SFTTrainer

# get dataset
dataset = load_dataset("imdb", split="train")

# get trainer
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

# train
trainer.train()
```

### `RewardTrainer`

This is a basic example on how to use the `RewardTrainer` from the library. The `RewardTrainer` is a wrapper around the `transformers` Trainer to easily fine-tune reward models or adapters on a custom preference dataset.

```python
# imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
```

### `PPOTrainer`

This is a basic example on how to use the `PPOTrainer` from the library. Based on a query the language model creates a response which is then evaluated. The evaluation could be a human in the loop or another model's output.

```python
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = PPOConfig(
    batch_size=1,
)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

## References

### 근접 정책 최적화
PPO 구현은 대부분 D. Ziegler 외의 논문 **"인간의 선호도에서 언어 모델 미세 조정"**에서 소개된 구조를 따릅니다. [논문, 코드].

### 언어 모델
언어 모델은 🤗 Hugging Face의 transformers 라이브러리를 사용합니다.

## Citation

```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```
