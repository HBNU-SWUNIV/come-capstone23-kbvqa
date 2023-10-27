# 한밭대학교 컴퓨터공학과 KBVQA팀

**팀 구성**
- 20181796 김민준 
- 20181620 송승우
- 20202364 송지현

## <u> KBVQA</u> Project Background
- ### 필요성
  - 기존 한국어 기반 대규모 멀티모달 언어모델 및 언어모델의 한계점
최근 대규모 언어모델(LLM)이 급속도로 발전함에 따라, 멀티모달분야에서도 대규모 언어모델의 활용 연구가 급격히 발전하고 있다. 하지만, 한국어 기반의 대규모 멀티모달모델은 영어 데이터를 기계번 역하여사용하기 때문에 (1) 한국어 능력이 떨어지고, (2) 영어 뉘앙스의 답변을 생성한다.
  - 번역 데이터셋을 활용한 instruction tuning
한국어-영어 번역 데이터셋을 통한 instruction tuning을 진행하여 한국어 어휘력 및 지식 강화및 GPT-4를 통한 한국어 visual instruction-following 데이터를 직접 생성하여 자연스러 운 한국어 응답을 학습한다.

- ### 기존 해결책의 문제점
  - 기존 연구에서는 한국어 데이터셋의 부족을 기계번역을 통해 해결하였다.
  - 기계번역을 통한 데이터 구축은, 모델이 자연스럽지 않은 영어 뉘앙스의 답변을 생성한다.
  
## System Design
  - ### System Requirements
Ubuntu 20.04.4 LTS

Python 3.8.10

CUDA 11.6

GPU : NVIDIA A100 80GB x 4

CPU : AMD EPYC 7352 24-Core Processor x 48

Memory : 126G

[[requirements.txt](https://github.com/HBNU-SWUNIV/come-capstone23-kbvqa/blob/main/003%20Code/requirements.txt)]

  - ### Model Architecture
<img src="https://github.com/HBNU-SWUNIV/come-capstone23-kbvqa/assets/72269271/d9c35a88-f77f-46bb-883e-7a4285f10458" width="500">


## Case Study

Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae. 2023.

**Visual Instruction Tuning(NeurIPS).**

> 기존의 LLM에서 활용하던 Instruction-tuning을 이미지로 확장하여 ChatGPT를 활용한 데이터 생성 및 visual instruction tuning.

Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi. 2023.

**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

> Q-Former를 제안하여 사전학습된 이미지 인코더와 언어모델을 효율적으로 활용
> 적은 파라미터 수로 학습 가능한 Q-Former 제안

Wenliang Dai and Junnan Li and Dongxu Li and Anthony Meng Huat Tiong and Junqi Zhao and Weisheng Wang and Boyang Li and Pascale Fung and Steven Hoi. 2023

**InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**

> 제안하는 BLIP-2 구조에 visual instruction tuning을 접목.
> 여러가지 Academical task를 활용하여 visual instruction uning 진행

---
> 본 프로젝트에서는 위의 기존 연구를 참고하여 한계점을 조사함.
> 기존 연구는 모두 영어 데이터를 활용하여 진행되었으며, 한국어는 데이터셋의 부족으로 접근이 어려움
> 본 프로젝트에서는 기계 번역이 아닌, GPT-4를 통한 자연스러운 한국어 문장 생성으로 데이터셋 부족을 해결
> 한-영 번역 데이터로 LLM에 한국어 지식을 학습시킴
  
## Conclusion


![image](https://github.com/HBNU-SWUNIV/come-capstone23-kbvqa/assets/72269271/bb3ff07b-d345-4aeb-9baa-7f09ce735f19)

![image](https://github.com/HBNU-SWUNIV/come-capstone23-kbvqa/assets/72269271/5db8a4da-46c0-46fd-8b97-3b6ea22ce975)

- 본 프로젝트에서는 한-영 번역 데이터셋을 통해 영어 데이터로 학습된 LLM이 한국어 단어 및 지식을 학습하도록 했다.
- 또한, 이미지-텍스트 paired 데이터를 통해 한국어, 영어 모두 visual instruction tuning을 진행했다.
  
## Project Outcome
- ### 2023년 정보과학회 학술대회 KCC 논문 3건 출판(2건 우수 발표 논문상 수상) 
- ### 2023년 컴퓨팅의실재 논문지(KCI) 2건 출판
- ### AAAI 2024 (심사중, 1차심사 합격)
- ### 2023년 SW중심대학 우수작품 선정
