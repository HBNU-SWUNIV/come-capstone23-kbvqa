# 한밭대학교 컴퓨터공학과 KBVQA팀

**팀 구성**
- 20181796 김민준 
- 20181620 송승우
- 20202364 송지현

## <u> KBVQA</u> Project Background
- ### 필요성
  - 기존 VQA 시스템의 한계점
기존 VQA 시스템은 이미지와 질의로부터 얻는 정보만 활용해서 답변하기 때문에 외부 지식이 필요한 질문에 대해서는 답변이 어려움.
이러한 한계점은 다양한 질문에 대한 대응이 필요한 다양한 애플리케이션에 적합하지 않음.
  - 지식 베이스 활용의 필요성
효과적인 유아 교육 솔루션을 제공하기 위해서는 이미지에 보이는 정보 뿐 아니라, 외부 지식을 활용해야 함 
지식 베이스를 통해 보다 깊고, 넓은 배경 지식 전달 가능. 지식 베이스를 활용하면 정확한 과학적, 역사적 사실 제공 가능.

- ### 기존 해결책의 문제점
  - 기존 연구에서는 지식 그래프에 n-hop을 적용하여 Graph Search를 통해 지식을 습득함. 
  - 이러한 방법은 정확도가 떨어지고, 모든 질문에 대해 적합한 답변을 찾아내기 어려움.
  - 또한, 지식 그래프의 복잡성과 크기 때문에 n-hop 탐색은 계산적으로 매우 비효율적임.
  
## System Design
  - ### System Requirements
Ubuntu 20.04.4 LTS
Python 3.8.10
CUDA 11.6
GPU : NVIDIA A100 80GB x 4
CPU : AMD EPYC 7352 24-Core Processor x 48
Memory : 126G
[[requirements.txt](https://github.com/HBNU-SWUNIV/come-capstone23-kbvqa/blob/main/003%20Code/requirements.txt)]

## Case Study
01. Antol, S.; Agrawal, A.; Lu, J.; Mitchell, M.; Batra, D.; Zitnick,
C. L.; and Parikh, D. 2015. Vqa: Visual question answering.
In Proceedings of the IEEE international conference
on computer vision, 2425–2433.

> VQA 데이터셋 최초 공개. ResNet과 LSTM 기반의 모델 (외부지식 사용 X)

2. Schwenk, D.; Khandelwal, A.; Clark, C.; Marino, K.; and
Mottaghi, R. 2022. A-okvqa: A benchmark for visual
question answering using world knowledge. In Computer
Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel,
October 23–27, 2022, Proceedings, Part VIII, 146–162.
Springer.

> OK-VQA 데이터셋 공개. Wikipedia corpus를 지식으로 사용하는 VQA 데이터셋
> n-hop, hyper graph 등을 활용하는 방법 기반의 모델

3. Wang, P.; Wu, Q.; Shen, C.; Dick, A.; and van den Hengel,
A. 2017a. Explicit Knowledge-based Reasoning for Visual
Question Answering. In Proceedings of the Twenty-Sixth
International Joint Conference on Artificial Intelligence. International
Joint Conferences on Artificial Intelligence Organization.

> KVQA 데이터셋 공개. 인물(고유명사)의 정보를 포함하는 Triplet 지식 포함.
> Q-former, QQ-former의 규칙기반의 방법을 통해 외부지식 활용
---
> 본 프로젝트에서는 위의 기존 연구를 참고하여 한계점을 조사함.
> 규칙기반으로 triplet을 활용하는 한계점을 개선하여 딥러닝 기반(지식 그래프 임베딩)의 방법을 통해 triplet 외부지식 활용.
> 지식 그래프 임베딩을 활용하면, 지식 베이스가 어떤 언어로 주어지든 one-hot encoding 방법을 사용하여 임베딩 형태로 변환하기 때문에 이미지와 같이 language-independent 함.
> 이미지와 외부지식이 language-independent 하게 변환되었기 때문에, 질의에 대하여 다국어 활용 방안을 탐색. -> multilingual LLM 을 활용
  
## Conclusion
본 연구에서는 고자원 언어의 풍부한 외부 지식을 활용하여 자원이 부족한 언어에 대한 외부 지식 VQA 학습 데이터를 효과적으로 구축하는 방법을 제안했습니다. 17만 개의 한국어-영어 질의응답 쌍과 28만 개의 정보 인스턴스를 사용하여 대규모 학습 데이터를 구축했습니다. 

또한, 멀티태스킹 방식으로 VQA와 KGE 학습을 수행하는 GEL-VQA 모델을 사용하여 언어에 구애받지 않는 KG 활용 방식을 시연했습니다. 실험을 통해 제안한 BOK-VQA와 GEL-VQA를 사용한 이중 언어 VQA 모델의 수용 가능한 성능을 입증했습니다. 

이 연구 결과를 통해 다국어 훈련의 이점과 평가 접근 방식이 향후 다국어 VQA 연구를 위한 성능 지표로 사용될 수 있을 것으로 기대합니다. 그럼에도 불구하고 이 연구에서는 몇 가지 해결되지 않은 한계가 있습니다. 

첫째, 질문 구성에 활용되는 지식은 K ≥ 1 개의 정보를 포함하지만 본 연구에서는 단 하나의 지식만을 예측하고 활용합니다. 따라서 여러 지식을 복합적으로 활용해야 하는 문제를 해결하기에는 부적절합니다. 둘째, 제안된 VQA 데이터 세트의 가장 큰 장점은 다국어 확장이 가능하다는 점이지만, 새로운 언어로 문제를 만들 때 번역 및 검토 과정이 필수적이라는 한계가 있습니다.
  
## Project Outcome
- ### 2023년 KCC "지식 그래프 임베딩을 활용한 시각정보 기반 질의응답"
- ### 2023년 KCC 우수발표논문상
- ### 2023년 KCC "Categorical 정보를 활용한 시각기반 질의 생성"
- ### 2023년 컴퓨팅의실재 논문지(KCI) "(우수발표논문) 지식 그래프 임베딩을 활용한 시각정보 기반 질의응답"
- ### 2024 AAAI (심사중, 1차심사 합격)
