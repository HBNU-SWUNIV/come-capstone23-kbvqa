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
  
## Conclusion
  - ### OOO
  - ### OOO
  
## Project Outcome
- ### 2023년 KCC "지식 그래프 임베딩을 활용한 시각정보 기반 질의응답" (우수발표논문)
- ### 2023년 KCC "Categorical 정보를 활용한 시각기반 질의 생성"
- ### 2023년 컴퓨팅의실재 논문지(KCI) "(우수발표논문) 지식 그래프 임베딩을 활용한 시각정보 기반 질의응답"
- ### 2024 AAAI (심사중, 1차심사 합격)
