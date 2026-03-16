---
name: training-method-preference
description: 학습은 항상 Python 평가 스크립트(evaluate_10min.py)를 통해 Claude가 직접 입출력을 관찰하며 진행해야 함. /train 자동학습이 아닌 직접 평가 방식 선호.
type: feedback
---

학습 요청 시 `/train` 파이프 입력이 아니라, Python 평가 스크립트(`scripts/evaluate_10min.py`)를 실행하여 Claude가 직접 입력하고 출력을 보면서 피드백하는 방식으로 진행해야 한다.

**Why:** 자동 입력(`echo "/train" | binary`)은 데이터가 정확하지 않을 수 있고, 사용자는 Claude가 직접 관찰하며 평가하는 것을 원함. 이전에도 같은 피드백을 여러 번 줌.

**How to apply:** "학습해줘" 요청 시 항상 `python3 scripts/evaluate_10min.py`를 실행하거나, 해당 스크립트의 DURATION을 조정해서 실행할 것.
