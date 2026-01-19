# Conv-LoRA Implementation Guide

## Quick Answer / 빠른 답변

**Q: conv_lora의 구현을 찾으려면 어떻게 해야해?**

**A: Conv-LoRA의 주요 구현 위치:**

### 1. 핵심 구현 (Core Implementation)
📍 **`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`**
- **Lines 606-728**: `ConvLoRALinear` 클래스 - Conv-LoRA의 메인 구현
- **Lines 730-843**: `MoEGate` 클래스 - Mixture-of-Experts 게이팅
- **Lines 846-960**: `SparseDispatcher` 클래스 - Expert 라우팅

### 2. 사용 예제 (Usage Examples)
📍 **`examples/automm/Conv-LoRA/run_semantic_segmentation.py`**
- Line 84: `"optim.peft": "conv_lora"` - Conv-LoRA 활성화 방법
- Line 83: `"optim.lora.r": args.rank` - Rank 설정
- Line 85: `"optim.lora.conv_lora_expert_num": args.expert_num` - Expert 수 설정

### 3. 유틸리티 및 연동 (Utilities & Integration)
📍 **`multimodal/src/autogluon/multimodal/models/utils.py`**
- **Line 59**: `ConvLoRALinear` 임포트
- **Lines 485-534**: `create_adaptation()` - Conv-LoRA 레이어 생성
- **Lines 536-594**: `inject_adaptation_to_linear_layer()` - 모델에 Conv-LoRA 주입

### 4. 설정 및 상수 (Configuration & Constants)
📍 **`multimodal/src/autogluon/multimodal/constants.py`**
- **Line 266**: `CONV_LORA = "conv_lora"` 상수 정의
- **Lines 267-278**: `PEFT_ADDITIVE_STRATEGIES` 리스트

---

## File Structure / 파일 구조

```
autogluon/
├── multimodal/src/autogluon/multimodal/
│   ├── models/
│   │   ├── adaptation_layers.py    ← 🔴 MAIN IMPLEMENTATION / 주요 구현
│   │   ├── utils.py                ← Helper functions / 헬퍼 함수
│   │   └── sam.py                  ← SAM integration / SAM 통합
│   ├── constants.py                ← Constants / 상수 정의
│   └── configs/optim/default.yaml  ← Default config / 기본 설정
└── examples/automm/Conv-LoRA/
    ├── README.md                    ← Setup guide / 설정 가이드
    └── run_semantic_segmentation.py ← 🔴 USAGE EXAMPLE / 사용 예제
```

---

## Quick Start / 빠른 시작

### View the Implementation / 구현 보기
```bash
# 메인 구현 파일 열기
code multimodal/src/autogluon/multimodal/models/adaptation_layers.py

# 606줄부터 ConvLoRALinear 클래스 확인
```

### Use Conv-LoRA / Conv-LoRA 사용하기
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    hyperparameters={
        "optim.peft": "conv_lora",              # Conv-LoRA 활성화
        "optim.lora.r": 3,                      # Rank (기본값: 3)
        "optim.lora.conv_lora_expert_num": 8,   # Expert 수 (기본값: 8)
    }
)
```

---

## Code Navigation / 코드 탐색

### 1️⃣ Start Here / 여기서 시작
**`adaptation_layers.py:606`** - `ConvLoRALinear` class definition

```python
class ConvLoRALinear(nn.Linear, LoRALayer):
    """Conv-LoRA incorporated in Linear Layer."""
```

### 2️⃣ See Usage / 사용법 보기
**`examples/automm/Conv-LoRA/run_semantic_segmentation.py:84`**

```python
hyperparameters = {
    "optim.peft": "conv_lora",
    "optim.lora.r": args.rank,
    "optim.lora.conv_lora_expert_num": args.expert_num,
}
```

### 3️⃣ Understand Integration / 통합 이해하기
**`models/utils.py:515`** - How Conv-LoRA is created and injected

```python
elif "conv_lora" in peft:
    return ConvLoRALinear(
        layer.in_features,
        layer.out_features,
        r=lora_r,
        lora_alpha=lora_alpha,
        conv_lora_expert_num=kwargs["conv_lora_expert_num"],
    )
```

---

## Key Components / 주요 구성요소

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **ConvLoRALinear** | adaptation_layers.py | 606-728 | 메인 Conv-LoRA 레이어 구현 |
| **MoEGate** | adaptation_layers.py | 730-843 | Mixture-of-Experts 게이팅 메커니즘 |
| **SparseDispatcher** | adaptation_layers.py | 846-960 | Expert에게 입력 라우팅 |
| **create_adaptation()** | utils.py | 485-534 | Conv-LoRA 레이어 팩토리 함수 |
| **inject_adaptation()** | utils.py | 536-594 | 모델에 Conv-LoRA 주입 |

---

## Detailed Documentation / 상세 문서

더 자세한 정보는 다음 문서를 참고하세요:
For more details, see:
**`docs/HOW_TO_FIND_CONV_LORA_IMPLEMENTATION.md`**

---

## References / 참고문헌

**Paper:** Zhong et al., "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", ICLR 2024
- 📄 arXiv: https://arxiv.org/abs/2401.17868
- 💻 Original Code: https://github.com/autogluon/autogluon

**Related Documentation:**
- SAM Model: `multimodal/src/autogluon/multimodal/models/sam.py`
- LoRA Implementation: `adaptation_layers.py:218-309` (LoRALinear class)
- Example README: `examples/automm/Conv-LoRA/README.md`
