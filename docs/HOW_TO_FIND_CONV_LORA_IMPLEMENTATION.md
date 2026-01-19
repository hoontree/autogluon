# How to Find Conv-LoRA Implementation

This guide helps you locate and understand the Conv-LoRA (Convolutional Low-Rank Adaptation) implementation in the AutoGluon Multimodal codebase.

## Overview

Conv-LoRA is a parameter-efficient fine-tuning method that combines convolutional operations with LoRA (Low-Rank Adaptation). It's particularly useful for fine-tuning the Segment Anything Model (SAM) and other vision models.

Reference: Zhong et al., "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", ICLR 2024
https://arxiv.org/abs/2401.17868

## Main Implementation Location

The core Conv-LoRA implementation is in:

**`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`**

### ConvLoRALinear Class (Lines 606-728)

This is the main Conv-LoRA implementation that wraps a Linear layer with:
- Low-rank decomposition matrices (lora_A and lora_B)
- Mixture-of-Experts (MoE) with convolutional experts
- Multiple upsampling ratios for different scales

Key components:
```python
class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer.
    
    Parameters:
    - in_features: input dimension
    - out_features: output dimension  
    - r: rank of low-rank decomposition
    - lora_alpha: scaling factor
    - conv_lora_expert_num: number of experts in MoE-Conv
    """
```

## Supporting Classes

The Conv-LoRA implementation uses these helper classes (also in `adaptation_layers.py`):

1. **MoEGate** (Lines 730-843): Mixture-of-Experts gating mechanism
2. **SparseDispatcher** (Lines 846-960): Routes inputs to different experts

## Related Files

### Constants
**`multimodal/src/autogluon/multimodal/constants.py`**
- Line 266: `CONV_LORA = "conv_lora"` constant definition
- Lines 267-278: `PEFT_ADDITIVE_STRATEGIES` list includes `CONV_LORA`

### Model Utilities
**`multimodal/src/autogluon/multimodal/models/utils.py`**
- Line 59: Import statement for `ConvLoRALinear`
- Lines 485-534: `create_adaptation()` function creates Conv-LoRA layers
- Lines 536-594: `inject_adaptation_to_linear_layer()` injects Conv-LoRA into models

### SAM Model Integration
**`multimodal/src/autogluon/multimodal/models/sam.py`**
- Uses Conv-LoRA for efficient fine-tuning of SAM models

### Configuration
**`multimodal/src/autogluon/multimodal/configs/optim/default.yaml`**
- Contains default hyperparameters for conv_lora

## Example Usage

### Location
**`examples/automm/Conv-LoRA/`**

This directory contains:
- `README.md`: Setup and usage instructions
- `run_semantic_segmentation.py`: Training script demonstrating Conv-LoRA usage
- `prepare_semantic_segmentation_datasets.py`: Dataset preparation script

### Basic Usage Example

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    validation_metric="iou",
    hyperparameters={
        "optim.peft": "conv_lora",  # Enable Conv-LoRA
        "optim.lora.r": 3,  # Rank of LoRA decomposition
        "optim.lora.conv_lora_expert_num": 8,  # Number of experts
    },
    path="./my_model"
)

predictor.fit(train_data=train_df)
```

## Key Hyperparameters

When using Conv-LoRA, configure these parameters:

- **`optim.peft`**: Set to `"conv_lora"` to enable Conv-LoRA
- **`optim.lora.r`**: Rank of low-rank decomposition (default: 3)
- **`optim.lora.alpha`**: Scaling factor (default: 8)
- **`optim.lora.conv_lora_expert_num`**: Number of convolutional experts (default: 8)
- **`optim.lora.module_filter`**: Apply Conv-LoRA only to filtered modules
- **`optim.lora.filter`**: Apply Conv-LoRA only to filtered layers

## How Conv-LoRA Works

1. **Low-Rank Decomposition**: Instead of updating all weights, Conv-LoRA uses two smaller matrices (A and B) where:
   - Original weight update: ΔW (large)
   - Conv-LoRA: ΔW = B @ A (smaller)

2. **Convolutional Experts**: Multiple Conv2d layers with different upsampling ratios process features at different scales

3. **Mixture-of-Experts Gating**: A learned gate determines which expert(s) to use for each input

4. **Sparse Dispatching**: Efficiently routes inputs to selected experts based on gate values

## Architecture Flow

```
Input → Linear(frozen) → LoRA Branch → MoE Gate → Dispatch to Experts
                              ↓                          ↓
                         lora_A (r×d)              Conv Experts
                              ↓                          ↓
                         Conv2d 3×3              Different scales
                              ↓                          ↓
                         lora_B (d×r)            Combine outputs
                              ↓                          ↓
                              └──────── Sum ────────────┘
                                        ↓
                                     Output
```

## Tests

To verify your understanding or test Conv-LoRA functionality:

**`multimodal/tests/unittests/others/test_semantic_segmentation.py`**

This file contains unit tests that exercise Conv-LoRA with semantic segmentation tasks.

## Additional Resources

- See the Conv-LoRA paper for theoretical background
- Check `examples/automm/Conv-LoRA/README.md` for hands-on examples
- Refer to `adaptation_layers.py` for implementation details

## Quick File Reference

| File | Purpose |
|------|---------|
| `models/adaptation_layers.py` | Main Conv-LoRA implementation |
| `models/utils.py` | Conv-LoRA injection utilities |
| `models/sam.py` | SAM model with Conv-LoRA support |
| `constants.py` | Conv-LoRA constants |
| `examples/automm/Conv-LoRA/` | Usage examples |
| `configs/optim/default.yaml` | Default hyperparameters |

---

**Korean Translation / 한국어 번역:**

Conv-LoRA 구현을 찾으려면:
1. 주요 구현: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py` (606-728줄)
2. 사용 예제: `examples/automm/Conv-LoRA/run_semantic_segmentation.py`
3. 설정 상수: `multimodal/src/autogluon/multimodal/constants.py` (266줄)

Conv-LoRA는 세그먼트 애니띵 모델(SAM)을 효율적으로 파인튜닝하기 위한 파라미터 효율적 방법입니다.
