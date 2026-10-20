모델 설정 사용법

모델 파라미터는 `source/model_config/config.py`에서 설정합니다. 기본값은 8개 블록,
임베딩 512, FFN 2048, attention head 8개, context length 128입니다.
설정 자료형은 `source/model_config/config_data_strucrue.py`에 정의되어 있습니다.
`main.ipynb` 첫 셀을 실행하면 `config.py`를 다시 읽어 전처리와 학습에 같은 설정을 전달합니다.
데이터 경로는 `DatasetConfig`로 지정하며, 데이터는 `load()` 호출 시 읽습니다.

```python
import torch

from source.model_config.config_data_strucrue import ModelConfig, PrenormTransformerConfig
from source.transformer_LM import TransformerLM
from source.train_model import train

config = ModelConfig(
    context_length=128,
    device="cpu",
    dtype=torch.float32,
    layers=[
        PrenormTransformerConfig(
            embed_dim=128, hidden_dim=512, out_dim=128, num_heads=4,
            device="cpu", dtype=torch.float32,
        )
        for _ in range(2)
    ],
)
model = TransformerLM(config)
# train_dataset의 input_ids 길이는 config.context_length + 1이어야 합니다.
# model, optimizer, checkpoint = train(train_dataset, config=config, num_epoch=3)
```

`layers` 목록의 길이가 블록 수입니다. 블록별 FFN 차원과 head 수는 다르게
지정할 수 있습니다. 현재 residual 구조에서는 모든 블록의 `embed_dim`과
`out_dim`이 같아야 하며, head당 차원은 RoPE를 위해 짝수여야 합니다.
활성화 함수는 기존에 구현된 `swiglu`만 지원합니다.
`TokenizerInfo`를 바꿀 때는 encoding 이름, vocabulary 크기, EOS 정보를 함께 맞춰주세요.
체크포인트에는 실제 모델 설정이 일반 dict로 저장되며, 로드 시 같은 구조를 복원합니다.
미구현 체크포인트 함수, optimizer 함수, 평가 및 학습 재개 기능은 그대로 남겨두었습니다.


두 config의 `device` 기본값은 `config_data_strucrue.py`의 CUDA 감지 결과(`"cuda"` 또는 `"cpu"`)이며,
`dtype` 기본값은 기존 학습 설정과 같은 `torch.float32`입니다. 두 값 모두 `None`을 허용하지 않습니다.
각 `PrenormTransformerConfig`에도 `device="cuda:0"`, `dtype=torch.bfloat16`처럼 지정할 수 있습니다.
각 config는 자체 기본값을 사용합니다. 모델과 모든 블록의 장치·dtype을 통일하려면
각각 같은 값을 지정하세요. 모델 설정 변경이 이미 생성된 블록 설정을 덮어쓰지는 않습니다.
블록 사이의 텐서는 해당 장치와 dtype으로 변환되며, KV 캐시도 블록 설정을 따릅니다.
임베딩과 최종 출력은 모델 설정을 사용합니다.
모델과 블록 생성자는 config의 `device`, `dtype`만 사용합니다.
원본 config 객체는 변경하지 않습니다.
`generate_text()`는 모델의 장치에 정수 토큰(`torch.long`)을 준비합니다.
체크포인트에는 dtype을 문자열로 저장하고 로드할 때 `torch.dtype`으로 복원합니다.
`train_model.load_checkpoint()`는 체크포인트에 저장된 config를 사용합니다.

Things to do

- [x] **텍스트 생성 기능**  
  [TransformerLM (line 41)](/E:/Program/assignment1-basics/cs336_basics/transformer_LM.py:41)은 다음 토큰의 점수인 logits까지 반환해. 여기서 토큰을 선택하고 다시 입력하는 생성 루프, EOS 종료, temperature/top-p 처리는 아직 없어. KV cache는 준비되어 있어.

- [ ] **검증 데이터로 성능 평가**  
  [학습 루프 (line 37)](/E:/Program/assignment1-basics/cs336_basics/train_model.py:37)는 training loss만 출력해. Validation loss나 perplexity를 계산하지 않아서, 일반화 성능이나 과적합 여부를 확인하는 부분이 빠져 있어. 학습률 스케줄 함수도 있지만 학습에는 연결되지 않았어.

- [ ] **체크포인트에서 학습 이어가기**  
  [train_model.py (line 79)](/E:/Program/assignment1-basics/cs336_basics/train_model.py:79)에 저장·로드는 구현되어 있어. 다만 `train()`은 항상 모델과 optimizer를 새로 만들어서, 불러온 상태로 학습을 이어가는 경로가 없어. 별도의 [checkpoint.py (line 27)](/E:/Program/assignment1-basics/cs336_basics/checkpoint.py:27)는 저장 함수가 미완성이고 로드는 `pass`야.

- [ ] **별도 데이터로더의 배치 처리 오류**  
  [StoryDataLoader (line 76)](/E:/Program/assignment1-basics/cs336_basics/dataloader.py:76)는 첫 배치를 반환한 뒤 리스트를 비우지 않아. 이후 길이가 `batch_size`를 초과하면서 다음 배치를 반환하지 못해. 현재 노트북은 다른 DataLoader 경로를 사용하고 있어.