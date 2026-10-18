Things to do

- [ ] **텍스트 생성 기능**  
  [TransformerLM (line 41)](/E:/Program/assignment1-basics/cs336_basics/transformer_LM.py:41)은 다음 토큰의 점수인 logits까지 반환해. 여기서 토큰을 선택하고 다시 입력하는 생성 루프, EOS 종료, temperature/top-p 처리는 아직 없어. KV cache는 준비되어 있어.

- [ ] **검증 데이터로 성능 평가**  
  [학습 루프 (line 37)](/E:/Program/assignment1-basics/cs336_basics/train_model.py:37)는 training loss만 출력해. Validation loss나 perplexity를 계산하지 않아서, 일반화 성능이나 과적합 여부를 확인하는 부분이 빠져 있어. 학습률 스케줄 함수도 있지만 학습에는 연결되지 않았어.

- [ ] **체크포인트에서 학습 이어가기**  
  [train_model.py (line 79)](/E:/Program/assignment1-basics/cs336_basics/train_model.py:79)에 저장·로드는 구현되어 있어. 다만 `train()`은 항상 모델과 optimizer를 새로 만들어서, 불러온 상태로 학습을 이어가는 경로가 없어. 별도의 [checkpoint.py (line 27)](/E:/Program/assignment1-basics/cs336_basics/checkpoint.py:27)는 저장 함수가 미완성이고 로드는 `pass`야.

- [ ] **테스트 연결**  
  [tests/adapters.py (line 13)](/E:/Program/assignment1-basics/tests/adapters.py:13)의 **21개 함수가 모두 `NotImplementedError`**야. 구현한 모듈을 과제 테스트에서 검증할 수 있도록 연결해야 해. 또 [데이터 테스트 (line 10)](/E:/Program/assignment1-basics/tests/test_training_data.py:10)가 가져오는 `cs336_basics.training_data` 소스 파일이 현재 없어.

- [ ] **별도 데이터로더의 배치 처리 오류**  
  [StoryDataLoader (line 76)](/E:/Program/assignment1-basics/cs336_basics/dataloader.py:76)는 첫 배치를 반환한 뒤 리스트를 비우지 않아. 이후 길이가 `batch_size`를 초과하면서 다음 배치를 반환하지 못해. 현재 노트북은 다른 DataLoader 경로를 사용하고 있어.