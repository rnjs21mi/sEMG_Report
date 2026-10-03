# sEMG 손바닥 인증 재현 및 성능 비교 (CWT + DenseNet161)

문손잡이 회전 중 손바닥 2채널 sEMG로 등록된 5명 중 누구인지 맞히는 closed-set 사용자 식별 실험이다.
재현 대상 논문: Shin, Y., Kim, J. & Choi, S.-I. (2026). Palm sEMG-based user identification during doorknob rotation using a convolutional neural network. *Scientific Reports* 16, 22244. https://doi.org/10.1038/s41598-026-46294-3

> ⚠ `[ ]` 와 `[…]` 표시는 실제 실행 결과·본인 판단으로 교체한다. 실행하지 않은 값은 적지 않는다.

---

## 1. 코드 설명

### 사용한 데이터 및 전처리 방법
- 데이터: 논문 공개 데이터 (5명 × 50시행, 2채널, 1,000 Hz, 시행당 3초 = 3000×2). 출처: https://github.com/sea3551/palm-sEMG-doorknob-filtered (CC BY 4.0). 폴더명 A~E를 클래스 라벨로 사용한다.
- 공개 데이터는 이미 60 Hz 노치 + 20~500 Hz 대역통과가 적용된 상태이다. 본 코드는 60 Hz 노치(Q=30) → 20~499 Hz 4차 Butterworth 대역통과를 `filtfilt`로 **다시 적용**한다(이중 필터링. 500 Hz는 나이퀴스트 한계와 같아 499 Hz 사용). 필터 동작은 60 Hz 사인파를 얹은 합성 검증으로 확인했다(약 38 dB 감쇠).
- 분할: **시행(파일) 단위로 먼저 8:2 계층 분할**(학습 200시행 / 테스트 50시행, 겹침 0)한 뒤, 각 집합 안에서 윈도우를 생성한다 (데이터 누수 방지). 분할 시드 42, 학습 3,800윈도우 / 테스트 950윈도우(클래스당 190).
- 윈도우 300 ms, 50% 오버랩 (시행당 19개) → 윈도우 단위 min–max 정규화(두 채널을 합쳐서 정규화하므로 진폭이 작은 Ch2는 상대적으로 눌린다) → CWT(스케일 1~32, Morlet `morl`, 절댓값) → 입력 텐서 3×32×300 (채널 1, 채널 2, 두 채널의 평균을 3채널로 사용).

### 사용한 AI/ML 모델
| 모델 | 설명 |
|---|---|
| DenseNet161 | 제안 모델. torchvision DenseNet161, 분류층을 5클래스로 교체, 사전학습 없음 (26,483,045 파라미터) |
| 2D CNN | 베이스라인 A. Conv(3→32)-Pool-Conv(32→64)-GAP-Linear 구조의 초경량 모델 (19,717 파라미터). 논문의 2D-CNN과는 다른 모델이다 |
| ResNet18 | 베이스라인 B. torchvision ResNet18, `fc`를 5클래스로 교체, 사전학습 없음 (11,179,077 파라미터) |

### 학습 및 테스트 방법
- 공통 조건: 같은 분할, Adam, 학습률 0.001, 배치 16, 45 에폭, Cross-Entropy, 사전학습 없음, **Dropout 미적용**(논문은 Dropout을 사용했으나 비율은 공개되지 않음).
- 에폭마다 테스트 정확도를 참고용으로 기록하되 모델 선택에는 쓰지 않고, **마지막 에폭의 모델**로 평가한다.
- 평가: 테스트 정확도, macro Precision/Recall/F1, 혼동행렬, 5-fold 교차검증(시행 단위 StratifiedKFold, 폴드마다 새 모델), 학습 시간·파라미터 수·추론 시간(샘플 1개, 예열 후 측정).
- 학습 시드 `[ ]`개(예: 42, 43, 44)의 평균을 보고한다. 분할은 고정하고 초기화·셔플 시드만 바꾼다.

### 코드 실행 방법
```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/sea3551/palm-sEMG-doorknob-filtered.git
python main.py --data-dir palm-sEMG-doorknob-filtered/data --epochs 45 --seeds 42 43 44 --cv
# 빠른 동작 확인: python main.py --epochs 1 --max-batches 3 --models cnn2d --infer-iters 5
```
결과는 `results/` 폴더에 저장된다 (`results_table.md`, `confusion_*.png`, `train_log_*.csv`, `cv_*.csv` 등).
환경: Google Colab GPU, Python `[ ]`, PyTorch `[ ]`

### 코드 설명 (파일별 역할)
| 파일 | 설명 |
|---|---|
| main.py | 전체 실험 실행 (분할 → 모델별 학습·평가 → 교차검증 → 결과 표 저장) |
| data.py | 시행 단위 분할, 필터·윈도우·정규화·CWT |
| model.py | 모델 구성(DenseNet161, 2D CNN, ResNet18)과 학습 루프 |
| evaluation.py | 지표, 혼동행렬, 최대 오류 쌍, 추론 시간, 5-fold 교차검증 |

---

## 2. 모델 성능 비교

(`results/results_table.md`의 값을 붙여 넣는다. Precision/Recall/F1은 macro 평균)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| DenseNet161 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 2D CNN | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| ResNet18 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

### 추가 지표
| Model | 학습 시간(초) | 파라미터 수 | 추론 시간(ms/샘플) |
|---|---|---|---|
| DenseNet161 | `[ ]` | `[ ]` | `[ ]` |
| 2D CNN | `[ ]` | `[ ]` | `[ ]` |
| ResNet18 | `[ ]` | `[ ]` | `[ ]` |

### 5-fold 교차검증 (DenseNet161)
| Fold | 1 | 2 | 3 | 4 | 5 | 평균 ± 표준편차 |
|---|---|---|---|---|---|---|
| Accuracy (%) | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` (ddof=`[ ]`) |

### 성능 분석
`[어떤 모델이 가장 우수했는지와 그 이유. 베이스라인이 더 좋으면 그대로 보고하고 해석]`

### 논문 수치와의 비교
| 항목 | 논문 | 본 실험 |
|---|---|---|
| 테스트 정확도 | 94.00% | `[ ]` |
| F1-score | 93.99% | `[ ]` |
| 5-fold 평균 | 91.66 ± 2.78% | `[ ]` |
| 최대 오류 쌍 | D → C 13건 | `[ ]` |
| 파라미터 수 (DenseNet161) | 26,483,045 | `[ ]` |

차이가 난 이유: `[한 문단. 예) Dropout 미적용, 공개 데이터 이중 필터링, 정규화 방식, 시드, 에폭 등]`

---

## 3. Confusion Matrix 분석

### Model 1 (DenseNet161)
![confusion DenseNet161](results/confusion_densenet161.png)
- 가장 잘 분류된 클래스: `[ ]`
- 가장 많이 오분류된 클래스: `[ ]`
- 주요 오분류 유형: `[ ]`
- 오분류가 발생한 이유에 대한 분석: `[ ]`

### Model 2 (2D CNN)
![confusion 2D CNN](results/confusion_cnn2d.png)
- 가장 잘 분류된 클래스: `[ ]`
- 가장 많이 오분류된 클래스: `[ ]`
- 주요 오분류 유형: `[ ]`
- 오분류가 발생한 이유에 대한 분석: `[ ]`

### Model 3 (ResNet18)
![confusion ResNet18](results/confusion_resnet18.png)
- 가장 잘 분류된 클래스: `[ ]`
- 가장 많이 오분류된 클래스: `[ ]`
- 주요 오분류 유형: `[ ]`
- 오분류가 발생한 이유에 대한 분석: `[ ]`

---

## 4. 최종 결과

- 가장 성능이 좋은 모델: `[ ]`
- 가장 성능이 낮은 모델: `[ ]`
- 주요 오분류 클래스: `[ ]`
- 전체적인 실험 결과 및 느낀 점: `[ ]`

### 한계
1. 피험자가 5명뿐이라 결과를 일반화하기 어렵다. `[한 문장 더]`
2. 모든 데이터가 단일 측정 세션에서 수집되어 날짜·근피로·피부 상태 변화가 반영되지 않았다. `[한 문장 더]`
3. closed-set 전제라 미등록자가 들어오면 반드시 등록자 중 한 명으로 오분류한다. `[한 문장 더]`
4. `[추가: 학습 시드 수, Dropout 미적용, 두 채널을 합친 정규화 등 본 실험 고유의 제약]`

### 재현 정보
환경 Colab GPU · 분할 시드 42, 학습 시드 `[ ]` · 실행 순서: `python main.py --seeds 42 43 44 --cv` · 저장소: `[GitHub 주소]`
