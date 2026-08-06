# Day 2 종합실습 — 결혼 경험에 따른 평균 교육수준 비교

Adult 데이터에서 결혼 경험이 있는 집단과 없는 집단의 평균 교육수준(`education-num`) 차이를 분석합니다.

## 연구 가설

- H0: 결혼 경험이 있는 집단과 없는 집단의 평균 교육수준은 같다.
- H1: 결혼 경험이 있는 집단과 없는 집단의 평균 교육수준은 다르다.
- 유의수준: 0.05, 양측검정

## 집단 정의

- 결혼 경험 있음: 현재 기혼, 이혼, 별거, 사별
- 결혼 경험 없음: `Never-married`

분석 집단 명칭은 현재 혼인 상태와 구분하기 위해 `결혼 경험 있음/없음`으로 통일합니다.

## 분석 방법

1. 데이터 로딩과 결측·중복 점검
2. 원래 혼인 상태를 결혼 경험 있음/없음으로 재분류
3. 집단별 표본 수·평균·분포 확인
4. 분산 및 Q-Q plot 점검
5. Welch 독립표본 t-test로 비보정 평균 차이 검정
6. 나이·나이 제곱·성별을 통제한 OLS 회귀분석
7. 부트스트랩과 Mann–Whitney U test로 강건성 확인
8. 비보정·보정 결과 비교 및 결론

End-to-End 파이프라인(`day2_pipeline.py`)은 다음 작업을 수행합니다.

1. Pandas·Polars 데이터 로딩 결과, 시간, 추정 메모리 비교
2. 분할 전 완전 중복 제거 및 분할 후 Pipeline 내부 결측치 대체
3. 분위수를 포함한 기술통계와 수치형 상관행렬 저장
4. Seaborn 정적 차트 및 Plotly 인터랙티브 HTML 생성
5. scikit-learn Pipeline으로 결혼 경험 여부 분류 모델 학습
6. 정확도·F1 평가, joblib 모델 저장, `report.md` 자동 생성

결혼 경험 예측 모델은 `marital-status`에서 타깃을 생성합니다. 타깃을 직접 드러내는
`marital-status`와 `relationship`은 입력 변수에서 제외해 데이터 누수를 방지합니다.

Welch t-test는 실제 두 집단의 전체 평균을 비교합니다. 보정 회귀분석은 나이와 성별 분포를 동일하게 고려했을 때의 모델 기반 평균 차이를 추정합니다. 보정 분석은 실제로 동일한 사람을 일대일 매칭한 결과가 아닙니다.

## 실행 방법

```bash
cd day2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
curl -L \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data \
  -o adult.data
.venv/bin/jupyter lab day2.ipynb
```

JupyterLab에서 `Restart Kernel and Run All Cells`를 실행합니다. 원본 데이터인
`adult.data`는 Git에 포함하지 않으며, 위 명령으로 내려받아 `day2.ipynb`와 같은
폴더에 둡니다.

파이프라인만 별도로 실행할 수 있습니다.

```bash
.venv/bin/python day2_pipeline.py
```

실행 결과는 `artifacts/`에 저장됩니다.

- `pandas_polars_comparison.csv`: 로딩 결과·성능 비교
- `cleaning_summary.csv`: 분할 전 중복 제거와 결측치 처리 단계 요약
- `descriptive_statistics.csv`: 분위수를 포함한 기술통계
- `correlation_matrix.csv`: 수치형 상관행렬
- `marriage_ttest_results.csv`: 결혼 경험 여부별 교육수준 Welch t-test 결과
- `seaborn_eda.png`: 정적 EDA 차트
- `plotly_education_by_marriage.html`: 결혼 경험 여부별 교육수준 인터랙티브 차트
- `model_metrics.json`: 정확도·F1 및 상세 평가 결과
- `marriage_experience_pipeline.joblib`: 저장된 결혼 경험 예측 Pipeline
- `report.md`: 자동 생성 분석 보고서

## 연령 범위 변경

기본 분석 범위는 Adult 데이터의 전체 연령입니다. 연령 민감도 분석은 노트북 3번 코드 셀의 `MIN_AGE`, `MAX_AGE`로 설정합니다.

```python
MIN_AGE = 25
MAX_AGE = None
```

30~54세만 분석하려면 다음처럼 변경합니다.

```python
MIN_AGE = 30
MAX_AGE = 54
```

## 파일 구성

- `day2.ipynb`: 전체 분석 노트북
- `day2_pipeline.py`: End-to-End 분석 파이프라인
- `requirements.txt`: 실행 패키지 목록
- `README.md`: 연구 정의와 실행 안내
- `artifacts/`: 차트·모델·평가 결과·자동 생성 보고서

원본 `adult.data`, 강의 PDF, 가상환경과 캐시는 `.gitignore`로 제외합니다.

## 해석 제한

이 결과는 1994년 미국 Adult 데이터에서 관찰된 집단 간 연관성입니다. 결혼 경험이 교육수준을 변화시켰다는 인과관계로 해석할 수 없습니다. 표본이 크면 작은 차이도 유의할 수 있으므로 p-value뿐 아니라 평균 차이, 95% 신뢰구간, 효과크기를 함께 확인해야 합니다.
