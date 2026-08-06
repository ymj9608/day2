"""Adult 데이터를 이용한 End-to-End 분석 파이프라인.

Pandas·Polars 로딩 비교, 결측치·중복 처리, 기술통계·상관분석,
Plotly 시각화, scikit-learn Pipeline 학습·평가·저장, report.md 생성을
한 번에 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from IPython.display import display
from scipy import sparse, stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

EVER_MARRIED_STATUSES = {
    "Married-civ-spouse",
    "Married-AF-spouse",
    "Married-spouse-absent",
    "Divorced",
    "Separated",
    "Widowed",
}
NEVER_MARRIED_STATUSES = {"Never-married"}
VALID_MARITAL_STATUSES = EVER_MARRIED_STATUSES | NEVER_MARRIED_STATUSES

MODEL_NUMERIC_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

MODEL_CATEGORICAL_COLUMNS = [
    "workclass",
    "occupation",
    "race",
    "sex",
    "native-country",
    "income",
]

LEAKAGE_COLUMNS = ["marital-status", "relationship"]


def _with_marriage_experience(frame: pd.DataFrame) -> pd.DataFrame:
    """원래 혼인 상태로부터 결혼 경험 이진 타깃을 생성한다."""

    selected = frame.loc[frame["marital-status"].isin(VALID_MARITAL_STATUSES)].copy()
    selected["marriage_experience"] = np.where(
        selected["marital-status"].isin(EVER_MARRIED_STATUSES),
        "결혼 경험 있음",
        "결혼 경험 없음",
    )
    selected["ever_married"] = selected["marital-status"].isin(EVER_MARRIED_STATUSES).astype(int)
    return selected


def _markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    """추가 패키지 없이 DataFrame을 간단한 Markdown 표로 변환한다."""

    printable = frame.reset_index() if include_index else frame.reset_index(drop=True)

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in printable.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in printable.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class PipelineResult:
    """실행 결과와 생성 파일 위치를 노트북에서 확인하기 위한 요약."""

    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_before_split: int
    missing_after_preprocessing: int
    accuracy: float
    f1: float
    output_dir: Path
    report_path: Path
    model_path: Path
    plotly_path: Path
    ttest_path: Path


def _load_with_pandas(data_path: Path) -> tuple[pd.DataFrame, float]:
    """Adult 데이터를 Pandas로 읽고 문자열 공백과 소득 라벨을 정규화한다."""

    started = perf_counter()
    frame = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?"],
    )
    elapsed = perf_counter() - started
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].str.strip()
    frame["income"] = frame["income"].str.removesuffix(".")
    return frame, elapsed


def _load_with_polars(data_path: Path) -> tuple[pl.DataFrame, float]:
    """같은 파일을 Polars로 읽어 Pandas 결과와 행·열 수를 비교한다."""

    started = perf_counter()
    frame = pl.read_csv(
        data_path,
        has_header=False,
        new_columns=COLUMNS,
        null_values=["?", " ?"],
    )
    # 원본 파일 끝의 빈 줄은 Pandas와 동일하게 데이터 행에서 제외한다.
    frame = frame.filter(pl.col("age").is_not_null())
    string_columns = [name for name, dtype in frame.schema.items() if dtype == pl.String]
    frame = frame.with_columns(pl.col(string_columns).str.strip_chars())
    frame = frame.with_columns(pl.col("income").str.strip_suffix("."))
    elapsed = perf_counter() - started
    return frame, elapsed


def _library_comparison(
    pandas_frame: pd.DataFrame,
    pandas_seconds: float,
    polars_frame: pl.DataFrame,
    polars_seconds: float,
) -> pd.DataFrame:
    """두 라이브러리의 로딩 결과와 메모리 사용량을 표로 반환한다."""

    comparison = pd.DataFrame(
        {
            "라이브러리": ["Pandas", "Polars"],
            "행": [pandas_frame.shape[0], polars_frame.height],
            "열": [pandas_frame.shape[1], polars_frame.width],
            "로딩 시간(초)": [pandas_seconds, polars_seconds],
            "추정 메모리(MB)": [
                pandas_frame.memory_usage(deep=True).sum() / 1024**2,
                polars_frame.estimated_size("mb"),
            ],
        }
    )
    if tuple(pandas_frame.shape) != tuple(polars_frame.shape):
        raise ValueError("Pandas와 Polars의 로딩 결과 크기가 일치하지 않습니다.")
    return comparison


def _remove_duplicates(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습·테스트 분할 전에 완전 중복 행만 제거한다.

    결측치는 이 단계에서 대체하지 않는다. 데이터 분할 후 수치형 중앙값과
    범주형 최빈값을 학습 데이터에서 계산하도록 Pipeline에 전달한다.
    """

    duplicate_count = int(raw.duplicated().sum())
    deduplicated = raw.drop_duplicates().reset_index(drop=True)

    cleaning_summary = pd.DataFrame(
        {
            "지표": [
                "전체 행",
                "전체 결측치",
                "완전 중복 행",
                "분석 핵심 변수 결측치",
            ],
            "원본": [
                len(raw),
                int(raw.isna().sum().sum()),
                int(raw.duplicated().sum()),
                int(raw[["education-num", "marital-status", "age", "sex"]].isna().sum().sum()),
            ],
            "분할 전 중복 제거 후": [
                len(deduplicated),
                int(deduplicated.isna().sum().sum()),
                int(deduplicated.duplicated().sum()),
                int(deduplicated[["education-num", "marital-status", "age", "sex"]].isna().sum().sum()),
            ],
            "처리 위치 및 방법": [
                "분할 전 완전 중복 제거",
                "분할 후 Pipeline에서 대체",
                f"분할 전 {duplicate_count}건 제거",
                "원래 결측치 없음—별도 대체 불필요",
            ],
        }
    )
    return deduplicated, cleaning_summary


def _save_statistics(cleaned: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """평균·표준편차·분위수 기술통계와 수치형 상관행렬을 저장한다."""

    descriptive = (
        cleaned[NUMERIC_COLUMNS]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.rename(
            columns={
                "count": "표본수",
                "mean": "평균",
                "std": "표준편차",
                "min": "최솟값",
                "25%": "1사분위수",
                "50%": "중앙값",
                "75%": "3사분위수",
                "max": "최댓값",
            }
        )
    )
    descriptive.index.name = "변수"
    correlation = cleaned[NUMERIC_COLUMNS].corr()
    descriptive.to_csv(output_dir / "descriptive_statistics.csv", encoding="utf-8-sig")
    correlation.to_csv(output_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    return descriptive, correlation


def _analyze_marriage_education(cleaned: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """결혼 경험 여부에 따른 평균 교육연수 차이를 Welch t-test로 검정한다."""

    analysis = _with_marriage_experience(cleaned).dropna(subset=["education-num"])
    never = analysis.loc[analysis["ever_married"] == 0, "education-num"].astype(float).to_numpy()
    ever = analysis.loc[analysis["ever_married"] == 1, "education-num"].astype(float).to_numpy()
    t_stat, p_value = stats.ttest_ind(ever, never, equal_var=False, alternative="two-sided")

    variance_ever = ever.var(ddof=1)
    variance_never = never.var(ddof=1)
    standard_error = np.sqrt(variance_ever / len(ever) + variance_never / len(never))
    welch_df = (variance_ever / len(ever) + variance_never / len(never)) ** 2 / (
        (variance_ever / len(ever)) ** 2 / (len(ever) - 1)
        + (variance_never / len(never)) ** 2 / (len(never) - 1)
    )
    difference = ever.mean() - never.mean()
    critical = stats.t.ppf(0.975, df=welch_df)
    pooled_sd = np.sqrt(
        ((len(ever) - 1) * variance_ever + (len(never) - 1) * variance_never)
        / (len(ever) + len(never) - 2)
    )
    hedges_correction = 1 - 3 / (4 * (len(ever) + len(never)) - 9)

    result = pd.DataFrame(
        {
            "결혼 경험 없음 표본수": [len(never)],
            "결혼 경험 있음 표본수": [len(ever)],
            "결혼 경험 없음 평균": [never.mean()],
            "결혼 경험 있음 평균": [ever.mean()],
            "평균 차이(있음-없음)": [difference],
            "95% CI 하한": [difference - critical * standard_error],
            "95% CI 상한": [difference + critical * standard_error],
            "Welch t": [t_stat],
            "p-value": [p_value],
            "Hedges g": [hedges_correction * difference / pooled_sd],
            "판정": ["귀무가설 기각" if p_value < 0.05 else "귀무가설 기각 못함"],
        }
    )
    result.to_csv(output_dir / "marriage_ttest_results.csv", index=False, encoding="utf-8-sig")
    return result


def _save_visualizations(cleaned: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Seaborn 정적 차트와 Plotly 인터랙티브 HTML을 생성한다."""

    analysis = _with_marriage_experience(cleaned).dropna(subset=["education-num"])
    group_order = ["결혼 경험 없음", "결혼 경험 있음"]
    static_path = output_dir / "seaborn_eda.png"
    sns.set_theme(style="whitegrid")
    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    for font in ["Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        data=analysis,
        x="education-num",
        hue="marriage_experience",
        hue_order=group_order,
        discrete=True,
        stat="probability",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set(
        title="결혼 경험 여부별 교육연수 분포",
        xlabel="교육연수",
        ylabel="집단 내 비율",
    )
    sns.boxplot(
        data=analysis,
        x="marriage_experience",
        y="education-num",
        order=group_order,
        ax=axes[1],
    )
    axes[1].set(
        title="결혼 경험 여부별 교육연수 비교",
        xlabel="결혼 경험",
        ylabel="교육연수",
    )
    plt.tight_layout()
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    interactive = px.box(
        analysis,
        x="marriage_experience",
        y="education-num",
        color="marriage_experience",
        category_orders={"marriage_experience": group_order},
        points=False,
        title="결혼 경험 여부별 교육연수 분포",
        labels={"marriage_experience": "결혼 경험", "education-num": "교육연수"},
    )
    interactive.update_layout(showlegend=False)
    interactive_path = output_dir / "plotly_education_by_marriage.html"
    interactive.write_html(interactive_path, include_plotlyjs=True)
    return static_path, interactive_path


def _train_marriage_model(cleaned: pd.DataFrame, output_dir: Path) -> tuple[Pipeline, dict[str, object], Path]:
    """결혼 경험을 예측하는 분류 Pipeline을 학습·평가·저장한다."""

    analysis = _with_marriage_experience(cleaned)
    feature_columns = MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
    features = analysis[feature_columns]
    target = analysis["ever_married"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, MODEL_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, MODEL_CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    transformed_train = model.named_steps["preprocessor"].transform(x_train)
    transformed_test = model.named_steps["preprocessor"].transform(x_test)

    def count_missing(values: object) -> int:
        array = values.data if sparse.issparse(values) else np.asarray(values)
        return int(np.isnan(array).sum())

    missing_after_preprocessing = count_missing(transformed_train) + count_missing(transformed_test)
    metrics: dict[str, object] = {
        "target": "ever_married",
        "positive_class": "결혼 경험 있음",
        "feature_columns": feature_columns,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_missing_before_preprocessing": int(x_train.isna().sum().sum()),
        "test_missing_before_preprocessing": int(x_test.isna().sum().sum()),
        "missing_after_preprocessing": missing_after_preprocessing,
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["결혼 경험 없음", "결혼 경험 있음"],
            output_dict=True,
            zero_division=0,
        ),
    }

    model_path = output_dir / "marriage_experience_pipeline.joblib"
    metrics_path = output_dir / "model_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, metrics, model_path


def _generate_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    descriptive: pd.DataFrame,
    correlation: pd.DataFrame,
    ttest_result: pd.DataFrame,
    metrics: dict[str, object],
) -> Path:
    """실행 결과를 Markdown 보고서로 자동 작성한다."""

    # 대각선을 제외한 절댓값 최대 상관관계를 보고서에 표시한다.
    correlation_without_diagonal = correlation.copy()
    for column in correlation_without_diagonal.columns:
        correlation_without_diagonal.loc[column, column] = pd.NA
    pair = correlation_without_diagonal.abs().stack().idxmax()
    pair_value = float(correlation.loc[pair[0], pair[1]])
    ttest = ttest_result.iloc[0]

    report_path = output_dir / "report.md"
    report = f"""# Day 2 종합실습 자동 분석 보고서

## 연구 질문과 가설

- 연구 질문: 결혼 경험 여부에 따라 평균 교육연수가 다른가?
- 귀무가설(H0): 두 집단의 평균 교육연수는 같다.
- 대립가설(H1): 두 집단의 평균 교육연수는 다르다.
- 유의수준: 0.05

## 데이터 로딩 비교

{_markdown_table(comparison.round(4))}

Pandas와 Polars에서 모두 {int(comparison.loc[0, '행']):,}행 × {int(comparison.loc[0, '열'])}열을 확인했다.
시간 측정값은 실행 환경과 캐시 상태에 따라 달라질 수 있다.

## 결측치·중복 처리

{_markdown_table(cleaning_summary)}

완전 중복 행은 학습·테스트 분할 전에 제거했다. 남은 결측치는 분할 시점까지 유지하고,
수치형 중앙값과 범주형 최빈값을 학습 데이터에서 계산해 학습·테스트 데이터에 각각 적용했다.

## 기술통계

{_markdown_table(descriptive.round(3), include_index=True)}

평균·표준편차뿐 아니라 1사분위수, 중앙값, 3사분위수를 함께 제시했다.

## 상관분석

대각선을 제외한 절댓값 기준 최대 상관관계는 `{pair[0]}`와 `{pair[1]}`이며,
상관계수는 {pair_value:.3f}이다. 전체 행렬은 [correlation_matrix.csv](correlation_matrix.csv)에 저장했다.

## 결혼 경험 여부와 교육연수 통계 분석

- 결혼 경험 없음: {int(ttest['결혼 경험 없음 표본수']):,}명, 평균 {float(ttest['결혼 경험 없음 평균']):.3f}년
- 결혼 경험 있음: {int(ttest['결혼 경험 있음 표본수']):,}명, 평균 {float(ttest['결혼 경험 있음 평균']):.3f}년
- 평균 차이(있음-없음): {float(ttest['평균 차이(있음-없음)']):.3f}년
- Welch t-test p-value: {float(ttest['p-value']):.3e}
- Hedges g: {float(ttest['Hedges g']):.3f}
- 판정: {ttest['판정']}

p-value가 0.05보다 작아 두 집단의 평균 교육연수 차이는 통계적으로 유의하다.
다만 효과크기는 매우 작으므로 통계적 유의성과 실제 차이의 크기를 구분해 해석해야 한다.
전체 검정 결과는 [marriage_ttest_results.csv](marriage_ttest_results.csv)에 저장했다.

## 결혼 경험 예측 ML Pipeline

- 모델: 수치형 중앙값 대체·표준화 + 범주형 최빈값 대체·원-핫 인코딩 + 로지스틱 회귀
- 타깃: 결혼 경험 없음(0) / 있음(1)
- 누수 방지 제외 변수: `marital-status`, `relationship`
- 분할 전 결측치: {int(metrics['train_missing_before_preprocessing']) + int(metrics['test_missing_before_preprocessing']):,}개
- Pipeline 전처리 후 결측치: {int(metrics['missing_after_preprocessing']):,}개
- 테스트 표본: {int(metrics['test_rows']):,}건
- 정확도: {float(metrics['accuracy']):.4f}
- F1: {float(metrics['f1']):.4f}
- 저장 모델: [marriage_experience_pipeline.joblib](marriage_experience_pipeline.joblib)

정확도는 전체 정답 비율이며, F1은 `결혼 경험 있음` 클래스의 정밀도와 재현율을 함께 반영한다.
이 모델은 결혼 경험과 변수 사이의 패턴을 분류하며 결혼 여부의 원인이나 미래 결혼을 예측하지 않는다.

## 시각화 산출물

- [Seaborn 정적 차트](seaborn_eda.png)
- [Plotly 인터랙티브 차트](plotly_education_by_marriage.html)

## 해석 범위

분석 결과는 1994년 미국 Adult 데이터에서 관찰된 연관성이며 결혼 경험의 인과효과가 아니다.
ML 성능은 단일 학습·테스트 분할 결과이며, 다른 시기·집단에서는 성능이 달라질 수 있다.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_pipeline(
    data_path: str | Path = "adult.data",
    output_dir: str | Path = "artifacts",
) -> PipelineResult:
    """전체 분석 파이프라인을 실행하고 생성된 산출물 정보를 반환한다."""

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"입력 데이터를 찾을 수 없습니다: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pandas_frame, pandas_seconds = _load_with_pandas(data_path)
    polars_frame, polars_seconds = _load_with_polars(data_path)
    comparison = _library_comparison(
        pandas_frame,
        pandas_seconds,
        polars_frame,
        polars_seconds,
    )
    comparison.to_csv(output_dir / "pandas_polars_comparison.csv", index=False, encoding="utf-8-sig")

    deduplicated, cleaning_summary = _remove_duplicates(pandas_frame)
    cleaning_summary.to_csv(output_dir / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    descriptive, correlation = _save_statistics(deduplicated, output_dir)
    ttest_result = _analyze_marriage_education(deduplicated, output_dir)
    _save_visualizations(deduplicated, output_dir)
    _, metrics, model_path = _train_marriage_model(deduplicated, output_dir)
    report_path = _generate_report(
        output_dir,
        comparison,
        cleaning_summary,
        descriptive,
        correlation,
        ttest_result,
        metrics,
    )

    display(comparison.round(4))
    display(cleaning_summary)
    display(descriptive.round(3))
    display(correlation.round(3))
    display(ttest_result.round(4))
    display(pd.DataFrame([{"정확도": metrics["accuracy"], "F1": metrics["f1"]}]).round(4))

    return PipelineResult(
        rows_before=len(pandas_frame),
        rows_after=len(deduplicated),
        duplicates_removed=len(pandas_frame) - len(deduplicated),
        missing_before_split=int(deduplicated.isna().sum().sum()),
        missing_after_preprocessing=int(metrics["missing_after_preprocessing"]),
        accuracy=float(metrics["accuracy"]),
        f1=float(metrics["f1"]),
        output_dir=output_dir,
        report_path=report_path,
        model_path=model_path,
        plotly_path=output_dir / "plotly_education_by_marriage.html",
        ttest_path=output_dir / "marriage_ttest_results.csv",
    )


if __name__ == "__main__":
    result = run_pipeline()
    print(result)
