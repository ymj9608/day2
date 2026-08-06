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
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from IPython.display import display
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

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


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
    missing_before: int
    missing_after: int
    accuracy: float
    f1: float
    output_dir: Path
    report_path: Path
    model_path: Path
    plotly_path: Path


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


def _clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측 범주를 명시적으로 대체하고 완전 중복 행을 제거한다.

    분석 핵심 변수에는 결측치가 없지만, 원본의 workclass·occupation·
    native-country 결측치는 'Unknown'으로 대체해 처리 결과를 남긴다.
    """

    missing_before_by_column = raw.isna().sum()
    missing_columns = missing_before_by_column[missing_before_by_column > 0].index.tolist()
    cleaned = raw.copy()

    for column in missing_columns:
        if cleaned[column].dtype.kind in "biufc":
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            cleaned[column] = cleaned[column].fillna("Unknown")

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaning_summary = pd.DataFrame(
        {
            "지표": [
                "전체 행",
                "전체 결측치",
                "완전 중복 행",
                "분석 핵심 변수 결측치",
            ],
            "처리 전": [
                len(raw),
                int(raw.isna().sum().sum()),
                int(raw.duplicated().sum()),
                int(raw[["education-num", "marital-status", "age", "sex"]].isna().sum().sum()),
            ],
            "처리 후": [
                len(cleaned),
                int(cleaned.isna().sum().sum()),
                int(cleaned.duplicated().sum()),
                int(cleaned[["education-num", "marital-status", "age", "sex"]].isna().sum().sum()),
            ],
            "처리 방법": [
                "완전 중복 제거",
                "범주형 Unknown 대체·수치형 중앙값 대체",
                f"{duplicate_count}건 제거",
                "원래 결측치 없음—별도 대체 불필요",
            ],
        }
    )
    return cleaned, cleaning_summary


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


def _save_visualizations(cleaned: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Seaborn 정적 차트와 Plotly 인터랙티브 HTML을 생성한다."""

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
        data=cleaned,
        x="education-num",
        hue="income",
        discrete=True,
        stat="probability",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set(
        title="소득 집단별 교육연수 분포",
        xlabel="교육연수",
        ylabel="집단 내 비율",
    )
    sns.boxplot(data=cleaned, x="income", y="hours-per-week", ax=axes[1])
    axes[1].set(
        title="소득 집단별 주당 근로시간",
        xlabel="소득 집단",
        ylabel="주당 근로시간",
    )
    plt.tight_layout()
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    income_by_education = (
        cleaned.assign(income_over_50k=cleaned["income"].eq(">50K").astype(int))
        .groupby("education", as_index=False)
        .agg(표본수=("income_over_50k", "size"), 고소득비율=("income_over_50k", "mean"))
        .sort_values("고소득비율", ascending=False)
    )
    interactive = px.bar(
        income_by_education,
        x="education",
        y="고소득비율",
        color="표본수",
        hover_data={"표본수": ":,", "고소득비율": ":.2%"},
        title="교육수준별 연소득 5만 달러 초과 비율",
        labels={"education": "교육수준", "고소득비율": "고소득 비율"},
    )
    interactive.update_layout(xaxis_tickangle=-45)
    interactive.update_yaxes(tickformat=".0%")
    interactive_path = output_dir / "plotly_income_by_education.html"
    interactive.write_html(interactive_path, include_plotlyjs=True)
    return static_path, interactive_path


def _train_income_model(cleaned: pd.DataFrame, output_dir: Path) -> tuple[Pipeline, dict[str, object], Path]:
    """소득 분류 Pipeline을 학습하고 정확도·F1을 계산한 뒤 모델을 저장한다."""

    features = cleaned[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    target = cleaned["income"].eq(">50K").astype(int)
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
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
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
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["<=50K", ">50K"],
            output_dict=True,
            zero_division=0,
        ),
    }

    model_path = output_dir / "adult_income_pipeline.joblib"
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
    metrics: dict[str, object],
) -> Path:
    """실행 결과를 Markdown 보고서로 자동 작성한다."""

    # 대각선을 제외한 절댓값 최대 상관관계를 보고서에 표시한다.
    correlation_without_diagonal = correlation.copy()
    for column in correlation_without_diagonal.columns:
        correlation_without_diagonal.loc[column, column] = pd.NA
    pair = correlation_without_diagonal.abs().stack().idxmax()
    pair_value = float(correlation.loc[pair[0], pair[1]])

    report_path = output_dir / "report.md"
    report = f"""# Day 2 종합실습 자동 분석 보고서

## 데이터 로딩 비교

{_markdown_table(comparison.round(4))}

Pandas와 Polars에서 모두 {int(comparison.loc[0, '행']):,}행 × {int(comparison.loc[0, '열'])}열을 확인했다.
시간 측정값은 실행 환경과 캐시 상태에 따라 달라질 수 있다.

## 결측치·중복 처리

{_markdown_table(cleaning_summary)}

분석 핵심 변수에는 원래 결측치가 없었다. 그 외 범주형 결측치는 `Unknown`으로 명시적으로 대체했고,
완전 중복 행은 독립 관측으로 볼 근거가 부족하므로 제거했다.

## 기술통계

{_markdown_table(descriptive.round(3), include_index=True)}

평균·표준편차뿐 아니라 1사분위수, 중앙값, 3사분위수를 함께 제시했다.

## 상관분석

대각선을 제외한 절댓값 기준 최대 상관관계는 `{pair[0]}`와 `{pair[1]}`이며,
상관계수는 {pair_value:.3f}이다. 전체 행렬은 [correlation_matrix.csv](correlation_matrix.csv)에 저장했다.

## ML Pipeline 평가

- 모델: 수치형 중앙값 대체·표준화 + 범주형 최빈값 대체·원-핫 인코딩 + 로지스틱 회귀
- 테스트 표본: {int(metrics['test_rows']):,}건
- 정확도: {float(metrics['accuracy']):.4f}
- F1: {float(metrics['f1']):.4f}
- 저장 모델: [adult_income_pipeline.joblib](adult_income_pipeline.joblib)

정확도는 전체 정답 비율이고, F1은 상대적으로 적은 `>50K` 집단의 정밀도와 재현율을 함께 반영한다.

## 시각화 산출물

- [Seaborn 정적 차트](seaborn_eda.png)
- [Plotly 인터랙티브 차트](plotly_income_by_education.html)

## 해석 범위

소득 예측 결과는 변수 간 연관성을 학습한 것으로 개인의 능력이나 미래 소득에 대한 인과적 판단이 아니다.
현재 성능은 단일 학습·테스트 분할 결과이며 데이터와 분할 기준이 달라지면 평가 지표도 달라질 수 있다.
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

    cleaned, cleaning_summary = _clean_data(pandas_frame)
    cleaning_summary.to_csv(output_dir / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    descriptive, correlation = _save_statistics(cleaned, output_dir)
    _save_visualizations(cleaned, output_dir)
    _, metrics, model_path = _train_income_model(cleaned, output_dir)
    report_path = _generate_report(
        output_dir,
        comparison,
        cleaning_summary,
        descriptive,
        correlation,
        metrics,
    )

    display(comparison.round(4))
    display(cleaning_summary)
    display(descriptive.round(3))
    display(correlation.round(3))
    display(pd.DataFrame([{"정확도": metrics["accuracy"], "F1": metrics["f1"]}]).round(4))

    return PipelineResult(
        rows_before=len(pandas_frame),
        rows_after=len(cleaned),
        duplicates_removed=len(pandas_frame) - len(cleaned),
        missing_before=int(pandas_frame.isna().sum().sum()),
        missing_after=int(cleaned.isna().sum().sum()),
        accuracy=float(metrics["accuracy"]),
        f1=float(metrics["f1"]),
        output_dir=output_dir,
        report_path=report_path,
        model_path=model_path,
        plotly_path=output_dir / "plotly_income_by_education.html",
    )


if __name__ == "__main__":
    result = run_pipeline()
    print(result)
