import os
import math
import numpy as np
import pandas as pd

# 집계 행 식별용 공통 키워드
AGGREGATE_KEYWORDS = [
    "world", "oecd members", "euro area", "european union",
    "high income", "low income", "income", "area", "total",
    "ida & ibrd", "caribbean", "middle east"
]


def sanitize_for_json(val):
    """NaN, None, Infinity를 JSON 표준에 호환되도록 정제"""
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_for_json(v) for v in val]
    if pd.isna(val):
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return str(val)


def get_label_column(df, user_label_col=None):
    """데이터프레임의 대표 라벨(국가명 등) 열 추출"""
    if user_label_col and user_label_col in df.columns:
        return user_label_col
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not str(col).startswith("Unnamed:"):
            return col
    return df.columns[0]


def is_aggregate_row(row_val):
    """행 라벨 값이 집계 그룹(대륙, 소득분위, 전세계 등)에 해당하는지 판별"""
    if pd.isna(row_val):
        return False
    val_lower = str(row_val).lower()
    return any(kw in val_lower for kw in AGGREGATE_KEYWORDS)


def filter_aggregates_if_needed(df, label_col, exclude_aggregates=True):
    """집계 행 제외 옵션 적용 및 제외 행 수 계산"""
    if not exclude_aggregates or not label_col or label_col not in df.columns:
        return df, 0, []

    is_agg_mask = df[label_col].apply(is_aggregate_row)
    excluded_names = df.loc[is_agg_mask, label_col].dropna().tolist()
    filtered_df = df.loc[~is_agg_mask].copy()
    excluded_count = int(is_agg_mask.sum())
    return filtered_df, excluded_count, excluded_names


# -------------------------------------------------------------
# Tool 1: reshape_wide_to_long (연도 열을 행으로 접기)
# -------------------------------------------------------------
def reshape_wide_to_long(df, label_col=None, value_name="value", var_name="year", exclude_aggregates=True, **kwargs):
    """
    가로로 펼쳐진 연도 열들을 [라벨, 연도, 값] 형태의 세로형(Long Format/Tidy Data)으로 변환
    """
    target_label = get_label_column(df, label_col)
    
    year_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    if not year_cols:
        raise ValueError("데이터에서 4자리 연도 형태의 열을 찾을 수 없습니다.")

    filtered_df, excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)
    
    id_vars = [c for c in filtered_df.columns if c not in year_cols and not str(c).startswith("Unnamed:")]
    if not id_vars:
        id_vars = [target_label]

    melted = pd.melt(
        filtered_df,
        id_vars=id_vars,
        value_vars=year_cols,
        var_name=var_name,
        value_name=value_name
    )

    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    melted_clean = melted.dropna(subset=[value_name]).copy()
    melted_clean = melted_clean.sort_values(by=[target_label, var_name])

    preview_rows = melted_clean.head(15).to_dict(orient="records")

    return {
        "tool": "reshape_wide_to_long",
        "result": {
            "total_long_rows": int(len(melted_clean)),
            "unique_entities_count": int(melted_clean[target_label].nunique()),
            "years_span": f"{min(year_cols)} ~ {max(year_cols)}",
            "sample_data": sanitize_for_json(preview_rows)
        },
        "reference_period": f"{min(year_cols)}~{max(year_cols)} (전체 연도)",
        "used_rows_count": int(len(filtered_df)),
        "excluded_rows_count": excluded_count,
        "excluded_reason": f"집계 행(대륙/소득군 등 {excluded_count}개) 제외" if excluded_count > 0 else "제외된 행 없음",
        "used_columns": id_vars + [var_name, value_name],
        "unit": "행 (Tidy Data Record)",
        "caution": "연도별 결측치가 존재하는 행은 세로형 변환 과정에서 제외되었습니다.",
        "chart_data": None
    }


# -------------------------------------------------------------
# Tool 2: quality_report (결측·중복·타입·이상한 열 이름)
# -------------------------------------------------------------
def quality_report(df, **kwargs):
    """
    데이터 전체의 결측치, 중복 행, 열 이름 이상 여부, 타입 품질을 전수 검사
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    missing_summary = []
    for col in df.columns:
        na_cnt = int(df[col].isna().sum())
        na_pct = round((na_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        if na_cnt > 0:
            missing_summary.append({
                "column": str(col),
                "missing_count": na_cnt,
                "missing_pct": na_pct
            })
    missing_summary.sort(key=lambda x: x["missing_pct"], reverse=True)

    duplicate_rows = int(df.duplicated().sum())
    strange_cols = [str(c) for c in df.columns if str(c).startswith("Unnamed:") or str(c).strip() == ""]
    all_empty_cols = [str(c) for c in df.columns if df[c].isna().all()]

    return {
        "tool": "quality_report",
        "result": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows,
            "columns_with_missing_count": len(missing_summary),
            "strange_columns": strange_cols,
            "all_empty_columns": all_empty_cols,
            "top_missing_columns": sanitize_for_json(missing_summary[:10])
        },
        "reference_period": "전체 데이터셋 기준",
        "used_rows_count": total_rows,
        "excluded_rows_count": 0,
        "excluded_reason": "전체 품질 진단이므로 제외된 행 없음",
        "used_columns": [str(c) for c in df.columns],
        "unit": "개 (항목수)",
        "caution": "결측치가 100%인 열은 실제 분석 계산 시 자동으로 배제되어야 합니다.",
        "chart_data": None
    }


# -------------------------------------------------------------
# Tool 3: year_coverage (연도별 값 개수와 권장 기준연도)
# -------------------------------------------------------------
def year_coverage(df, **kwargs):
    """
    연도 열들의 유효 데이터 개수를 전수 계산하고, 가장 분석에 적합한 권장 기준연도 판별
    """
    year_cols = [str(c) for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    if not year_cols:
        raise ValueError("분석할 수 있는 4자리 연도 열이 존재하지 않습니다.")

    coverage_list = []
    total_rows = len(df)

    for y in sorted(year_cols, key=lambda x: int(x)):
        valid_cnt = int(df[y].notna().sum())
        cov_pct = round((valid_cnt / total_rows) * 100, 1) if total_rows > 0 else 0.0
        coverage_list.append({
            "year": y,
            "valid_count": valid_cnt,
            "missing_count": total_rows - valid_cnt,
            "coverage_pct": cov_pct
        })

    recommended_year = None
    for item in sorted(coverage_list, key=lambda x: int(x["year"]), reverse=True):
        if item["coverage_pct"] >= 70.0:
            recommended_year = item["year"]
            break
    if not recommended_year:
        recommended_year = max(coverage_list, key=lambda x: x["valid_count"])["year"]

    recent_coverage = coverage_list[-20:] if len(coverage_list) > 20 else coverage_list
    chart_data = {
        "type": "bar",
        "labels": [item["year"] for item in recent_coverage],
        "datasets": [{
            "label": "데이터 충실도 (%)",
            "data": [item["coverage_pct"] for item in recent_coverage],
            "backgroundColor": ["#4f46e5" if item["year"] == recommended_year else "#94a3b8" for item in recent_coverage]
        }]
    }

    return {
        "tool": "year_coverage",
        "result": {
            "total_years_count": len(year_cols),
            "year_span": f"{min(year_cols)} ~ {max(year_cols)}",
            "recommended_reference_year": recommended_year,
            "coverage_table": sanitize_for_json(coverage_list)
        },
        "reference_period": f"{min(year_cols)}~{max(year_cols)} (전체 연도)",
        "used_rows_count": total_rows,
        "excluded_rows_count": 0,
        "excluded_reason": "전체 연도 커버리지 검사이므로 제외된 행 없음",
        "used_columns": year_cols,
        "unit": "% (충실도 비율)",
        "caution": f"최신 연도라도 데이터 충실도가 급감할 수 있습니다. 70% 이상 채워진 {recommended_year}년을 분석 기준시점으로 권장합니다.",
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# Tool 4: detect_aggregates (집계 행 후보 탐지)
# -------------------------------------------------------------
def detect_aggregates(df, label_col=None, **kwargs):
    """
    개별 국가/관측치가 아닌 World, High income, Euro area 등 묶음 집계 행을 정밀 탐지
    """
    target_label = get_label_column(df, label_col)
    total_rows = len(df)
    
    aggregate_candidates = []
    pure_candidates = []

    for idx, val in df[target_label].items():
        if pd.isna(val):
            continue
        val_str = str(val)
        val_lower = val_str.lower()
        matched = [kw for kw in AGGREGATE_KEYWORDS if kw in val_lower]
        if matched:
            aggregate_candidates.append({
                "row_index": int(idx),
                "name": val_str,
                "matched_keyword": matched[0]
            })
        else:
            pure_candidates.append(val_str)

    return {
        "tool": "detect_aggregates",
        "result": {
            "total_rows": total_rows,
            "aggregate_count": len(aggregate_candidates),
            "pure_entities_count": len(pure_candidates),
            "aggregate_candidates": sanitize_for_json(aggregate_candidates),
            "sample_pure_entities": sanitize_for_json(pure_candidates[:10])
        },
        "reference_period": "전체 행 기준",
        "used_rows_count": total_rows,
        "excluded_rows_count": 0,
        "excluded_reason": "전체 집계 행 탐지 목적이므로 제외된 행 없음",
        "used_columns": [target_label],
        "unit": "개 (행 수)",
        "caution": "국가 간 순위 매기기나 평균 산출 시 이 집계 행들을 제외하지 않으면 이중 집계 왜곡이 발생합니다.",
        "chart_data": {
            "type": "pie",
            "labels": ["순수 개별 관측치", "대륙/그룹 집계 행"],
            "datasets": [{
                "data": [len(pure_candidates), len(aggregate_candidates)],
                "backgroundColor": ["#3b82f6", "#ef4444"]
            }]
        }
    }


# -------------------------------------------------------------
# Tool 5: describe_numeric (개수·평균·중앙값·최소·최대·표준편차)
# -------------------------------------------------------------
def describe_numeric(df, value_column=None, reference_period=None, exclude_aggregates=True, **kwargs):
    """
    선택된 숫자 열(또는 기준연도)의 기술통계량(개수, 평균, 중앙값, 최소, 최대, 표준편차)을 계산
    """
    target_col = value_column or reference_period
    if not target_col:
        year_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
        target_col = max(year_cols) if year_cols else None
        if not target_col:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_col = num_cols[0] if num_cols else None

    if not target_col or target_col not in df.columns:
        raise ValueError(f"분석할 숫자형 열 또는 기준시점 '{target_col}'을 찾을 수 없습니다.")

    target_label = get_label_column(df)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    numeric_series = pd.to_numeric(filtered_df[target_col], errors="coerce")
    valid_series = numeric_series.dropna()

    missing_count = int(numeric_series.isna().sum())
    total_excluded = agg_excluded_count + missing_count
    used_count = int(len(valid_series))

    if used_count == 0:
        raise ValueError(f"선택한 열 '{target_col}'에 유효한 수치 데이터가 존재하지 않습니다.")

    mean_val = float(valid_series.mean())
    median_val = float(valid_series.median())
    std_val = float(valid_series.std()) if len(valid_series) > 1 else 0.0
    min_val = float(valid_series.min())
    max_val = float(valid_series.max())
    q25 = float(valid_series.quantile(0.25))
    q75 = float(valid_series.quantile(0.75))

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"결측치 {missing_count}개 제외")

    chart_data = {
        "type": "bar",
        "labels": ["최소값", "25% 분위수", "중앙값(50%)", "평균값", "75% 분위수", "최대값"],
        "datasets": [{
            "label": f"{target_col} 기술통계치",
            "data": [min_val, q25, median_val, mean_val, q75, max_val],
            "backgroundColor": ["#94a3b8", "#64748b", "#3b82f6", "#10b981", "#64748b", "#94a3b8"]
        }]
    }

    return {
        "tool": "describe_numeric",
        "result": {
            "target_column": target_col,
            "count": used_count,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2)
        },
        "reference_period": str(target_col),
        "used_rows_count": used_count,
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_col],
        "unit": "% (또는 해당 열 단위)",
        "caution": "평균값은 극단값(100% 또는 0%)의 영향을 많이 받으므로 중앙값과 함께 해석해야 합니다.",
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# Tool 6: trend_line (선택 항목들의 연도별 추이 - 선 차트)
# -------------------------------------------------------------
def trend_line(df, items=None, label_col=None, exclude_aggregates=False, **kwargs):
    """
    선택된 국가/그룹 항목들의 연도별 시계열 변화 추이를 선 차트 데이터로 생성
    """
    target_label = get_label_column(df, label_col)
    
    year_cols = sorted([c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4], key=lambda x: int(x))
    if not year_cols:
        raise ValueError("시계열 추이를 분석할 연도 열을 찾을 수 없습니다.")

    if not items or len(items) == 0:
        candidates = ["Korea, Rep.", "United States", "World", "China", "Japan"]
        existing_candidates = [c for c in candidates if c in df[target_label].values]
        items = existing_candidates if existing_candidates else df[target_label].dropna().head(3).tolist()

    sub_df = df[df[target_label].isin(items)].copy()
    if sub_df.empty:
        raise ValueError(f"선택한 항목 {items}에 해당하는 데이터를 찾을 수 없습니다.")

    active_years = [y for y in year_cols if int(y) >= 1990]
    if not active_years:
        active_years = year_cols

    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
    datasets = []
    trend_summary = {}

    for i, row in sub_df.iterrows():
        name = row[target_label]
        color = palette[len(datasets) % len(palette)]
        values = []
        for y in active_years:
            val = row[y]
            try:
                num = float(val) if pd.notna(val) and str(val).strip() != "" else None
            except Exception:
                num = None
            values.append(sanitize_for_json(num))

        datasets.append({
            "label": str(name),
            "data": values,
            "borderColor": color,
            "backgroundColor": color,
            "fill": False,
            "tension": 0.2
        })

        valid_nums = [v for v in values if v is not None]
        if valid_nums:
            trend_summary[str(name)] = {
                "start_value": valid_nums[0],
                "latest_value": valid_nums[-1],
                "change": round(valid_nums[-1] - valid_nums[0], 2)
            }

    return {
        "tool": "trend_line",
        "result": {
            "selected_items": items,
            "years_span": f"{min(active_years)} ~ {max(active_years)}",
            "trend_summary": sanitize_for_json(trend_summary)
        },
        "reference_period": f"{min(active_years)}~{max(active_years)}",
        "used_rows_count": len(sub_df),
        "excluded_rows_count": len(df) - len(sub_df),
        "excluded_reason": f"선택한 {len(items)}개 항목 외 다른 행 제외",
        "used_columns": [target_label] + active_years,
        "unit": "% (연도별 수치)",
        "caution": "연도별로 조사가 진행되지 않아 결측된 시점은 선 그래프가 끊기거나 보간될 수 있습니다.",
        "chart_data": {
            "type": "line",
            "labels": active_years,
            "datasets": datasets
        }
    }


# -------------------------------------------------------------
# Tool 7: top_bottom_n (기준연도 기준 상위·하위 N - 수평 막대)
# -------------------------------------------------------------
def top_bottom_n(df, reference_period, n=10, mode="top", label_col=None, value_column=None, exclude_aggregates=True, **kwargs):
    """
    지정된 기준연도(필수)를 기준으로 상위 N개 또는 하위 N개 국가 순위를 산출
    """
    if not reference_period:
        raise ValueError("top_bottom_n 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    target_period = str(reference_period).strip()
    target_val_col = value_column or target_period
    if target_val_col not in df.columns:
        raise ValueError(f"데이터셋에 기준 시점 열 '{target_val_col}'이 존재하지 않습니다.")

    target_label = get_label_column(df, label_col)
    n = int(n)

    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    filtered_df["_numeric_val"] = pd.to_numeric(filtered_df[target_val_col], errors="coerce")
    valid_df = filtered_df.dropna(subset=["_numeric_val"]).copy()
    
    missing_count = len(filtered_df) - len(valid_df)
    total_excluded = agg_excluded_count + missing_count

    if valid_df.empty:
        raise ValueError(f"기준 시점 '{target_period}'에 유효한 값이 존재하는 행이 없습니다.")

    sorted_df = valid_df.sort_values(by="_numeric_val", ascending=False)
    
    if mode == "bottom":
        result_df = sorted_df.tail(n).sort_values(by="_numeric_val", ascending=True)
        chart_color = "#f97316"
        chart_title = f"{target_period}년 하위 {len(result_df)}개국"
    elif mode == "both":
        top_half = sorted_df.head(n)
        bottom_half = sorted_df.tail(n).sort_values(by="_numeric_val", ascending=True)
        result_df = pd.concat([top_half, bottom_half]).drop_duplicates()
        chart_color = "#3b82f6"
        chart_title = f"{target_period}년 상위/하위 {n}개국 비교"
    else:
        result_df = sorted_df.head(n)
        chart_color = "#3b82f6"
        chart_title = f"{target_period}년 상위 {len(result_df)}개국"

    rank_list = []
    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        rank_list.append({
            "rank": rank if mode != "bottom" else len(valid_df) - len(result_df) + rank,
            "name": str(row[target_label]),
            "value": round(float(row["_numeric_val"]), 2)
        })

    korea_row = valid_df[valid_df[target_label].str.contains("Korea, Rep.", case=False, na=False)]
    korea_info = None
    if not korea_row.empty:
        k_val = float(korea_row["_numeric_val"].iloc[0])
        k_rank = int((valid_df["_numeric_val"] > k_val).sum() + 1)
        korea_info = {"name": "대한민국 (Korea, Rep.)", "rank": k_rank, "value": round(k_val, 2)}

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"대륙/그룹 집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"{target_period}년 결측치 {missing_count}개 제외")

    # 차트용 라벨과 데이터 및 색상 구성
    # 만약 대한민국 정보가 있고 현재 순위 목록(예: 1~10위)에 포함되어 있지 않다면, 비교를 위해 차트 및 목록에 명시적 구분자로 함께 표현
    chart_labels = []
    chart_values = []
    chart_colors = []
    
    for item in rank_list:
        chart_labels.append(f"{item['rank']}위 {item['name']}")
        chart_values.append(item["value"])
        if "korea" in item["name"].lower() or "대한민국" in item["name"]:
            chart_colors.append("#ef4444")  # 대한민국은 강조 색상 (붉은색/포인트)
        else:
            chart_colors.append(chart_color)

    # 10위권 밖인 대한민국을 함께 비교하도록 요청된 맥락 지원: 한국이 순위목록에 없으면 비교용으로 차트에 추가
    if korea_info and not any("korea" in str(it["name"]).lower() for it in rank_list):
        chart_labels.append(f"🇰🇷 {korea_info['rank']}위 {korea_info['name']}")
        chart_values.append(korea_info["value"])
        chart_colors.append("#ef4444")  # 한국 강조 포인트 색상

    return {
        "tool": "top_bottom_n",
        "result": {
            "mode": mode,
            "reference_period": target_period,
            "total_ranked_countries": len(valid_df),
            "rankings": sanitize_for_json(rank_list),
            "korea_status": sanitize_for_json(korea_info)
        },
        "reference_period": target_period,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_label, target_val_col],
        "unit": "% (비율)",
        "caution": f"{target_period}년 한 해의 단일 시점 기준 순위이며, 데이터가 누락된 국가는 순위 산정에서 제외되었습니다.",
        "chart_data": {
            "type": "bar",
            "indexAxis": "y",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{target_period} 이용률 (%)",
                "data": chart_values,
                "backgroundColor": chart_colors
            }]
        }
    }


# -------------------------------------------------------------
# Tool 8: group_summary (그룹별 평균·합계·개수 - 막대)
# -------------------------------------------------------------
def group_summary(df, reference_period, group_col=None, value_column=None, agg_func="mean", exclude_aggregates=True, **kwargs):
    """
    기준연도(필수)를 기준으로 대륙별(Region) 또는 소득분위별(IncomeGroup) 평균/통계 집계
    """
    if not reference_period:
        raise ValueError("group_summary 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    target_period = str(reference_period).strip()
    target_val_col = value_column or target_period
    if target_val_col not in df.columns:
        raise ValueError(f"데이터셋에 기준 시점 열 '{target_val_col}'이 존재하지 않습니다.")

    working_df = df.copy()

    target_group_col = group_col or "Region"
    if target_group_col not in working_df.columns and "Country Code" in working_df.columns:
        meta_candidates = [f for f in os.listdir("data") if "metadata_country" in f.lower() and f.endswith(".csv")]
        if meta_candidates:
            meta_path = os.path.join("data", meta_candidates[0])
            try:
                mdf = pd.read_csv(meta_path, encoding="utf-8-sig")
                cols_to_merge = [c for c in ["Country Code", "Region", "IncomeGroup"] if c in mdf.columns]
                working_df = working_df.merge(mdf[cols_to_merge], on="Country Code", how="left")
            except Exception:
                pass

    if target_group_col not in working_df.columns:
        cat_cols = [c for c in working_df.columns if not pd.api.types.is_numeric_dtype(working_df[c])]
        if len(cat_cols) > 1:
            target_group_col = cat_cols[1]
        else:
            raise ValueError(f"그룹 분석에 사용할 열 '{target_group_col}'을 찾을 수 없습니다.")

    target_label = get_label_column(working_df)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(working_df, target_label, exclude_aggregates)

    filtered_df["_numeric_val"] = pd.to_numeric(filtered_df[target_val_col], errors="coerce")
    valid_df = filtered_df.dropna(subset=["_numeric_val", target_group_col]).copy()
    valid_df = valid_df[valid_df[target_group_col].astype(str).str.strip() != ""]

    missing_count = len(filtered_df) - len(valid_df)
    total_excluded = agg_excluded_count + missing_count

    if valid_df.empty:
        raise ValueError(f"그룹 분석에 필요한 유효 데이터가 '{target_period}' 시점에 없습니다.")

    grouped = valid_df.groupby(target_group_col)["_numeric_val"].agg(
        mean="mean",
        median="median",
        count="count",
        min="min",
        max="max"
    ).reset_index()

    grouped = grouped.sort_values(by="mean", ascending=False)

    summary_list = []
    for _, row in grouped.iterrows():
        summary_list.append({
            "group": str(row[target_group_col]),
            "mean": round(float(row["mean"]), 2),
            "median": round(float(row["median"]), 2),
            "count": int(row["count"]),
            "min": round(float(row["min"]), 2),
            "max": round(float(row["max"]), 2)
        })

    labels = [item["group"] for item in summary_list]
    values = [item["mean"] for item in summary_list]

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"그룹 정보 또는 {target_period}년 결측치 {missing_count}개 제외")

    return {
        "tool": "group_summary",
        "result": {
            "group_column": target_group_col,
            "reference_period": target_period,
            "groups_count": len(summary_list),
            "group_summary": sanitize_for_json(summary_list)
        },
        "reference_period": target_period,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_group_col, target_val_col],
        "unit": "% (그룹별 평균 이용률)",
        "caution": "단순 국가별 산술 평균이므로 국가별 인구 가중치는 반영되지 않았습니다.",
        "chart_data": {
            "type": "bar",
            "labels": labels,
            "datasets": [{
                "label": f"{target_period}년 {target_group_col}별 평균 (%)",
                "data": values,
                "backgroundColor": "#6366f1"
            }]
        }
    }


# -------------------------------------------------------------
# Tool 9: change_rate (두 시점 사이 변화량·변화율 - 막대)
# -------------------------------------------------------------
def change_rate(df, reference_period, base_period=None, n=10, label_col=None, exclude_aggregates=True, **kwargs):
    """
    두 시점(기준연도 vs 과거 비교연도) 사이의 절대 변화량(%p) 및 변화율(%)을 계산
    """
    if not reference_period:
        raise ValueError("change_rate 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    end_period = str(reference_period).strip()
    if end_period not in df.columns:
        raise ValueError(f"기준 시점 열 '{end_period}'이 존재하지 않습니다.")

    # 비교 시점(base_period) 결정 (기본값: 10년 전, 없으면 5년 전 또는 이전 첫 유효연도)
    year_cols = sorted([c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4], key=lambda x: int(x))
    if not base_period:
        if end_period.isdigit():
            target_prev = str(int(end_period) - 10)
            if target_prev in df.columns:
                start_period = target_prev
            else:
                prev_candidates = [y for y in year_cols if int(y) < int(end_period)]
                start_period = prev_candidates[-5] if len(prev_candidates) >= 5 else prev_candidates[0]
        else:
            raise ValueError("비교 대상 시작 시점(base_period)을 지정해 주세요.")
    else:
        start_period = str(base_period).strip()
        if start_period not in df.columns:
            raise ValueError(f"비교 시점 열 '{start_period}'이 존재하지 않습니다.")

    target_label = get_label_column(df, label_col)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    # 두 시점 수치형 변환
    s_start = pd.to_numeric(filtered_df[start_period], errors="coerce")
    s_end = pd.to_numeric(filtered_df[end_period], errors="coerce")
    valid_mask = s_start.notna() & s_end.notna()

    valid_df = filtered_df[valid_mask].copy()
    valid_df["_start_val"] = s_start[valid_mask]
    valid_df["_end_val"] = s_end[valid_mask]
    
    # 변화량(차이, %p) 및 변화율(%)
    valid_df["_diff"] = valid_df["_end_val"] - valid_df["_start_val"]
    valid_df["_pct_change"] = np.where(
        valid_df["_start_val"] > 0,
        ((valid_df["_end_val"] - valid_df["_start_val"]) / valid_df["_start_val"]) * 100,
        np.nan
    )

    missing_count = len(filtered_df) - len(valid_df)
    total_excluded = agg_excluded_count + missing_count

    if valid_df.empty:
        raise ValueError(f"{start_period}년과 {end_period}년 모두에 데이터가 존재하는 행이 없습니다.")

    # 가장 많이 성장한 상위 N개국
    sorted_growth = valid_df.sort_values(by="_diff", ascending=False).head(int(n))
    growth_list = []
    for _, row in sorted_growth.iterrows():
        growth_list.append({
            "name": str(row[target_label]),
            "start_value": round(float(row["_start_val"]), 2),
            "end_value": round(float(row["_end_val"]), 2),
            "difference_p": round(float(row["_diff"]), 2),
            "pct_change": round(float(row["_pct_change"]), 1) if pd.notna(row["_pct_change"]) else None
        })

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"비교 시점 중 결측치 존재 행 {missing_count}개 제외")

    chart_labels = [item["name"] for item in growth_list]
    chart_diffs = [item["difference_p"] for item in growth_list]

    return {
        "tool": "change_rate",
        "result": {
            "start_period": start_period,
            "end_period": end_period,
            "top_growth_countries": sanitize_for_json(growth_list)
        },
        "reference_period": f"{start_period} -> {end_period}",
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_label, start_period, end_period],
        "unit": "%p (포인트 변화량)",
        "caution": f"초기값({start_period}년)이 매우 낮은 국가는 작은 증가로도 변화율(%)이 수백%로 폭증할 수 있으므로 절대 증가량(%p)을 함께 보아야 합니다.",
        "chart_data": {
            "type": "bar",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{start_period}~{end_period} 증가폭 (%p)",
                "data": chart_diffs,
                "backgroundColor": "#10b981"
            }]
        }
    }


# -------------------------------------------------------------
# Tool 10: distribution (구간별 분포 - 히스토그램)
# -------------------------------------------------------------
def distribution(df, reference_period, bins=5, value_column=None, exclude_aggregates=True, **kwargs):
    """
    기준연도(필수) 수치의 구간별 빈도(히스토그램) 및 비율을 계산
    """
    if not reference_period:
        raise ValueError("distribution 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    target_period = str(reference_period).strip()
    target_val_col = value_column or target_period
    if target_val_col not in df.columns:
        raise ValueError(f"데이터셋에 기준 시점 열 '{target_val_col}'이 존재하지 않습니다.")

    target_label = get_label_column(df)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    series = pd.to_numeric(filtered_df[target_val_col], errors="coerce").dropna()
    missing_count = len(filtered_df) - len(series)
    total_excluded = agg_excluded_count + missing_count

    if series.empty:
        raise ValueError(f"'{target_period}' 시점에 유효한 수치 데이터가 없습니다.")

    # 0~100% 범위 또는 자동 n개 구간 분할
    min_v, max_v = series.min(), series.max()
    if 0 <= min_v and max_v <= 100:
        bin_edges = [0, 20, 40, 60, 80, 100]
        bin_labels = ["0~20%", "20~40%", "40~60%", "60~80%", "80~100%"]
    else:
        bins = int(bins)
        bin_edges = np.linspace(min_v, max_v, bins + 1)
        bin_labels = [f"{round(bin_edges[i], 1)}~{round(bin_edges[i+1], 1)}" for i in range(bins)]

    cut_series = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True)
    counts = cut_series.value_counts(sort=False)

    dist_list = []
    for label, count in counts.items():
        cnt = int(count)
        pct = round((cnt / len(series)) * 100, 1)
        dist_list.append({
            "bin": str(label),
            "count": cnt,
            "percentage": pct
        })

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"결측치 {missing_count}개 제외")

    chart_labels = [item["bin"] for item in dist_list]
    chart_counts = [item["count"] for item in dist_list]

    return {
        "tool": "distribution",
        "result": {
            "reference_period": target_period,
            "total_countries": len(series),
            "distribution_table": sanitize_for_json(dist_list)
        },
        "reference_period": target_period,
        "used_rows_count": len(series),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_val_col],
        "unit": "개국 (국가 수)",
        "caution": f"{target_period}년 단일 시점 분포이며, 최빈 구간에 전 세계 국가가 얼마나 쏠려 있는지(양극화 여부)를 파악할 수 있습니다.",
        "chart_data": {
            "type": "bar",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{target_period}년 국가 수",
                "data": chart_counts,
                "backgroundColor": "#8b5cf6"
            }]
        }
    }


# -------------------------------------------------------------
# Tool 11: correlation_scatter (두 숫자 열 상관계수 - 산점도)
# -------------------------------------------------------------
def correlation_scatter(df, reference_period, compare_column=None, label_col=None, exclude_aggregates=True, **kwargs):
    """
    기준연도(필수)와 비교 대상 열 사이의 피어슨 상관계수 및 산점도 데이터를 계산
    """
    if not reference_period:
        raise ValueError("correlation_scatter 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    col1 = str(reference_period).strip()
    if col1 not in df.columns:
        raise ValueError(f"기준 시점 열 '{col1}'이 존재하지 않습니다.")

    # 비교 열 선정 (지정되지 않은 경우 10년 전 또는 이전 유효 연도)
    if not compare_column:
        year_cols = sorted([c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4], key=lambda x: int(x))
        if col1.isdigit():
            target_comp = str(int(col1) - 10)
            col2 = target_comp if target_comp in df.columns else year_cols[0]
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != col1]
            col2 = num_cols[0] if num_cols else None
    else:
        col2 = str(compare_column).strip()

    if not col2 or col2 not in df.columns:
        raise ValueError(f"비교 대상 열 '{col2}'이 존재하지 않습니다.")

    target_label = get_label_column(df, label_col)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    s1 = pd.to_numeric(filtered_df[col1], errors="coerce")
    s2 = pd.to_numeric(filtered_df[col2], errors="coerce")
    valid_mask = s1.notna() & s2.notna()

    valid_df = filtered_df[valid_mask].copy()
    valid_df["_x"] = s2[valid_mask]
    valid_df["_y"] = s1[valid_mask]

    missing_count = len(filtered_df) - len(valid_df)
    total_excluded = agg_excluded_count + missing_count

    if len(valid_df) < 3:
        raise ValueError("상관계수를 계산하기 위한 쌍(Pair) 데이터가 3개 미만으로 부족합니다.")

    # 피어슨 상관계수 계산
    r_val = float(valid_df["_x"].corr(valid_df["_y"]))

    scatter_points = []
    for _, row in valid_df.iterrows():
        scatter_points.append({
            "x": round(float(row["_x"]), 2),
            "y": round(float(row["_y"]), 2),
            "label": str(row[target_label])
        })

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"두 시점 중 하나라도 결측된 행 {missing_count}개 제외")

    return {
        "tool": "correlation_scatter",
        "result": {
            "x_column": col2,
            "y_column": col1,
            "correlation_coefficient": round(r_val, 4),
            "correlation_strength": "강한 양의 상관관계" if r_val >= 0.7 else "보통 양의 상관관계" if r_val >= 0.4 else "낮은 상관관계",
            "pairs_count": len(valid_df),
            "sample_points": sanitize_for_json(scatter_points[:15])
        },
        "reference_period": f"{col2} vs {col1}",
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_label, col2, col1],
        "unit": "피어슨 상관계수 r (-1.0 ~ +1.0)",
        "caution": "상관관계는 인과관계가 아닙니다. 두 변수 간의 통계적 상관성이 높다고 해서 하나가 다른 하나의 원인이 되는 것은 아닙니다.",
        "chart_data": {
            "type": "scatter",
            "datasets": [{
                "label": f"{col2} (X축) vs {col1} (Y축)",
                "data": scatter_points,
                "backgroundColor": "#3b82f6"
            }]
        }
    }


# -------------------------------------------------------------
# Tool 12: compare_one_vs_all (특정 항목 vs 전체 평균·중앙값 - 강조 막대)
# -------------------------------------------------------------
def compare_one_vs_all(df, reference_period, target_item="Korea, Rep.", label_col=None, exclude_aggregates=True, **kwargs):
    """
    기준연도(필수)에서 특정 관심 국가(예: 대한민국)의 위치를 전체 국가 평균/중앙값과 직접 대조
    """
    if not reference_period:
        raise ValueError("compare_one_vs_all 도구는 기준 시점(reference_period)이 필수 입력값입니다.")

    target_period = str(reference_period).strip()
    if target_period not in df.columns:
        raise ValueError(f"데이터셋에 기준 시점 열 '{target_period}'이 존재하지 않습니다.")

    target_label = get_label_column(df, label_col)
    filtered_df, agg_excluded_count, _ = filter_aggregates_if_needed(df, target_label, exclude_aggregates)

    series = pd.to_numeric(filtered_df[target_period], errors="coerce")
    valid_mask = series.notna()
    valid_df = filtered_df[valid_mask].copy()
    valid_df["_numeric_val"] = series[valid_mask]

    missing_count = len(filtered_df) - len(valid_df)
    total_excluded = agg_excluded_count + missing_count

    if valid_df.empty:
        raise ValueError(f"'{target_period}' 시점에 유효한 수치 데이터가 없습니다.")

    # 대상 항목 탐색
    matched_target = valid_df[valid_df[target_label].str.contains(str(target_item), case=False, na=False)]
    if matched_target.empty:
        raise ValueError(f"'{target_item}' 항목을 데이터셋에서 찾을 수 없거나 해당 연도에 데이터가 없습니다.")

    target_name = str(matched_target[target_label].iloc[0])
    target_value = float(matched_target["_numeric_val"].iloc[0])

    overall_mean = float(valid_df["_numeric_val"].mean())
    overall_median = float(valid_df["_numeric_val"].median())
    overall_max = float(valid_df["_numeric_val"].max())
    overall_min = float(valid_df["_numeric_val"].min())

    rank = int((valid_df["_numeric_val"] > target_value).sum() + 1)
    percentile = round((1 - (rank / len(valid_df))) * 100, 1)
    diff_from_mean = round(target_value - overall_mean, 2)

    excluded_reasons = []
    if agg_excluded_count > 0:
        excluded_reasons.append(f"집계 행 {agg_excluded_count}개 제외")
    if missing_count > 0:
        excluded_reasons.append(f"결측치 {missing_count}개 제외")

    chart_labels = [target_name, f"전체 평균 ({round(overall_mean, 1)}%)", f"전체 중앙값 ({round(overall_median, 1)}%)"]
    chart_values = [round(target_value, 2), round(overall_mean, 2), round(overall_median, 2)]

    return {
        "tool": "compare_one_vs_all",
        "result": {
            "target_item": target_name,
            "target_value": round(target_value, 2),
            "overall_mean": round(overall_mean, 2),
            "overall_median": round(overall_median, 2),
            "diff_from_mean": diff_from_mean,
            "rank": rank,
            "total_countries": len(valid_df),
            "percentile": percentile
        },
        "reference_period": target_period,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "excluded_reason": ", ".join(excluded_reasons) if excluded_reasons else "제외된 행 없음",
        "used_columns": [target_label, target_period],
        "unit": "% (이용률)",
        "caution": f"{target_period}년 단일 시점 기준이며, 전체 평균 대비 {diff_from_mean}%p의 편차를 보입니다.",
        "chart_data": {
            "type": "bar",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{target_period}년 비교 (%)",
                "data": chart_values,
                "backgroundColor": ["#ef4444", "#94a3b8", "#64748b"]
            }]
        }
    }


# -------------------------------------------------------------
# 12대 분석 도구 카탈로그 딕셔너리
# -------------------------------------------------------------
TOOL_CATALOG = {
    "reshape_wide_to_long": {
        "fn": reshape_wide_to_long,
        "name": "연도 열을 행으로 접기",
        "description": "가로로 펼쳐진 연도별 열들을 [국가, 연도, 수치] 형태의 세로형 표준 Tidy Data로 변환합니다.",
        "requires_reference_period": False
    },
    "quality_report": {
        "fn": quality_report,
        "name": "데이터 품질 진단 보고서",
        "description": "결측치 비율, 중복 행, 이상한 열 이름, 완전 빈 열을 종합 검사합니다.",
        "requires_reference_period": False
    },
    "year_coverage": {
        "fn": year_coverage,
        "name": "연도별 충실도 및 권장 기준연도",
        "description": "연도별 유효 데이터 개수 추이를 분석하고 분석에 가장 적합한 권장 기준연도를 제안합니다.",
        "requires_reference_period": False
    },
    "detect_aggregates": {
        "fn": detect_aggregates,
        "name": "대륙 및 소득군 집계 행 탐지",
        "description": "World, Euro area 등 개별 국가가 아닌 묶음 집계 행을 탐지하여 통계 왜곡을 예방합니다.",
        "requires_reference_period": False
    },
    "describe_numeric": {
        "fn": describe_numeric,
        "name": "기술통계량 분석",
        "description": "평균, 중앙값, 표준편차, 사분위수(25%, 75%), 최소, 최대값을 계산합니다.",
        "requires_reference_period": False
    },
    "trend_line": {
        "fn": trend_line,
        "name": "연도별 추이 선 차트",
        "description": "선택한 국가들의 연도별 시계열 변화 궤적을 비교합니다.",
        "requires_reference_period": False
    },
    "top_bottom_n": {
        "fn": top_bottom_n,
        "name": "상위/하위 N개국 순위 비교",
        "description": "기준연도 기준 상위 N개국 또는 하위 N개국의 순위를 매기고 한국의 위치를 확인합니다.",
        "requires_reference_period": True
    },
    "group_summary": {
        "fn": group_summary,
        "name": "대륙/소득군 그룹별 요약",
        "description": "대륙별(Region) 또는 소득분위별(IncomeGroup) 평균 이용률을 비교합니다.",
        "requires_reference_period": True
    },
    "change_rate": {
        "fn": change_rate,
        "name": "두 시점 간 변화량/성장률",
        "description": "과거 연도 대비 기준연도의 절대 증가폭(%p)과 변화율(%)을 계산합니다.",
        "requires_reference_period": True
    },
    "distribution": {
        "fn": distribution,
        "name": "구간별 국가 분포 (히스토그램)",
        "description": "수치 구간(0~20%, 20~40% 등)별 국가 수 분포를 히스토그램으로 시각화합니다.",
        "requires_reference_period": True
    },
    "correlation_scatter": {
        "fn": correlation_scatter,
        "name": "두 시점/변수 간 상관계수 (산점도)",
        "description": "두 연도 간 피어슨 상관계수(r)를 산출하고 산점도로 분포를 확인합니다.",
        "requires_reference_period": True
    },
    "compare_one_vs_all": {
        "fn": compare_one_vs_all,
        "name": "특정 항목 vs 전체 평균 대조",
        "description": "특정 관심 국가(한국 등)를 전체 국가 평균 및 중앙값과 직관적으로 비교합니다.",
        "requires_reference_period": True
    }
}
