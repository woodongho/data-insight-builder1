import os
import io
import csv
import json
import logging
import math
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tools import TOOL_CATALOG

# .env 환경변수 로드
load_dotenv()

# Python logging 설정 (print() 대신 사용, API Key 출력 금지)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("data-insight-builder")

app = Flask(__name__)

# 업로드 최대 크기 설정 (기본 10MB)
max_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

# 메모리 단일 데이터셋 보관용 변수 (디스크 영구 저장 안 함)
CURRENT_DATASET = None


def sanitize_value(val):
    """NaN, None, Infinity를 JSON 표준에 맞게 None(null) 또는 안전한 기본값으로 변환"""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return str(val)


def detect_encoding_and_parse(raw_bytes):
    """
    1) utf-8-sig, utf-8, cp949, euc-kr 순서로 인코딩 시도
    2) 상단 행을 분석하여 헤더 행(skip_rows) 자동 감지
    3) Pandas DataFrame 파싱
    """
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    text = None
    used_encoding = None

    for enc in encodings:
        try:
            logger.info(f"[ENCODING] {enc} 시도 중...")
            text = raw_bytes.decode(enc)
            used_encoding = enc
            logger.info(f"[ENCODING] {enc} 성공")
            break
        except (UnicodeDecodeError, LookupError):
            logger.info(f"[ENCODING] {enc} 실패")
            continue

    if text is None:
        return None, None, None, None, "파일 인코딩을 인식하지 못했습니다. UTF-8 또는 CP949(EUC-KR) 인코딩으로 저장된 CSV 파일인지 확인해 주세요."

    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return None, None, None, None, "파일의 내용이 비어 있습니다."

    # 상단 25행 중 비어있지 않은 줄 파싱하여 헤더 후보 탐색
    parsed_rows = []
    for idx, line in enumerate(lines[:25]):
        if not line.strip():
            continue
        try:
            row_tokens = list(csv.reader([line]))[0]
            parsed_rows.append((idx, row_tokens))
        except Exception:
            continue

    if not parsed_rows:
        return None, None, None, None, "CSV 데이터를 파싱할 수 없습니다."

    max_cols = max(len(r) for idx, r in parsed_rows)
    skip_rows = 0

    for idx, r in parsed_rows:
        non_empty = [c for c in r if c.strip() != ""]
        if len(r) == max_cols and len(non_empty) >= 1:
            skip_rows = idx
            break

    logger.info(f"[HEADER] 헤더 줄 감지 결과 (건너뛴 줄 수: {skip_rows})")

    try:
        df = pd.read_csv(io.StringIO(text), skiprows=skip_rows)
    except Exception as e:
        logger.error(f"[ERROR] Pandas read_csv 실패: {str(e)}", exc_info=True)
        return None, None, None, None, f"CSV 읽기 중 오류가 발생했습니다: {str(e)}"

    return df, used_encoding, skip_rows, text, None


@app.route("/", methods=["GET"])
def index():
    logger.info("[REQUEST] GET /")
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    logger.info("[REQUEST] GET /health")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_api_key = bool(api_key and api_key != "your_gemini_api_key_here")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return jsonify({
        "status": "ok",
        "api_key_configured": has_api_key,
        "model": model_name
    })


@app.route("/inspect", methods=["POST"])
def inspect():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /inspect")

    # 1. 파일 검증
    if "file" not in request.files:
        logger.warning("[ERROR] 파일이 전송되지 않음")
        return jsonify({"error": "업로드할 CSV 파일을 선택해 주세요."}), 400

    file = request.files["file"]
    filename = file.filename or ""

    if not filename:
        logger.warning("[ERROR] 파일명이 없음")
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    if not filename.lower().endswith(".csv"):
        logger.warning(f"[ERROR] 허용되지 않은 확장자: {filename}")
        return jsonify({"error": "CSV 형식의 파일(.csv)만 업로드할 수 있습니다."}), 400

    try:
        raw_bytes = file.read()
    except Exception as e:
        logger.error(f"[ERROR] 파일 읽기 실패: {str(e)}", exc_info=True)
        return jsonify({"error": "파일을 읽는 도중 오류가 발생했습니다."}), 400

    file_size = len(raw_bytes)
    logger.info(f"[FILE] name={filename} size={file_size} bytes")

    if file_size == 0:
        logger.warning("[ERROR] 빈 파일 업로드됨")
        return jsonify({"error": "업로드된 파일 내용이 비어 있습니다."}), 400

    # 2. 인코딩 및 헤더 감지 파싱
    df, used_encoding, skip_rows, raw_text, err = detect_encoding_and_parse(raw_bytes)
    if err:
        return jsonify({"error": err}), 400

    logger.info(f"[CSV] parsing 완료 rows={len(df)} columns={len(df.columns)}")

    # 3. 상세 진단(Diagnosis) 계산
    warnings = []
    columns = [str(c) for c in df.columns]

    # Unnamed 및 전체 빈 열
    unnamed_cols = [c for c in columns if c.startswith("Unnamed:")]
    empty_cols = [c for c in columns if df[c].isna().all()]
    if unnamed_cols:
        warnings.append(f"이름이 없는 Unnamed 열이 {len(unnamed_cols)}개 있습니다: {', '.join(unnamed_cols[:3])}")
    if empty_cols:
        warnings.append(f"전체 값이 비어 있는 열이 {len(empty_cols)}개 있습니다: {', '.join(empty_cols[:3])}{'...' if len(empty_cols) > 3 else ''}")

    # 중복 열 검사
    if len(columns) != len(set(columns)):
        warnings.append("중복된 열 이름이 존재합니다.")

    # 중복 행 개수
    duplicate_rows_count = int(df.duplicated().sum())
    if duplicate_rows_count > 0:
        warnings.append(f"완전히 중복된 데이터 행이 {duplicate_rows_count}개 발견되었습니다.")

    # 데이터 타입 분류
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = []
    categorical_cols = []
    year_cols = []

    for c in columns:
        c_clean = c.strip()
        # 연도 열 후보 (4자리 정수형 연도 1900~2100)
        if c_clean.isdigit() and len(c_clean) == 4 and 1900 <= int(c_clean) <= 2100:
            year_cols.append(c)

        # 숫자형/문자형 열 분류
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    # 열별 결측치 개수 및 비율
    missing_values = {}
    for c in columns:
        na_cnt = int(df[c].isna().sum())
        na_pct = round((na_cnt / len(df)) * 100, 2) if len(df) > 0 else 0.0
        missing_values[c] = {"count": na_cnt, "percentage": na_pct}

    # 연도별 데이터 충실도 (Data Coverage) 및 권장 기준연도
    year_coverage = {}
    for y in year_cols:
        valid_cnt = int(df[y].notna().sum())
        cov_pct = round((valid_cnt / len(df)) * 100, 1) if len(df) > 0 else 0.0
        year_coverage[y] = {
            "valid_count": valid_cnt,
            "total": len(df),
            "percentage": cov_pct
        }

    # 권장 기준연도 계산 (충실도 70% 이상인 가장 최근 연도, 없으면 유효치가 가장 많은 최근 연도)
    recommended_reference_year = None
    sorted_years_desc = sorted(year_cols, key=lambda x: int(x), reverse=True)
    for y in sorted_years_desc:
        if year_coverage[y]["percentage"] >= 70.0:
            recommended_reference_year = y
            break
    if not recommended_reference_year and sorted_years_desc:
        recommended_reference_year = max(sorted_years_desc, key=lambda y: year_coverage[y]["valid_count"])

    # 집계 행 후보 탐지 (World, OECD members, Euro area, European Union, High income, Low income, income, area)
    agg_keywords = ["world", "oecd members", "euro area", "european union", "high income", "low income", "income", "area"]
    aggregate_rows = []

    # 첫 번째 문자형 열을 라벨(국가명 등) 열로 간주
    label_col = categorical_cols[0] if categorical_cols else columns[0]
    for idx, val in df[label_col].items():
        if pd.notna(val):
            val_str = str(val).lower()
            matched = [kw for kw in agg_keywords if kw in val_str]
            if matched:
                aggregate_rows.append({
                    "row_index": int(idx),
                    "name": str(val),
                    "matched_keyword": matched[0]
                })

    if aggregate_rows:
        warnings.append(f"개별 국가가 아닌 대륙/소득 집계 행 후보가 {len(aggregate_rows)}개 감지되었습니다 (예: {aggregate_rows[0]['name']}). 순위 분석 시 제외를 권장합니다.")

    # 미리보기 10행 (안전한 JSON 직렬화)
    preview_df = df.head(10).copy()
    preview = []
    for record in preview_df.to_dict(orient="records"):
        sanitized_record = {k: sanitize_value(v) for k, v in record.items()}
        preview.append(sanitized_record)

    logger.info(f"[DIAGNOSE] 진단 완료 warnings={len(warnings)}")

    # 4. 메모리에 데이터셋 보관
    CURRENT_DATASET = {
        "df": df,
        "filename": filename,
        "file_size": file_size,
        "encoding": used_encoding,
        "skip_rows": skip_rows,
        "raw_text": raw_text,
        "label_col": label_col,
        "year_cols": year_cols,
        "recommended_year": recommended_reference_year,
        "aggregate_indices": [item["row_index"] for item in aggregate_rows]
    }

    # 5. 전체 13개 진단 항목 JSON 반환
    return jsonify({
        "filename": filename,
        "file_size": file_size,
        "encoding": used_encoding,
        "skip_rows": skip_rows,
        "rows": len(df),
        "cols": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "year_columns": year_cols,
        "missing_values": missing_values,
        "duplicate_rows_count": duplicate_rows_count,
        "unnamed_columns": unnamed_cols,
        "empty_columns": empty_cols,
        "year_coverage": year_coverage,
        "recommended_reference_year": recommended_reference_year,
        "aggregate_rows": aggregate_rows,
        "preview": preview,
        "warnings": warnings,
        "storage_note": "업로드된 파일은 메모리에만 임시 보관되며 서버를 재시작하면 사라집니다."
    })


@app.route("/suggest", methods=["POST"])
def suggest():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /suggest")

    if not CURRENT_DATASET or CURRENT_DATASET.get("df") is None:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 제안 요청")
        return jsonify({"error": "분석할 데이터가 없습니다. 먼저 CSV 파일을 업로드하고 진단을 완료해 주세요."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] GEMINI_API_KEY 미설정")
        return jsonify({"error": "Gemini API Key가 .env 파일에 설정되어 있지 않습니다."}), 400

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    df = CURRENT_DATASET["df"]
    filename = CURRENT_DATASET["filename"]
    rec_year = CURRENT_DATASET.get("recommended_year") or "2024"
    label_col = CURRENT_DATASET.get("label_col", df.columns[0])
    columns_list = [str(c) for c in df.columns]

    # 도구 카탈로그 요약 텍스트
    catalog_desc = []
    for k, v in TOOL_CATALOG.items():
        catalog_desc.append(f"- {k}: {v['name']} ({v['description']}, 기준시점 필수={v['requires_reference_period']})")
    catalog_text = "\n".join(catalog_desc)

    prompt = f"""당신은 데이터 분석 전문가입니다.
사용자가 업로드한 CSV 데이터의 진단 정보와 실행 가능한 도구 카탈로그를 확인하고,
이 데이터로 실제로 실행해볼 수 있는 최적의 분석 3~5개를 골라 제안해 주세요.

[절대 규칙]
1. 반드시 아래 도구 카탈로그에 존재하는 도구(tool 키값)만 사용해야 합니다. 카탈로그에 없는 도구는 제안할 수 없습니다.
2. 데이터셋에 실제 존재하는 열 이름만 파라미터로 지정해야 합니다.
3. 직접 수치를 계산하거나 추정하지 마십시오. (계산은 Python Pandas가 수행합니다.)
4. 기준 시점(reference_period)이 필요한 도구(top_bottom_n, group_summary, change_rate, distribution, correlation_scatter, compare_one_vs_all)는 권장 기준연도('{rec_year}')를 반드시 파라미터와 reference_period 필드에 넣으십시오.
5. 제안 개수는 반드시 3개 이상 5개 이하로 작성하십시오.
6. 반드시 지정된 JSON 형식으로만 응답하십시오. 마크다운 코드블록(```json) 없이 순수 JSON 문자열만 출력하십시오.

[도구 카탈로그]
{catalog_text}

[데이터셋 진단 요약]
- 파일명: {filename}
- 행 수: {len(df)}행, 열 수: {len(df.columns)}열
- 대표 라벨 열: {label_col}
- 열 목록: {', '.join(columns_list[:20])}{'...' if len(columns_list) > 20 else ''}
- 권장 기준연도: {rec_year} (충실도가 가장 높은 안정적 최근 연도)
- 제외 권장 집계 행: World, Arab World, High income 등 (exclude_aggregates=True 권장)

[응답 JSON 규격]
{{
  "suggestions": [
    {{
      "tool": "도구_키값",
      "params": {{"value_column": "{rec_year}", "label_column": "{label_col}", "n": 10, "reference_period": "{rec_year}", "exclude_aggregates": true}},
      "reference_period": "{rec_year}",
      "why": "이 분석이 이 데이터에서 의미 있는 이유",
      "caution": "해석할 때 주의할 점"
    }}
  ]
}}
"""

    logger.info(f"[AI] suggest 요청 시작 (model={model_name})")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        logger.info("[AI] suggest 응답 수신 완료")
    except Exception as e:
        logger.error(f"[ERROR] Gemini API 호출 실패: {str(e)}", exc_info=True)
        return jsonify({"error": f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}"}), 500

    # JSON 파싱 및 카탈로그 필터링 검증
    try:
        raw_text = response.text.strip()
        parsed_json = json.loads(raw_text)
        suggestions_raw = parsed_json.get("suggestions", [])
    except Exception as e:
        logger.error(f"[ERROR] JSON 파싱 실패: {str(e)}", exc_info=True)
        return jsonify({"error": "AI의 응답을 JSON 형식으로 해석할 수 없습니다."}), 500

    valid_suggestions = []
    for s in suggestions_raw:
        tool_name = s.get("tool")
        if tool_name in TOOL_CATALOG:
            if TOOL_CATALOG[tool_name]["requires_reference_period"] and not s.get("reference_period"):
                s["reference_period"] = rec_year
                if "params" in s and isinstance(s["params"], dict):
                    s["params"]["reference_period"] = rec_year
            valid_suggestions.append(s)

    # 3개 미만인 경우 카탈로그의 대표 도구로 기본 보충
    fallback_defaults = [
        {"tool": "top_bottom_n", "params": {"reference_period": rec_year, "n": 10, "exclude_aggregates": True}, "reference_period": rec_year, "why": f"{rec_year}년 기준 글로벌 상위 10개국 순위 분석", "caution": "대륙 집계 행 제외 필수"},
        {"tool": "trend_line", "params": {"items": ["Korea, Rep.", "United States", "Japan", "World"]}, "reference_period": "1990~2024", "why": "주요국 및 전 세계 평균의 연도별 시계열 변화 궤적 비교", "caution": "결측 연도 주의"},
        {"tool": "group_summary", "params": {"reference_period": rec_year, "exclude_aggregates": True}, "reference_period": rec_year, "why": f"{rec_year}년 대륙별 인터넷 이용률 평균 격차 분석", "caution": "인구 가중치 미반영 산술평균"}
    ]
    for fb in fallback_defaults:
        if len(valid_suggestions) >= 3:
            break
        if not any(item.get("tool") == fb["tool"] for item in valid_suggestions):
            valid_suggestions.append(fb)

    valid_suggestions = valid_suggestions[:5]
    logger.info(f"[AI] JSON 파싱 결과: {len(valid_suggestions)}개 제안 확정")

    return jsonify({"suggestions": valid_suggestions})


@app.route("/run", methods=["POST"])
def run_tool():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /run")

    if not CURRENT_DATASET or CURRENT_DATASET.get("df") is None:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 도구 실행 요청")
        return jsonify({"error": "분석할 데이터가 없습니다. 먼저 CSV 파일을 업로드해 주세요."}), 400

    data = request.get_json(silent=True) or {}
    tool_name = data.get("tool")
    if not tool_name:
        return jsonify({"error": "실행할 도구(tool)를 지정해 주세요."}), 400

    if tool_name not in TOOL_CATALOG:
        logger.warning(f"[ERROR] 등록되지 않은 도구 요청: {tool_name}")
        return jsonify({"error": f"도구 '{tool_name}'은(는) 지원되지 않는 도구입니다. 카탈로그에 등록된 도구를 선택해 주세요."}), 400

    # 파라미터 병합
    params = dict(data.get("params", {}))
    if "reference_period" in data and "reference_period" not in params:
        params["reference_period"] = data["reference_period"]
    if "exclude_aggregates" in data and "exclude_aggregates" not in params:
        params["exclude_aggregates"] = data["exclude_aggregates"]

    # 기준시점이 필수인 도구 검증
    if TOOL_CATALOG[tool_name]["requires_reference_period"] and not params.get("reference_period"):
        rec_year = CURRENT_DATASET.get("recommended_year")
        if rec_year:
            params["reference_period"] = rec_year
        else:
            return jsonify({"error": f"도구 '{tool_name}'은(는) 기준 시점(reference_period)을 필수로 입력해야 합니다."}), 400

    df = CURRENT_DATASET["df"]
    tool_fn = TOOL_CATALOG[tool_name]["fn"]

    try:
        run_result = tool_fn(df, **params)
    except ValueError as ve:
        logger.warning(f"[ERROR] 도구 실행 유효성 검사 실패: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"[ERROR] 도구 실행 실패 ({tool_name}): {str(e)}", exc_info=True)
        return jsonify({"error": f"분석 계산 중 오류가 발생했습니다: {str(e)}"}), 500

    used_rows = run_result.get("used_rows_count", 0)
    excluded_rows = run_result.get("excluded_rows_count", 0)
    logger.info(f"[TOOL] name={tool_name}, params={params}, used_rows={used_rows}, excluded_rows={excluded_rows}")
    logger.info("[RESPONSE] status=200")

    # 최근 계산 결과 메모리 보관 (POST /explain 연계용)
    CURRENT_DATASET["last_run_result"] = run_result

    return jsonify(run_result)


def get_indicator_metadata():
    """data/ 폴더에 위치한 Metadata_Indicator 파일로부터 공식 정의와 출처 로드"""
    if os.path.exists("data"):
        meta_files = [f for f in os.listdir("data") if "metadata_indicator" in f.lower() and f.endswith(".csv")]
        if meta_files:
            try:
                mdf = pd.read_csv(os.path.join("data", meta_files[0]), encoding="utf-8-sig")
                source_note = mdf["SOURCE_NOTE"].dropna().iloc[0] if "SOURCE_NOTE" in mdf.columns else ""
                source_org = mdf["SOURCE_ORGANIZATION"].dropna().iloc[0] if "SOURCE_ORGANIZATION" in mdf.columns else ""
                indicator_name = mdf["INDICATOR_NAME"].dropna().iloc[0] if "INDICATOR_NAME" in mdf.columns else ""
                return {
                    "name": str(indicator_name),
                    "note": str(source_note),
                    "org": str(source_org)
                }
            except Exception:
                pass
    return {
        "name": "인구 중 인터넷 이용자 비율 (% of population)",
        "note": "지난 3개월 동안 장소에 구애받지 않고 컴퓨터, 휴대폰 등을 통해 인터넷을 사용한 사람의 비율",
        "org": "World Telecommunication/ICT Indicators Database, International Telecommunication Union (ITU)"
    }


@app.route("/explain", methods=["POST"])
def explain():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /explain")

    if not CURRENT_DATASET or CURRENT_DATASET.get("df") is None:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 해석 요청")
        return jsonify({"error": "분석할 데이터가 없습니다. 먼저 CSV 파일을 업로드해 주세요."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] GEMINI_API_KEY 미설정")
        return jsonify({"error": "Gemini API Key가 .env 파일에 설정되어 있지 않습니다."}), 400

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "B").upper()
    if mode not in ["A", "B"]:
        mode = "B"

    df = CURRENT_DATASET["df"]
    filename = CURRENT_DATASET["filename"]

    # 모드 A (비교용, 의도적으로 허술하게): 계산 결과를 주지 않고 진단 요약만 전달
    if mode == "A":
        columns_summary = [str(c) for c in df.columns[:25]]
        prompt = f"""당신은 데이터 분석 보고서 작성자입니다.
사용자가 업로드한 데이터 파일의 요약 정보는 다음과 같습니다:
- 파일명: {filename}
- 행 수: {len(df)}행, 열 수: {len(df.columns)}열
- 주요 열 목록: {', '.join(columns_summary)}{'...' if len(df.columns) > 25 else ''}

이 데이터를 바탕으로 자유롭게 종합 분석 보고서를 작성해 주세요.
(주의: 별도의 Pandas 정밀 계산 결과는 제공되지 않으므로, 데이터의 맥락을 추정하여 작성하세요.)
"""
        run_result_info = None

    # 모드 B (정식): Pandas 계산 결과만 엄격히 전달하고 8대 규칙 적용
    else:
        run_result = data.get("run_result") or CURRENT_DATASET.get("last_run_result")
        if not run_result:
            return jsonify({"error": "해석할 분석 계산 결과가 없습니다. 먼저 분석 도구를 실행(/run)해 주세요."}), 400

        tool_name = run_result.get("tool", "알 수 없음")
        meta_info = get_indicator_metadata()

        prompt = f"""당신은 고도로 정밀하고 정직한 Senior Data Analyst입니다.
Python Pandas가 실제로 계산을 마친 아래의 [Pandas 계산 결과] 데이터만 엄격하게 사용하여 분석 보고서를 작성하세요.

[절대 규칙 - 반드시 준수]
1. 제공된 [Pandas 계산 결과]에 명시된 숫자만 사용하십시오. 직접 계산하거나 추정하지 마십시오.
2. 데이터에 없는 국가·연도·열을 절대 언급하지 마십시오.
3. 기준 시점과 제외된 행 수, 제외 이유를 반드시 본문에 명시하십시오.
4. 상관관계를 인과관계로 절대 표현하지 마십시오. (A가 높다고 해서 B의 원인이 아님)
5. 지표의 정의와 출처를 보고서 앞부분('## 데이터 출처와 정의')에 정확히 표기하십시오.
6. 데이터로 확인할 수 없거나 알 수 없는 사실은 반드시 "이 데이터로는 확인할 수 없다"라고 솔직하게 작성하십시오.
7. 중학생도 명확하게 이해할 수 있는 쉬운 문장으로 쓰되, 통계적 정확성을 절대 희생하지 마십시오.
8. 반드시 아래 지정된 7개 마크다운 헤더 형식을 순서대로 그대로 출력하십시오.

[지표 공식 정의 및 출처]
- 지표명: {meta_info['name']}
- 지표 정의: {meta_info['note']}
- 출처 기관: {meta_info['org']}

[Pandas 실행 도구 및 계산 결과]
- 실행된 도구: {tool_name}
- 기준 시점: {run_result.get('reference_period')}
- 사용한 행 수: {run_result.get('used_rows_count')}개
- 제외한 행 수: {run_result.get('excluded_rows_count')}개
- 제외 이유: {run_result.get('excluded_reason')}
- 단위: {run_result.get('unit')}
- 주의사항: {run_result.get('caution')}
- 세부 수치 결과:
{json.dumps(run_result.get('result', {}), ensure_ascii=False, indent=2)}

[반드시 사용할 출력 형식]
## 무엇을 봤는가
## 데이터 출처와 정의
## 기준 시점과 제외된 항목
## 계산 결과
## 읽어낼 수 있는 것
## 이 데이터로는 알 수 없는 것
## 다음에 확인해볼 것
"""
        run_result_info = run_result

    logger.info(f"[AI] explain 요청 시작 (mode={mode}, model={model_name})")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        logger.info("[AI] explain 응답 수신 완료")
        explanation_text = response.text
    except Exception as e:
        logger.error(f"[ERROR] Gemini explain 호출 실패: {str(e)}", exc_info=True)
        return jsonify({"error": f"Gemini 해석 생성 중 오류가 발생했습니다: {str(e)}"}), 500

    # 후속 대화(POST /chat)에서 활용할 수 있도록 보관
    CURRENT_DATASET["last_explanation"] = explanation_text
    CURRENT_DATASET["last_mode"] = mode

    logger.info("[RESPONSE] status=200")
    return jsonify({
        "mode": mode,
        "explanation": explanation_text,
        "run_result": run_result_info
    })


@app.route("/chat", methods=["POST"])
def chat():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /chat")

    if not CURRENT_DATASET or CURRENT_DATASET.get("df") is None:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 chat 요청")
        return jsonify({"error": "분석할 데이터가 없습니다. 먼저 CSV 파일을 업로드해 주세요."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] GEMINI_API_KEY 미설정")
        return jsonify({"error": "Gemini API Key가 .env 파일에 설정되어 있지 않습니다."}), 400

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    data = request.get_json(silent=True) or {}
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "질문 내용을 입력해 주세요."}), 400

    last_run = CURRENT_DATASET.get("last_run_result") or {}
    last_exp = CURRENT_DATASET.get("last_explanation") or ""
    meta_info = get_indicator_metadata()

    prompt = f"""당신은 데이터 리터러시를 가르치는 친절하고 정직한 AI 데이터 멘토입니다.
사용자가 최근 수행한 데이터 분석 결과 및 보고서에 대해 후속 질문을 했습니다.

[절대 규칙]
1. 아래 [Pandas 계산 결과]에 명시된 숫자만 사용하고, 새로운 숫자를 지어내거나 추정하지 마십시오.
2. "쉽게 설명해줘", "중학생도 이해할 수 있게" 같은 요청이 오면 일상 속 쉬운 비유(성적표, 카페, 달리기 등)를 적극적으로 사용하여 초보자 눈높이에서 명쾌하고 따뜻하게 설명하십시오.
3. 상관관계와 인과관계를 절대 혼동하여 설명하지 마십시오.
4. 데이터로 확인 불가능한 부분은 솔직하게 "이 데이터셋으로는 확인할 수 없습니다"라고 답하십시오.

[지표 정의 및 출처]
- 지표명: {meta_info.get('name')}
- 지표 정의: {meta_info.get('note')}
- 출처: {meta_info.get('org')}

[Pandas 실행 도구 및 계산 결과]
{json.dumps(last_run, ensure_ascii=False, indent=2)}

[이전에 작성된 보고서 요약]
{last_exp[:1500]}

[사용자의 후속 질문]
{user_question}
"""

    logger.info(f"[AI] chat 요청 시작: question='{user_question[:30]}...'")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        answer_text = response.text
        logger.info("[AI] chat 응답 수신 완료")
    except Exception as e:
        logger.error(f"[ERROR] Gemini chat 호출 실패: {str(e)}", exc_info=True)
        return jsonify({"error": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}), 500

    logger.info("[RESPONSE] status=200")
    return jsonify({
        "question": user_question,
        "answer": answer_text
    })


@app.route("/query", methods=["POST"])
def query():
    global CURRENT_DATASET
    logger.info("[REQUEST] POST /query")

    if not CURRENT_DATASET or CURRENT_DATASET.get("df") is None:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 자연어 질문 요청")
        return jsonify({"error": "분석할 데이터가 없습니다. 먼저 CSV 파일을 업로드해 주세요."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] GEMINI_API_KEY 미설정")
        return jsonify({"error": "Gemini API Key가 .env 파일에 설정되어 있지 않습니다."}), 400

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    data = request.get_json(silent=True) or {}
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "질문 내용을 입력해 주세요."}), 400

    df = CURRENT_DATASET["df"]
    rec_year = CURRENT_DATASET.get("recommended_year") or "2024"
    label_col = CURRENT_DATASET.get("label_col", df.columns[0])

    # 1. Gemini를 통해 자연어 질문에 가장 적합한 도구 1개 및 파라미터 매핑
    catalog_desc = []
    for k, v in TOOL_CATALOG.items():
        catalog_desc.append(f"- {k}: {v['name']} ({v['description']}, 기준시점 필수={v['requires_reference_period']})")
    catalog_text = "\n".join(catalog_desc)

    prompt_map = f"""당신은 데이터 분석 오케스트레이터입니다.
사용자의 자연어 질문을 해결하기 위해 도구 카탈로그에서 실행할 도구 1개와 파라미터를 JSON으로 선택하세요.

[도구 카탈로그]
{catalog_text}

[사용자 질문]
{user_question}

[권장 기준연도]
{rec_year}

[대표 라벨 열]
{label_col}

[규칙]
1. 반드시 도구 카탈로그에 존재하는 도구(tool) 1개만 고르십시오.
2. 기준시점이 필요한 도구는 질문에 특정 연도가 있다면 그 연도를, 없다면 권장 기준연도('{rec_year}')를 사용하십시오.
3. exclude_aggregates는 특별한 요청이 없는 한 true로 설정하십시오.
4. 사용자에게 보여줄 친절한 시각화 버튼 문구(button_text, 예: '📊 2024년 상위 10개국 및 한국 순위 차트 보기')를 포함하십시오.
5. 반드시 순수 JSON만 출력하십시오.

[응답 JSON 형식]
{{
  "tool": "top_bottom_n",
  "params": {{"reference_period": "{rec_year}", "n": 10, "exclude_aggregates": true}},
  "button_text": "📊 2024년 상위 10개국 및 한국 순위 차트 보기"
}}
"""

    logger.info(f"[AI] query 매핑 요청 시작: question='{user_question[:30]}...'")
    try:
        client = genai.Client(api_key=api_key)
        resp_map = client.models.generate_content(
            model=model_name,
            contents=prompt_map,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        map_json = json.loads(resp_map.text.strip())
        mapped_tool = map_json.get("tool")
        mapped_params = map_json.get("params", {})
        button_text = map_json.get("button_text") or "📊 시각화 차트 보기"
    except Exception as e:
        logger.error(f"[ERROR] Gemini 도구 매핑 실패: {str(e)}", exc_info=True)
        mapped_tool = "top_bottom_n"
        mapped_params = {"reference_period": rec_year, "n": 10, "exclude_aggregates": True}
        button_text = f"📊 {rec_year}년 상위 10개국 순위 차트 보기"

    if mapped_tool not in TOOL_CATALOG:
        mapped_tool = "top_bottom_n"
        mapped_params = {"reference_period": rec_year, "n": 10, "exclude_aggregates": True}

    # 2. Pandas로 실제 수치 계산 (절대 규칙: 계산은 Pandas가 수행)
    tool_fn = TOOL_CATALOG[mapped_tool]["fn"]
    try:
        run_result = tool_fn(df, **mapped_params)
    except Exception as e:
        logger.error(f"[ERROR] Pandas 계산 실행 실패: {str(e)}", exc_info=True)
        return jsonify({"error": f"데이터 계산 중 오류가 발생했습니다: {str(e)}"}), 500

    # 3. 계산된 Pandas 수치를 바탕으로 Gemini가 자연어 답변 작성
    meta_info = get_indicator_metadata()
    prompt_answer = f"""당신은 신뢰할 수 있는 전문 데이터 분석관입니다.
사용자의 질문에 대해 아래 Python Pandas가 계산한 실제 수치만을 엄격히 바탕으로 친절하고 명쾌하게 답변해 주세요.

[절대 규칙]
1. 아래 [Pandas 계산 결과]에 있는 실제 숫자만 사용하십시오. 직접 계산하거나 추측하지 마십시오.
2. 사용자가 1~10위나 한국 비교를 물어본 경우, 1위 국가부터 순위와 수치, 그리고 한국의 순위와 수치를 명확히 짚어주십시오.
3. 중학생도 쉽게 이해할 수 있는 자연스럽고 신뢰감 있는 문장으로 작성하십시오.

[지표 정의]
- {meta_info.get('name')}: {meta_info.get('note')}

[사용자 질문]
{user_question}

[Pandas 실제 계산 결과]
- 실행 도구: {mapped_tool}
- 기준 시점: {run_result.get('reference_period')}
- 사용 행 수: {run_result.get('used_rows_count')}개, 제외 행 수: {run_result.get('excluded_rows_count')}개 ({run_result.get('excluded_reason')})
- 세부 결과:
{json.dumps(run_result.get('result', {}), ensure_ascii=False, indent=2)}
"""

    try:
        resp_ans = client.models.generate_content(
            model=model_name,
            contents=prompt_answer
        )
        answer_text = resp_ans.text.strip()
    except Exception as e:
        logger.error(f"[ERROR] Gemini 답변 작성 실패: {str(e)}", exc_info=True)
        answer_text = f"계산이 완료되었습니다. ({mapped_tool})"

    # 최근 계산 결과로 저장 (다음 단계 연계용)
    CURRENT_DATASET["last_run_result"] = run_result

    logger.info("[RESPONSE] status=200")
    return jsonify({
        "question": user_question,
        "answer": answer_text,
        "tool": mapped_tool,
        "params": mapped_params,
        "button_text": button_text,
        "run_result": run_result
    })


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    logger.info(f"[START] Starting Flask server on http://127.0.0.1:5000 (debug={debug_mode})")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
