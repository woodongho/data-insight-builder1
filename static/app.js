/**
 * AI Data Insight Builder - Frontend Vanilla JavaScript
 * Console log format:
 * [PAGE], [UPLOAD], [INSPECT], [SUGGEST], [RUN], [CHART], [EXPLAIN], [ERROR]
 */

document.addEventListener("DOMContentLoaded", () => {
    console.log("[PAGE] 페이지 로드 완료");

    // 전역 상태 변수
    let currentFile = null;
    let inspectionData = null;
    let currentChartInstance = null;
    let lastRunResult = null;
    let rawExplanationText = "";

    // 12대 도구 옵션 목록
    const TOOL_OPTIONS = [
        { id: "top_bottom_n", name: "7. 상위/하위 N개국 순위 비교 (수평 막대)", requiresRef: true },
        { id: "trend_line", name: "6. 연도별 시계열 추이 (선 차트)", requiresRef: false },
        { id: "group_summary", name: "8. 대륙/소득군 그룹별 요약 (막대)", requiresRef: true },
        { id: "change_rate", name: "9. 두 시점 간 변화량/성장률 (막대)", requiresRef: true },
        { id: "describe_numeric", name: "5. 기술통계량 분석 (평균·중앙값 등)", requiresRef: false },
        { id: "distribution", name: "10. 구간별 국가 분포 (히스토그램)", requiresRef: true },
        { id: "correlation_scatter", name: "11. 두 시점 간 상관계수 (산점도)", requiresRef: true },
        { id: "compare_one_vs_all", name: "12. 특정 항목 vs 전체 평균 대조 (강조 막대)", requiresRef: true },
        { id: "year_coverage", name: "3. 연도별 데이터 충실도 분석", requiresRef: false },
        { id: "detect_aggregates", name: "4. 대륙/소득군 집계 행 탐지", requiresRef: false },
        { id: "quality_report", name: "2. 데이터 품질 진단 보고서", requiresRef: false },
        { id: "reshape_wide_to_long", name: "1. 연도 열을 행으로 접기 (Tidy Data)", requiresRef: false }
    ];

    // DOM 요소 캐싱
    const serverStatusBadge = document.getElementById("serverStatusBadge");
    const apiKeyStatusBadge = document.getElementById("apiKeyStatusBadge");
    const alertBanner = document.getElementById("alertBanner");

    const dropZone = document.getElementById("dropZone");
    const csvFileInput = document.getElementById("csvFileInput");
    const selectedFileInfo = document.getElementById("selectedFileInfo");
    const selectedFileName = document.getElementById("selectedFileName");
    const selectedFileSize = document.getElementById("selectedFileSize");
    const inspectBtn = document.getElementById("inspectBtn");

    const diagnosisContainer = document.getElementById("diagnosisContainer");
    const diagFilename = document.getElementById("diagFilename");
    const diagEncoding = document.getElementById("diagEncoding");
    const diagSkipRows = document.getElementById("diagSkipRows");
    const diagShape = document.getElementById("diagShape");
    const diagRecYear = document.getElementById("diagRecYear");
    const warningBox = document.getElementById("warningBox");
    const warningList = document.getElementById("warningList");
    const previewThead = document.getElementById("previewThead");
    const previewTbody = document.getElementById("previewTbody");

    // 자연어 질문 (Ask Section) 요소
    const askSection = document.getElementById("askSection");
    const askForm = document.getElementById("askForm");
    const askInput = document.getElementById("askInput");
    const askSubmitBtn = document.getElementById("askSubmitBtn");
    const askLoader = document.getElementById("askLoader");
    const askResultCard = document.getElementById("askResultCard");
    const askAnswerContent = document.getElementById("askAnswerContent");
    const askVisualizeBtn = document.getElementById("askVisualizeBtn");
    const askChips = document.querySelectorAll(".ask-chip");

    const step2Section = document.getElementById("step2Section");
    const suggestBtn = document.getElementById("suggestBtn");
    const suggestionsLoader = document.getElementById("suggestionsLoader");
    const suggestionsContainer = document.getElementById("suggestionsContainer");

    const step3Section = document.getElementById("step3Section");
    const runForm = document.getElementById("runForm");
    const toolSelect = document.getElementById("toolSelect");
    const refPeriodInput = document.getElementById("refPeriodInput");
    const nParamInput = document.getElementById("nParamInput");
    const excludeAggCheck = document.getElementById("excludeAggCheck");
    const runResultContainer = document.getElementById("runResultContainer");
    const resMetaPeriod = document.getElementById("resMetaPeriod");
    const resMetaUsed = document.getElementById("resMetaUsed");
    const resMetaExcluded = document.getElementById("resMetaExcluded");
    const resMetaUnit = document.getElementById("resMetaUnit");
    const resExcludedReasonBox = document.getElementById("resExcludedReasonBox");
    const resultDetails = document.getElementById("resultDetails");
    const resCautionBox = document.getElementById("resCautionBox");

    const step4Section = document.getElementById("step4Section");
    const explainBtn = document.getElementById("explainBtn");
    const explainLoader = document.getElementById("explainLoader");
    const explanationContainer = document.getElementById("explanationContainer");
    const explainModeBadge = document.getElementById("explainModeBadge");
    const markdownOutput = document.getElementById("markdownOutput");
    const copyReportBtn = document.getElementById("copyReportBtn");

    // -------------------------------------------------------------
    // 초기화: GET /health 확인 및 도구 드롭다운 채우기
    // -------------------------------------------------------------
    initToolSelect();
    checkHealth();

    function checkHealth() {
        fetch("/health")
            .then(res => res.json())
            .then(data => {
                if (data.status === "ok") {
                    serverStatusBadge.textContent = "서버 정상 연결 (OK)";
                    serverStatusBadge.className = "status-badge ok";
                } else {
                    serverStatusBadge.textContent = "서버 응답 비정상";
                    serverStatusBadge.className = "status-badge error";
                }

                if (data.api_key_configured) {
                    apiKeyStatusBadge.textContent = `API Key 연결됨 (${data.model})`;
                    apiKeyStatusBadge.className = "status-badge ok";
                } else {
                    apiKeyStatusBadge.textContent = "API Key 미설정 (.env 확인 필요)";
                    apiKeyStatusBadge.className = "status-badge error";
                }
            })
            .catch(err => {
                console.error("[ERROR] Health check 통신 실패:", err);
                serverStatusBadge.textContent = "서버 연결 불가";
                serverStatusBadge.className = "status-badge error";
            });
    }

    function initToolSelect() {
        toolSelect.innerHTML = "";
        TOOL_OPTIONS.forEach(opt => {
            const option = document.createElement("option");
            option.value = opt.id;
            option.textContent = opt.name;
            toolSelect.appendChild(option);
        });
    }

    function showAlert(message, type = "error") {
        alertBanner.textContent = message;
        alertBanner.className = `alert-banner ${type}`;
        alertBanner.classList.remove("hidden");
        window.scrollTo({ top: alertBanner.offsetTop - 20, behavior: "smooth" });
    }

    function hideAlert() {
        alertBanner.classList.add("hidden");
    }

    // -------------------------------------------------------------
    // Step 1: 파일 업로드 및 진단 (/inspect)
    // -------------------------------------------------------------
    csvFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file) return;
        hideAlert();

        if (!file.name.toLowerCase().endsWith(".csv")) {
            showAlert("CSV 형식(.csv)의 파일만 업로드할 수 있습니다.");
            console.error("[ERROR] 비정상 확장자 선택:", file.name);
            return;
        }

        currentFile = file;
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = formatBytes(file.size);
        selectedFileInfo.classList.remove("hidden");
        console.log(`[UPLOAD] 파일 선택 (파일명: ${file.name}, 크기: ${file.size} bytes)`);
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    inspectBtn.addEventListener("click", () => {
        if (!currentFile) {
            showAlert("먼저 CSV 파일을 선택해 주세요.");
            return;
        }

        hideAlert();
        inspectBtn.disabled = true;
        inspectBtn.textContent = "진단 분석 중...";
        console.log("[INSPECT] 진단 요청 시작");

        const formData = new FormData();
        formData.append("file", currentFile);

        fetch("/inspect", {
            method: "POST",
            body: formData
        })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || "파일 진단에 실패했습니다.");
                return data;
            })
            .then(data => {
                inspectionData = data;
                const warnCount = data.warnings ? data.warnings.length : 0;
                console.log(`[INSPECT] 응답 수신 (행 수: ${data.rows}, 열 수: ${data.cols}, 경고 수: ${warnCount})`);

                renderDiagnosis(data);
                askSection.classList.remove("hidden");
                step2Section.classList.remove("hidden");
                step3Section.classList.remove("hidden");

                // 권장 기준연도 자동 주입
                if (data.recommended_reference_year) {
                    refPeriodInput.value = data.recommended_reference_year;
                }
            })
            .catch(err => {
                console.error("[ERROR] 진단 요청 실패:", err.message);
                showAlert(err.message);
            })
            .finally(() => {
                inspectBtn.disabled = false;
                inspectBtn.textContent = "다시 진단하기";
            });
    });

    function renderDiagnosis(data) {
        diagFilename.textContent = data.filename;
        diagEncoding.textContent = data.encoding;
        diagSkipRows.textContent = `${data.skip_rows}줄 건너뜀`;
        diagShape.textContent = `${data.rows.toLocaleString()}행 × ${data.cols}열`;
        diagRecYear.textContent = data.recommended_reference_year ? `${data.recommended_reference_year}년` : "판별 불가";

        // 경고 목록
        warningList.innerHTML = "";
        if (data.warnings && data.warnings.length > 0) {
            data.warnings.forEach(w => {
                const li = document.createElement("li");
                li.textContent = w;
                warningList.appendChild(li);
            });
            warningBox.classList.remove("hidden");
        } else {
            warningBox.classList.add("hidden");
        }

        // 미리보기 테이블 (최대 10개 행)
        renderPreviewTable(data.columns, data.preview);
        diagnosisContainer.classList.remove("hidden");
    }

    function renderPreviewTable(columns, previewRows) {
        previewThead.innerHTML = "";
        previewTbody.innerHTML = "";

        if (!columns || !previewRows || previewRows.length === 0) return;

        const headerTr = document.createElement("tr");
        columns.forEach(col => {
            const th = document.createElement("th");
            th.textContent = col;
            headerTr.appendChild(th);
        });
        previewThead.appendChild(headerTr);

        previewRows.forEach(row => {
            const tr = document.createElement("tr");
            columns.forEach(col => {
                const td = document.createElement("td");
                const val = row[col];
                td.textContent = (val === null || val === undefined) ? "" : val;
                tr.appendChild(td);
            });
            previewTbody.appendChild(tr);
        });
    }

    // -------------------------------------------------------------
    // Step 2: AI 제안 요청 (/suggest)
    // -------------------------------------------------------------
    suggestBtn.addEventListener("click", () => {
        hideAlert();
        suggestBtn.disabled = true;
        suggestionsLoader.classList.remove("hidden");
        suggestionsContainer.innerHTML = "";
        console.log("[SUGGEST] 제안 요청 시작");

        fetch("/suggest", { method: "POST" })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || "AI 제안을 받아오지 못했습니다.");
                return data;
            })
            .then(data => {
                const suggestions = data.suggestions || [];
                console.log(`[SUGGEST] 응답 수신 (제안 개수: ${suggestions.length})`);
                renderSuggestions(suggestions);
            })
            .catch(err => {
                console.error("[ERROR] AI 제안 요청 실패:", err.message);
                showAlert(err.message);
            })
            .finally(() => {
                suggestBtn.disabled = false;
                suggestionsLoader.classList.add("hidden");
            });
    });

    function renderSuggestions(suggestions) {
        suggestionsContainer.innerHTML = "";
        if (suggestions.length === 0) {
            suggestionsContainer.innerHTML = "<p>제안된 분석 항목이 없습니다.</p>";
            return;
        }

        suggestions.forEach((sugg, idx) => {
            const card = document.createElement("div");
            card.className = "suggestion-card";

            const toolInfo = TOOL_OPTIONS.find(t => t.id === sugg.tool) || { name: sugg.tool };
            const refPeriodText = sugg.reference_period ? `기준: ${sugg.reference_period}` : "전체 시점";

            card.innerHTML = `
                <div>
                    <div class="suggestion-header">
                        <span class="suggestion-tool-name">${idx + 1}. ${toolInfo.name}</span>
                        <span class="period-badge">${refPeriodText}</span>
                    </div>
                    <p class="suggestion-why">${escapeHtml(sugg.why || "")}</p>
                    <p class="suggestion-caution">⚠️ ${escapeHtml(sugg.caution || "")}</p>
                </div>
                <button type="button" class="btn btn-sm btn-primary select-sugg-btn">이 분석 실행하기</button>
            `;

            const btn = card.querySelector(".select-sugg-btn");
            btn.addEventListener("click", () => {
                applySuggestionToForm(sugg);
            });

            suggestionsContainer.appendChild(card);
        });
    }

    function applySuggestionToForm(sugg) {
        toolSelect.value = sugg.tool;
        const params = sugg.params || {};

        if (sugg.reference_period) {
            refPeriodInput.value = sugg.reference_period;
        } else if (params.reference_period) {
            refPeriodInput.value = params.reference_period;
        }

        if (params.n) {
            nParamInput.value = params.n;
        }

        if (params.exclude_aggregates !== undefined) {
            excludeAggCheck.checked = Boolean(params.exclude_aggregates);
        }

        step3Section.scrollIntoView({ behavior: "smooth" });
        // 자동 실행
        runForm.dispatchEvent(new Event("submit"));
    }

    // -------------------------------------------------------------
    // Step 3: Pandas 계산 실행 (/run)
    // -------------------------------------------------------------
    runForm.addEventListener("submit", (e) => {
        e.preventDefault();
        hideAlert();

        const selectedTool = toolSelect.value;
        const refPeriod = refPeriodInput.value.trim();
        const nVal = parseInt(nParamInput.value, 10) || 10;
        const excludeAgg = excludeAggCheck.checked;

        const toolOpt = TOOL_OPTIONS.find(t => t.id === selectedTool);
        if (toolOpt && toolOpt.requiresRef && !refPeriod) {
            showAlert(`선택하신 '${toolOpt.name}' 도구는 기준 시점(연도)을 반드시 입력해야 합니다.`);
            return;
        }

        const runToolBtn = document.getElementById("runToolBtn");
        runToolBtn.disabled = true;
        runToolBtn.textContent = "Pandas 계산 중...";
        console.log(`[RUN] 도구 실행 시작 (도구명: ${selectedTool}, 기준시점: ${refPeriod || "없음"})`);

        const payload = {
            tool: selectedTool,
            reference_period: refPeriod,
            exclude_aggregates: excludeAgg,
            params: {
                reference_period: refPeriod,
                exclude_aggregates: excludeAgg,
                n: nVal
            }
        };

        fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || "분석 계산에 실패했습니다.");
                return data;
            })
            .then(data => {
                lastRunResult = data;
                console.log(`[RUN] 결과 수신 (사용한 행: ${data.used_rows_count}, 제외된 행: ${data.excluded_rows_count})`);
                renderRunResult(data);
                step4Section.classList.remove("hidden");
            })
            .catch(err => {
                console.error("[ERROR] 도구 실행 실패:", err.message);
                showAlert(err.message);
            })
            .finally(() => {
                runToolBtn.disabled = false;
                runToolBtn.textContent = "⚡ Pandas로 실제 계산 실행";
            });
    });

    function renderRunResult(data) {
        resMetaPeriod.textContent = data.reference_period || "전체 시점";
        resMetaUsed.textContent = `${data.used_rows_count}개 행`;
        resMetaExcluded.textContent = `${data.excluded_rows_count}개 행`;
        resMetaUnit.textContent = data.unit || "-";

        resExcludedReasonBox.textContent = `제외 사유: ${data.excluded_reason || "없음"}`;
        resCautionBox.textContent = `주의사항: ${data.caution || "없음"}`;

        // Chart.js 차트 렌더링
        renderChart(data.chart_data);

        // 결과 요약 텍스트/표 생성
        renderResultDetails(data);
        runResultContainer.classList.remove("hidden");
    }

    function renderChart(chartData) {
        const canvas = document.getElementById("resultChart");
        const chartWrapper = document.getElementById("chartWrapper");

        if (currentChartInstance) {
            currentChartInstance.destroy();
            currentChartInstance = null;
        }

        if (!chartData) {
            chartWrapper.style.display = "none";
            return;
        }

        chartWrapper.style.display = "block";
        const ctx = canvas.getContext("2d");

        currentChartInstance = new Chart(ctx, {
            type: chartData.type,
            data: {
                labels: chartData.labels,
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: chartData.indexAxis || "x",
                plugins: {
                    legend: { display: true, position: "top" }
                }
            }
        });

        console.log(`[CHART] 차트 렌더 완료 (차트 종류: ${chartData.type})`);
    }

    function renderResultDetails(data) {
        resultDetails.innerHTML = "";
        const res = data.result || {};

        // 순위 결과(top_bottom_n)
        if (res.rankings && Array.isArray(res.rankings)) {
            let html = `<h4>순위 데이터 목록 (${res.rankings.length}개)</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>순위</th><th>국가명</th><th>값 (${data.unit || "%"})</th></tr></thead><tbody>`;
            res.rankings.forEach(item => {
                html += `<tr><td><strong>${item.rank}위</strong></td><td>${escapeHtml(item.name)}</td><td>${item.value}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            if (res.korea_status) {
                html += `<p style="margin-top: 0.5rem; font-weight: 600; color: #1e40af;">🇰🇷 대한민국 위치: ${res.korea_status.rank}위 (${res.korea_status.value}%)</p>`;
            }
            resultDetails.innerHTML = html;
            return;
        }

        // 기술통계량(describe_numeric)
        if (res.mean !== undefined && res.median !== undefined) {
            resultDetails.innerHTML = `
                <h4>기술통계 요약 수치</h4>
                <div class="stats-grid">
                    <div class="stat-card"><span class="stat-label">평균값</span><span class="stat-value">${res.mean}</span></div>
                    <div class="stat-card"><span class="stat-label">중앙값</span><span class="stat-value">${res.median}</span></div>
                    <div class="stat-card"><span class="stat-label">표준편차</span><span class="stat-value">${res.std}</span></div>
                    <div class="stat-card"><span class="stat-label">최소값</span><span class="stat-value">${res.min}</span></div>
                    <div class="stat-card"><span class="stat-label">최대값</span><span class="stat-value">${res.max}</span></div>
                </div>
            `;
            return;
        }

        // 기본 JSON 뷰
        resultDetails.innerHTML = `<pre style="background: #f1f5f9; padding: 1rem; border-radius: 8px; font-size: 0.85rem; overflow-x: auto;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
    }

    // -------------------------------------------------------------
    // Step 4: Gemini AI 해석 보고서 (/explain)
    // -------------------------------------------------------------
    explainBtn.addEventListener("click", () => {
        hideAlert();
        const selectedMode = document.querySelector('input[name="explainMode"]:checked')?.value || "B";

        explainBtn.disabled = true;
        explainLoader.classList.remove("hidden");
        explanationContainer.classList.add("hidden");

        const spinnerText = document.getElementById("explainSpinnerText");
        spinnerText.textContent = selectedMode === "B"
            ? "Gemini가 Pandas의 검증된 수치와 7개 규격 목차로 보고서를 작성 중입니다..."
            : "Gemini가 데이터 요약만을 바탕으로 비교용 보고서를 작성 중입니다...";

        console.log(`[EXPLAIN] 해석 요청 시작 (모드: ${selectedMode})`);

        fetch("/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mode: selectedMode,
                run_result: lastRunResult
            })
        })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || "해석 생성에 실패했습니다.");
                return data;
            })
            .then(data => {
                console.log("[EXPLAIN] 응답 수신 완료");
                rawExplanationText = data.explanation || "";
                renderExplanation(data.mode, rawExplanationText);
            })
            .catch(err => {
                console.error("[ERROR] AI 해석 실패:", err.message);
                showAlert(err.message);
            })
            .finally(() => {
                explainBtn.disabled = false;
                explainLoader.classList.add("hidden");
            });
    });

    function renderExplanation(mode, markdownText) {
        explainModeBadge.textContent = mode === "B" ? "모드 B (정식 분석관 모드)" : "모드 A (의도적 비교 모드)";
        explainModeBadge.style.color = mode === "B" ? "#4f46e5" : "#b45309";

        if (window.marked) {
            markdownOutput.innerHTML = marked.parse(markdownText);
        } else {
            markdownOutput.innerHTML = `<pre>${escapeHtml(markdownText)}</pre>`;
        }

        explanationContainer.classList.remove("hidden");
        explanationContainer.scrollIntoView({ behavior: "smooth" });

        // Step 5 후속 Q&A 대화창 활성화 및 맞춤형 말풍선 칩 세팅
        if (step5Section) {
            step5Section.classList.remove("hidden");
            setupContextualChips(lastRunResult);
        }
    }

    // -------------------------------------------------------------
    // Step 5: 후속 Q&A 대화창 및 추천 질문 말풍선 칩
    // -------------------------------------------------------------
    const step5Section = document.getElementById("step5Section");
    const quickChipsContainer = document.getElementById("quickChipsContainer");
    const chatHistory = document.getElementById("chatHistory");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatSendBtn = document.getElementById("chatSendBtn");

    function setupContextualChips(runResult) {
        if (!quickChipsContainer) return;
        quickChipsContainer.innerHTML = "";

        // 반드시 첫 번째로 들어가는 필수 추천 질문
        const chips = [
            { text: "✨ 중학생도 이해할 수 있게 쉽게 설명해줘", isPrimary: true }
        ];

        const tool = runResult ? runResult.tool : "";
        if (tool === "top_bottom_n") {
            chips.push({ text: "🇰🇷 한국이 13위인 이유와 위상을 쉽게 설명해줘" });
            chips.push({ text: "🏆 상위 1위 나라들이 100%인 게 현실적으로 맞아?" });
            chips.push({ text: "⚠️ 제외된 71개 국가는 어떤 이유로 빠진 거야?" });
        } else if (tool === "trend_line") {
            chips.push({ text: "📈 한국의 인터넷 보급률이 가장 급성장한 시기가 언제야?" });
            chips.push({ text: "🌐 세계 평균선과 비교했을 때 한국의 궤적은 어때?" });
        } else if (tool === "group_summary") {
            chips.push({ text: "🌍 대륙별로 인터넷 격차가 이렇게 벌어지는 이유를 쉽게 설명해줘" });
            chips.push({ text: "📊 북미와 아프리카의 이용률 차이가 얼마나 심해?" });
        } else if (tool === "change_rate") {
            chips.push({ text: "🚀 10년 동안 가장 많이 성장한 나라들의 특징을 쉽게 설명해줘" });
            chips.push({ text: "💡 초기값이 낮으면 성장률이 뻥튀기되는 착시를 쉽게 알려줘" });
        } else if (tool === "describe_numeric") {
            chips.push({ text: "📊 평균값과 중앙값이 왜 다른지 쉽게 설명해줘" });
            chips.push({ text: "🎯 전 세계 나라들의 수치가 어디로 쏠려 있는 거야?" });
        } else {
            chips.push({ text: "💡 이 분석 결과에서 가장 눈여겨볼 핵심 포인트를 쉽게 짚어줘" });
            chips.push({ text: "🔍 이 데이터만으로는 알 수 없는 한계점이 뭐야?" });
        }

        chips.forEach(chip => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = chip.isPrimary ? "chip-btn primary" : "chip-btn";
            btn.textContent = chip.text;
            btn.addEventListener("click", () => {
                chatInput.value = chip.text.replace(/^[✨🇰🇷🏆⚠️📈🌐🌍📊🚀💡🎯🔍]\s*/, "");
                chatForm.dispatchEvent(new Event("submit"));
            });
            quickChipsContainer.appendChild(btn);
        });
    }

    // -------------------------------------------------------------
    // 자연어 질문 (POST /query) 및 원클릭 시각화 연계
    // -------------------------------------------------------------
    let lastQueryRunData = null;

    if (askChips && askChips.length > 0) {
        askChips.forEach(chip => {
            chip.addEventListener("click", () => {
                const questionText = chip.textContent.replace(/^[🏆📈🌍📊💡]\s*/, "").trim();
                askInput.value = questionText;
                askForm.dispatchEvent(new Event("submit"));
            });
        });
    }

    if (askForm) {
        askForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const question = askInput.value.trim();
            if (!question) {
                showAlert("질문 내용을 입력해 주세요.");
                return;
            }

            hideAlert();
            askSubmitBtn.disabled = true;
            askSubmitBtn.textContent = "🔍 분석 및 계산 중...";
            askLoader.classList.remove("hidden");
            askResultCard.classList.add("hidden");
            console.log(`[QUERY] 자연어 질문 전송: "${question}"`);

            fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            })
                .then(async res => {
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "자연어 질의 처리에 실패했습니다.");
                    return data;
                })
                .then(data => {
                    console.log(`[QUERY] 응답 수신 완료 (매핑 도구: ${data.tool})`);
                    lastQueryRunData = data;

                    // 답변 렌더링 (Markdown 파싱)
                    if (window.marked) {
                        askAnswerContent.innerHTML = marked.parse(data.answer);
                    } else {
                        askAnswerContent.textContent = data.answer;
                    }

                    // 시각화 버튼 문구 동적 반영
                    if (data.button_text) {
                        askVisualizeBtn.textContent = data.button_text;
                    }

                    askResultCard.classList.remove("hidden");
                })
                .catch(err => {
                    console.error("[ERROR] 자연어 질문 처리 실패:", err.message);
                    showAlert(err.message);
                })
                .finally(() => {
                    askSubmitBtn.disabled = false;
                    askSubmitBtn.textContent = "🔍 질문하고 데이터 확인하기";
                    askLoader.classList.add("hidden");
                });
        });
    }

    if (askVisualizeBtn) {
        askVisualizeBtn.addEventListener("click", () => {
            if (!lastQueryRunData || !lastQueryRunData.run_result) {
                showAlert("시각화할 계산 결과가 없습니다. 먼저 질문을 실행해 주세요.");
                return;
            }

            const runResult = lastQueryRunData.run_result;
            const mappedTool = lastQueryRunData.tool;
            const mappedParams = lastQueryRunData.params || {};

            console.log(`[CHART] 자연어 질의 기반 시각화 연계 실행 (도구: ${mappedTool})`);

            // 1. 3단계 폼 값 자동 동기화
            toolSelect.value = mappedTool;
            toolSelect.dispatchEvent(new Event("change"));
            if (mappedParams.reference_period) {
                refPeriodInput.value = mappedParams.reference_period;
            }
            if (mappedParams.n) {
                nParamInput.value = mappedParams.n;
            }
            if (mappedParams.exclude_aggregates !== undefined) {
                excludeAggCheck.checked = !!mappedParams.exclude_aggregates;
            }

            // 2. 전역 상태 갱신 및 차트 렌더링
            lastRunResult = runResult;
            renderRunResult(runResult);

            // 3. 4단계(AI 보고서 해석) 및 5단계(후속 Q&A 대화) 활성화
            step4Section.classList.remove("hidden");
            setupContextualChips(runResult);

            // 4. 시각화 결과 위치로 부드럽게 스크롤 이동
            step3Section.scrollIntoView({ behavior: "smooth" });
        });
    }

    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const question = chatInput.value.trim();
            if (!question) return;

            hideAlert();
            chatInput.value = "";
            chatSendBtn.disabled = true;

            // 1. 사용자 질문 말풍선 렌더링
            appendChatMessage("user", "나", question);
            console.log(`[CHAT] 후속 질문 전송: "${question}"`);

            // 2. AI 답변 대기 임시 말풍선 렌더링
            const loadingMsgId = "loadingMsg_" + Date.now();
            appendChatMessage("ai", "🤖 AI 데이터 멘토", "답변을 알기 쉽게 정리하고 있습니다...", loadingMsgId);

            fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question
                })
            })
                .then(async res => {
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "답변을 받아오지 못했습니다.");
                    return data;
                })
                .then(data => {
                    console.log("[CHAT] 후속 질문 응답 수신 완료");
                    updateChatMessage(loadingMsgId, data.answer);
                })
                .catch(err => {
                    console.error("[ERROR] 후속 질문 통신 실패:", err.message);
                    updateChatMessage(loadingMsgId, `⚠️ 오류가 발생했습니다: ${err.message}`);
                })
                .finally(() => {
                    chatSendBtn.disabled = false;
                    chatInput.focus();
                });
        });
    }

    function appendChatMessage(type, senderName, contentText, messageId = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-message ${type}`;
        if (messageId) msgDiv.id = messageId;

        const senderSpan = document.createElement("div");
        senderSpan.className = "message-sender";
        senderSpan.textContent = senderName;

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        if (type === "ai" && window.marked) {
            contentDiv.innerHTML = marked.parse(contentText);
        } else {
            contentDiv.textContent = contentText;
        }

        msgDiv.appendChild(senderSpan);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function updateChatMessage(messageId, newContentMarkdown) {
        const msgDiv = document.getElementById(messageId);
        if (!msgDiv) return;
        const contentDiv = msgDiv.querySelector(".message-content");
        if (contentDiv) {
            if (window.marked) {
                contentDiv.innerHTML = marked.parse(newContentMarkdown);
            } else {
                contentDiv.textContent = newContentMarkdown;
            }
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    copyReportBtn.addEventListener("click", () => {
        if (!rawExplanationText) return;
        navigator.clipboard.writeText(rawExplanationText)
            .then(() => {
                copyReportBtn.textContent = "복사 완료!";
                setTimeout(() => { copyReportBtn.textContent = "보고서 복사"; }, 2000);
            })
            .catch(() => {
                showAlert("클립보드 복사에 실패했습니다.");
            });
    });

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});

