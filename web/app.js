// 서버에서 받은 사례 재생. 실제 환자 모드는 로컬 전용
// 다음 시간 정답은 사후 확인을 켠 경우에만 표시
"use strict";
let state;
const byId = (id) => document.getElementById(id);
const number = (value) => Number(value).toFixed(1);
const svgNS = "http://www.w3.org/2000/svg";

function statusFor(value) {
  return value < state.policy.lower_bpm
    ? "하한 미만"
    : value > state.policy.upper_bpm
      ? "상한 초과"
      : "설정 범위 안";
}

function svgElement(name, attributes, text) {
  const element = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes))
    element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawChart(caseData, index, reveal) {
  const svg = byId("chart");
  svg.replaceChildren();
  const width = Math.max(340, Math.round(svg.getBoundingClientRect().width));
  const right = width - 36;
  svg.setAttribute("viewBox", `0 0 ${width} 300`);
  const rows = caseData.rows.slice(0, index + 1);
  const low = state.policy.lower_bpm,
    high = state.policy.upper_bpm;
  const visible = rows.flatMap((row, i) => [
    row.prediction,
    row.baseline,
    ...(i < index || reveal ? [row.actual] : []),
  ]);
  const minimum = Math.floor((Math.min(low, ...visible) - 10) / 10) * 10;
  const maximum = Math.ceil((Math.max(high, ...visible) + 10) / 10) * 10;
  const x = (value) =>
    52 + (value / Math.max(caseData.rows.length, 1)) * (right - 52);
  const y = (value) => 248 - ((value - minimum) / (maximum - minimum)) * 218;
  svg.append(
    svgElement("rect", {
      x: 52,
      y: y(high),
      width: right - 52,
      height: y(low) - y(high),
      fill: "#f1f8f5",
    }),
  );
  for (let tick = minimum; tick <= maximum; tick += 20) {
    svg.append(
      svgElement("line", {
        x1: 52,
        x2: right,
        y1: y(tick),
        y2: y(tick),
        stroke: "#e5ebe8",
      }),
    );
    svg.append(
      svgElement(
        "text",
        {
          x: 40,
          y: y(tick) + 4,
          "text-anchor": "end",
          fill: "#71818a",
          "font-size": 10,
        },
        tick,
      ),
    );
  }
  for (const bound of [low, high]) {
    svg.append(
      svgElement("line", {
        x1: 52,
        x2: right,
        y1: y(bound),
        y2: y(bound),
        stroke: "#c59965",
        "stroke-dasharray": "4 4",
      }),
    );
    svg.append(
      svgElement(
        "text",
        { x: right + 4, y: y(bound) + 4, fill: "#997441", "font-size": 10 },
        bound,
      ),
    );
  }
  for (let tick = 0; tick <= caseData.rows.length; tick += width < 500 ? 8 : 4)
    svg.append(
      svgElement(
        "text",
        {
          x: x(tick),
          y: 268,
          "text-anchor": "middle",
          fill: "#71818a",
          "font-size": 10,
        },
        `+${tick}h`,
      ),
    );
  svg.append(
    svgElement(
      "text",
      { x: 52, y: 15, fill: "#71818a", "font-size": 10 },
      "bpm",
    ),
  );
  svg.append(
    svgElement(
      "text",
      {
        x: width / 2,
        y: 291,
        "text-anchor": "middle",
        fill: "#71818a",
        "font-size": 10,
      },
      width < 500
        ? "상대 시간 · 다음 구간 끝"
        : "첫 예측 기준부터의 상대 시간 · 각 점은 다음 시간 구간의 끝에 배치",
    ),
  );
  const origin = caseData.rows[index].hour;
  svg.append(
    svgElement("line", {
      x1: x(origin),
      x2: x(origin),
      y1: 25,
      y2: 248,
      stroke: "#a2b6ae",
      "stroke-dasharray": "3 4",
    }),
  );
  for (const [key, color, dash] of [
    ["baseline", "#b79773", "4 4"],
    ["actual", "#304b62", ""],
    ["prediction", "#128777", ""],
  ]) {
    const available = rows.filter(
      (_, i) => key !== "actual" || i < index || reveal,
    );
    const points = available
      .map((row) => `${x(row.hour + 1)},${y(row[key])}`)
      .join(" ");
    svg.append(
      svgElement("polyline", {
        points,
        fill: "none",
        stroke: color,
        "stroke-width": key === "baseline" ? 1.5 : 2.4,
        "stroke-dasharray": dash,
      }),
    );
    for (const row of available)
      svg.append(
        svgElement("circle", {
          cx: x(row.hour + 1),
          cy: y(row[key]),
          r: 2.5,
          fill: color,
        }),
      );
  }
  const last = rows[rows.length - 1];
  svg.append(
    svgElement("circle", {
      cx: x(last.hour + 1),
      cy: y(last.prediction),
      r: 5,
      fill: "#fff",
      stroke: "#128777",
      "stroke-width": 2,
    }),
  );
}

function renderCase() {
  const selected = state.cases.find(
    (item) => item.id === byId("case-select").value,
  );
  const index = Number(byId("time-slider").value);
  const row = selected.rows[index];
  const reveal = byId("reveal").checked;
  byId("time-label").textContent =
    `예측 기준 +${row.hour}h → 다음 구간 (${row.hour}, ${row.hour + 1}]h`;
  byId("step-label").textContent = `${index + 1} / ${selected.rows.length}`;
  byId("next").disabled = index === selected.rows.length - 1;
  byId("last-value").textContent = number(row.inputs[5]);
  byId("prediction").textContent = number(row.prediction);
  byId("alert-status").textContent = statusFor(row.prediction);
  byId("status-card").className =
    statusFor(row.prediction) === "설정 범위 안" ? "inside" : "outside";
  byId("actual").textContent = reveal ? number(row.actual) : "숨김";
  byId("actual-note").textContent = reveal
    ? `${statusFor(row.actual)} · 사후 검증용`
    : "예측 시점에는 알 수 없는 값";
  byId("comparison-model").textContent = state.model.name;
  byId("comparison-prediction").textContent = number(row.prediction);
  byId("baseline-prediction").textContent = number(row.baseline);
  byId("prediction-error").textContent = reveal ? number(Math.abs(row.prediction - row.actual)) : "숨김";
  byId("baseline-error").textContent = reveal ? number(Math.abs(row.baseline - row.actual)) : "숨김";
  byId("comparison-note").textContent = reveal
    ? "절대오차 = |예측값 − 사후 실제값|. 반올림 전 값으로 계산합니다. 이 한 사례의 결과는 전체 성능을 뜻하지 않습니다."
    : "사후 정답을 켜면 두 방법의 절대오차를 비교할 수 있습니다. 정답은 모델 입력에 포함되지 않습니다.";
  byId("input-rows").replaceChildren();
  row.inputs.forEach((value, i) => {
    const tr = document.createElement("tr");
    for (const text of [
      `(${i - 6}, ${i - 5}]h`,
      number(value),
      `${row.masks[i]} · ${row.masks[i] ? "관측" : "보완"}`,
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    }
    byId("input-rows").append(tr);
  });
  byId("chart-note").textContent = reveal
    ? "사후 비교 모드 · 현재 예측의 정답을 함께 표시합니다. 정답은 모델 입력에 포함되지 않습니다."
    : "저장된 사례 재생 · 관측선은 사후 확보한 과거 구간 정답입니다. 당시 저장 지연을 재생하는 선은 아니며, 새 예측의 정답은 숨깁니다.";
  drawChart(selected, index, reveal);
}

async function initialize() {
  const response = await fetch("/api/state");
  if (!response.ok)
    throw new Error("사례를 불러오지 못했습니다. 로컬 서버를 확인하세요.");
  state = await response.json();
  if (!state.cases.length) throw new Error("표시할 사례가 없습니다.");
  const baseline = state.model.kind === "baseline";
  const local = state.mode === "local";
  byId("mode").textContent = local
    ? "실제 환자 사례 · 로컬 전용"
    : baseline
      ? "가상 데이터 · baseline 시연"
      : "가상 데이터 · LSTM 추론";
  byId("notice").textContent = local
    ? "로컬 전용 환자 자료입니다. 화면·사례를 외부에 공유하지 마세요. 현재 장치에서 다시 계산한 예측으로, 기존 GPU 평가와 미세한 수치 차이가 있을 수 있습니다. 표시 규칙은 임상 진단이 아닙니다."
    : baseline
      ? "데이터 없이 실행하는 화면 시연입니다. 현재 예측은 6시간 평균 baseline이며, 보고서의 학습된 LSTM이 아닙니다."
      : "가상 입력을 지정한 LSTM으로 예측합니다. CPU/GPU에 따라 작은 수치 차이가 있을 수 있습니다. 임상 경보 기준이 아닙니다.";
  byId("model-name").textContent = state.model.name;
  byId("model-detail").textContent = baseline
    ? "학습 모델 파일 없이 실행"
    : `epoch ${state.model.epoch} · ${state.model.device}`;
  byId("workflow-source").textContent = local
    ? "로컬 MIMIC 사례"
    : "가상 시계열";
  byId("workflow-source-note").textContent = local
    ? "전처리된 ICU 기록"
    : "수식으로 생성한 입력";
  byId("workflow-input").textContent = local
    ? "저장 시각 반영 입력"
    : "6시간 입력 준비";
  byId("workflow-input-note").textContent = local
    ? "전처리 결과를 읽음"
    : "환자 기록 사용 없음";
  byId("workflow-model").textContent = baseline ? "6시간 평균" : "고정 LSTM";
  byId("workflow-model-note").textContent = baseline
    ? "baseline 계산"
    : "학습 기준 정규화";
  byId("prediction-note").textContent = `bpm · ${state.model.name}`;
  byId("prediction-legend").textContent = baseline
    ? "baseline 예측"
    : "LSTM 예측";
  byId("predict-button").textContent = baseline
    ? "평균으로 예측하기"
    : "LSTM으로 예측하기";
  byId("input-note").textContent = local
    ? "예측 시점까지 저장된 기록으로 구성한 입력입니다. 관측값 1, 이전 구간에서 보완한 값 0입니다."
    : "수식으로 생성한 가상 입력입니다. 실제 전처리에서는 예측 시점의 저장 가용성과 결측 보완 규칙을 적용합니다.";
  byId("rule-label").textContent =
    `${state.policy.lower_bpm} 미만 / ${state.policy.upper_bpm} 초과 · 경계값은 범위 안`;
  byId("test-mae").textContent = state.test_metrics.lstm.mae_bpm.toFixed(3);
  byId("test-base").textContent =
    state.test_metrics.six_hour_mean.mae_bpm.toFixed(3);
  for (const item of state.cases) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    byId("case-select").append(option);
  }
  const changeCase = () => {
    const selected = state.cases.find(
      (item) => item.id === byId("case-select").value,
    );
    byId("time-slider").max = selected.rows.length - 1;
    byId("time-slider").value = selected.anchor;
    renderCase();
  };
  byId("case-select").addEventListener("change", changeCase);
  byId("time-slider").addEventListener("input", renderCase);
  byId("reveal").addEventListener("change", renderCase);
  window.addEventListener("resize", renderCase);
  byId("next").addEventListener("click", () => {
    byId("time-slider").value = Number(byId("time-slider").value) + 1;
    renderCase();
  });
  for (let i = 0; i < 6; i++) {
    const slot = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = `${i === 5 ? "최근" : 6 - i + "시간 전"} 구간`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "300";
    input.step = "any";
    input.required = true;
    input.value = 80 + i;
    input.setAttribute("aria-label", `입력 ${i + 1}`);
    label.append(input);
    const missingLabel = document.createElement("label");
    missingLabel.className = "missing-toggle";
    const missing = document.createElement("input");
    missing.type = "checkbox";
    missing.setAttribute("aria-label", `${i + 1}번째 구간 미측정`);
    missingLabel.append(missing, " 미측정");
    missing.addEventListener("change", () => {
      input.disabled = missing.checked;
    });
    slot.append(label, missingLabel);
    byId("manual-inputs").append(slot);
  }
  let inputVersion = 0;
  const clearManualResult = () => {
    inputVersion += 1;
    byId("manual-result").textContent = "입력이 변경되었습니다. 예측 버튼을 눌러 다시 계산하세요.";
  };
  byId("manual-form").addEventListener("input", clearManualResult);
  byId("manual-form").addEventListener("change", clearManualResult);
  byId("manual-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submittedVersion = inputVersion;
    byId("predict-button").disabled = true;
    byId("manual-result").textContent = "예측 계산 중…";
    try {
      const values = Array.from(
        byId("manual-inputs").querySelectorAll('input[type="number"]'),
      ).map((input) => input.disabled || input.value.trim() === "" ? null : Number(input.value));
      const masks = Array.from(
        byId("manual-inputs").querySelectorAll('input[type="checkbox"]'),
      ).map((input) => input.checked ? 0 : 1);
      const result = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values, masks }),
      });
      const data = await result.json();
      if (submittedVersion !== inputVersion) return;
      if (!result.ok) throw new Error(data.error);
      const filled = data.masks.some((mask) => mask === 0)
        ? ` 보완 후 입력: ${data.inputs.map(number).join(", ")} / 관측 표시: ${data.masks.join(", ")}.`
        : "";
      byId("manual-result").textContent =
        `다음 시간 평균 ${number(data.prediction)} bpm · ${statusFor(data.prediction)} | 6시간 평균 baseline ${number(data.baseline)} bpm.${filled} 실제 다음 값은 아직 알 수 없습니다.`;
    } catch (error) {
      if (submittedVersion === inputVersion) byId("manual-result").textContent = error.message;
    } finally {
      byId("predict-button").disabled = false;
    }
  });
  changeCase();
}
initialize().catch((error) => {
  byId("error").hidden = false;
  byId("error").textContent = error.message;
});
