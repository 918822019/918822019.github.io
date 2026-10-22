(() => {
  const groups = {
    gpu: {
      label: "统一 GPU",
      note: "GSM8K test 前 256 题，user-only、greedy、max-new 2048、batch=1；相同 target HQQ/bf16。速度以各方法同 run 的 AR 为基准。",
      metrics: {
        speed: {
          label: "相对 AR 速度",
          unit: "×",
          tip: "完整生成速度；越高越快。",
          rows: [
            ["DFlash full-head", 3.16],
            ["固定子词表", 3.03],
            ["Pcode E3", 2.7],
          ],
        },
        mal: {
          label: "MAL",
          unit: "",
          tip: "平均接受长度，沿用原记录口径；不直接等同于速度。",
          rows: [
            ["DFlash full-head", 4.09],
            ["固定子词表", 3.9, 3.92],
            ["Pcode E3", 3.53],
          ],
        },
      },
    },
    e2: {
      label: "历史 E2",
      note: "历史 byteio / Pcode E2 对照。与统一 GPU 对照的实验口径不同，只在本组内解释。",
      metrics: {
        head: {
          label: "头部耗时",
          unit: " ms/轮",
          tip: "头部微基准；越低越好，不代表整机加速相同比例。",
          rows: [
            ["串行 byteio", 1.38],
            ["Pcode E2", 0.23],
          ],
        },
        speed: {
          label: "相对 AR 速度",
          unit: "×",
          tip: "两种方案的完整生成速度均为 2.85×。",
          rows: [
            ["串行 byteio", 2.85],
            ["Pcode E2", 2.85],
          ],
        },
        mal: {
          label: "MAL",
          unit: "",
          tip: "双码读出降低成本，也带来了接受长度损失。",
          rows: [
            ["串行 byteio", 3.91],
            ["Pcode E2", 3.74],
          ],
        },
      },
    },
    phone: {
      label: "手机 CPU",
      note: "PEQM00 CPU high/normal，4 题 × 2 次，8/8 输出一致；仅该路径和小样本通过验收。low/low 两轮均 0/8 一致，不纳入有效加速对比；OpenCL 尚未完成提速修复。",
      metrics: {
        throughput: {
          label: "请求吞吐",
          unit: " tok/s",
          tip: "b4 为 7.41 tok/s，约为 AR 的 1.70 倍。",
          rows: [
            ["AR", 4.36],
            ["Pcode b4", 7.41],
          ],
        },
        memory: {
          label: "峰值 RSS",
          unit: " MiB",
          tip: "进程峰值驻留内存，越低越省内存。b4 的内存成本明显增加。",
          rows: [
            ["AR", 2874],
            ["Pcode b4", 3972],
          ],
        },
      },
    },
    package: {
      label: "交付体积",
      note: "约 29.7 MB 仅指 E3 draft 侧交付，不包含 target，也不是进程 RSS。原记录没有可同口径比较的完整模型包大小，本面板不补造其他方法的数值。",
      metrics: {
        size: {
          label: "Draft 侧交付",
          unit: " MB",
          tip: "交付包大小与运行内存采用不同口径，不能相互替代。",
          rows: [["Pcode E3", 29.7]],
        },
      },
    },
  };
  function enhance(path, content) {
    if (path !== "docs/infra/推理优化/pcode-report.md") return;
    const section = document.createElement("section");
    section.className = "experiment-panel";
    section.setAttribute("aria-label", "Pcode 实验结果交互图表");
    let group = "gpu",
      metric = "speed",
      hidden = new Set();
    function render() {
      const g = groups[group],
        m = g.metrics[metric];
      const max = Math.max(...m.rows.map((r) => r[2] || r[1]));
      const value = (r) => `${r[1]}${r[2] ? "–" + r[2] : ""}${m.unit}`;
      section.innerHTML = `<p class="eyebrow">EXPLORE THE EVIDENCE</p><h3 id="实验结果交互查看">实验结果交互查看</h3><div class="chart-tabs" role="group" aria-label="实验组">${Object.entries(
        groups,
      )
        .map(
          ([key, x]) =>
            `<button type="button" data-group="${key}" aria-pressed="${key === group}" class="${key === group ? "active" : ""}">${x.label}</button>`,
        )
        .join(
          "",
        )}</div><div class="chart-controls"><label>查看指标 <select class="chart-metric" aria-label="图表指标">${Object.entries(
        g.metrics,
      )
        .map(
          ([key, x]) =>
            `<option value="${key}" ${metric === key ? "selected" : ""}>${x.label}</option>`,
        )
        .join(
          "",
        )}</select></label><span>${m.tip}</span></div><fieldset class="chart-methods"><legend>显示方法</legend>${m.rows.map((r, i) => `<label><input type="checkbox" data-row="${i}" ${hidden.has(i) ? "" : "checked"}>${r[0]}</label>`).join("")}</fieldset><div class="comparison-chart" role="group" aria-label="${g.label}：${m.label}">${m.rows.map((r, i) => (hidden.has(i) ? "" : `<button class="chart-row" type="button" data-detail="${i}" title="${r[0]}：${value(r)}"><span class="chart-name">${r[0]}</span><span class="chart-track"><span class="chart-bar ${r[0].includes("Pcode") ? "pcode-bar" : ""}" style="width:${(r[1] / max) * 100}%"></span>${r[2] ? `<span class="chart-range" style="left:${(r[1] / max) * 100}%;width:${((r[2] - r[1]) / max) * 100}%"></span>` : ""}</span><strong>${value(r)}</strong></button>`)).join("")}${hidden.size === m.rows.length ? '<p class="notice">选择至少一种方法以查看图表。</p>' : ""}</div><p class="chart-detail" aria-live="polite">聚焦或点击数据条查看读数。所有柱形从 0 开始；带虚线的末端表示数值区间。</p><p class="chart-note">${g.note}</p><p class="chart-source">依据：本报告实验摘要 · 证据截至 2026-10-10</p>`;
      section.querySelectorAll("[data-group]").forEach((b) =>
        b.addEventListener("click", () => {
          group = b.dataset.group;
          metric = Object.keys(groups[group].metrics)[0];
          hidden = new Set();
          render();
          section
            .querySelector(`[data-group="${group}"]`)
            .focus({ preventScroll: true });
        }),
      );
      section.querySelector(".chart-metric").addEventListener("change", (e) => {
        metric = e.target.value;
        hidden = new Set();
        render();
        section.querySelector(".chart-metric").focus({ preventScroll: true });
      });
      section.querySelectorAll("[data-row]").forEach((input) =>
        input.addEventListener("change", () => {
          const i = Number(input.dataset.row);
          if (input.checked) hidden.delete(i);
          else hidden.add(i);
          render();
          section
            .querySelector(`[data-row="${i}"]`)
            .focus({ preventScroll: true });
        }),
      );
      section.querySelectorAll("[data-detail]").forEach((button) => {
        const show = () => {
          const r = m.rows[Number(button.dataset.detail)];
          section.querySelector(".chart-detail").textContent =
            `${r[0]} · ${m.label}：${value(r)}。${m.tip}`;
        };
        button.addEventListener("focus", show);
        button.addEventListener("click", show);
      });
    }
    render();
    const heading = [...content.querySelectorAll("h2")].find((h) =>
      h.textContent.startsWith("5."),
    );
    if (heading) heading.after(section);
    else content.append(section);
  }
  window.Experiments = { enhance };
})();
