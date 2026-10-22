const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const base = process.env.TEST_BASE_URL || "http://127.0.0.1:8765";
const artifactDir = path.resolve(".test-artifacts");
const pcode = "docs/infra/推理优化/pcode-report.md";
const qwen = "docs/模型相关的论文/Qwen3-TTS/Qwen3-TTS.md";
const book = "/?doc=" + encodeURIComponent(pcode);

(async () => {
  fs.mkdirSync(artifactDir, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1050 },
      acceptDownloads: true,
      reducedMotion: "reduce",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on("pageerror", (e) => {
      errors.push(e.message);
      console.error("Browser error:", e.message);
    });
    await page.goto(base);
    await page.locator("#post-list .post-card").first().waitFor();
    const generatedSearch = await (
      await page.request.get(base + "/assets/content/search.json")
    ).json();
    const articleCount = generatedSearch.articles.length;
    assert.equal(
      await page.locator("#post-list .post-card").count(),
      articleCount,
    );
    // Body-only query: absent from the article title/summary.
    await page.keyboard.press("/");
    await page.getByRole("searchbox").fill("CSR");
    await page.waitForFunction(() =>
      document.querySelector("#search-count").textContent.includes("找到 1 篇"),
    );
    assert.equal(await page.locator(".search-result mark").count(), 1);
    await page.locator(".search-result").click();
    await page.locator(".experiment-panel").waitFor();
    await page.getByRole("button", { name: "☆ 收藏", exact: true }).click();
    await page.getByRole("button", { name: "◷ 稍后阅读", exact: true }).click();
    await page.locator("#reader-content h2").nth(3).scrollIntoViewIfNeeded();
    await page.waitForTimeout(850);
    const saved = await page.evaluate(
      (p) =>
        JSON.parse(localStorage.getItem("field-notes-library-v1")).entries[p],
      pcode,
    );
    assert.ok(saved.progress > 0.03);
    assert.ok(saved.anchor);
    await page.reload();
    await page.locator("#resume-banner").waitFor({ state: "visible" });
    assert.equal(
      await page
        .getByRole("button", { name: "★ 已收藏", exact: true })
        .getAttribute("aria-pressed"),
      "true",
    );
    await page.getByRole("button", { name: "继续阅读", exact: true }).click();
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(() => scrollY > 300));
    await page.getByRole("link", { name: "书架", exact: true }).click();
    await page.locator(".shelf-row").waitFor();
    assert.equal(await page.locator(".shelf-row").count(), 1);
    await page.screenshot({
      path: path.join(artifactDir, "library-desktop.png"),
      fullPage: false,
    });
    await page.getByRole("link", { name: "稍后阅读", exact: true }).click();
    await page.locator(".shelf-row").waitFor();
    await page.getByRole("link", { name: "最近阅读", exact: true }).click();
    await page.locator(".shelf-row").waitFor();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "导出备份", exact: true }).click();
    const download = await downloadPromise;
    const exportPath = path.join(artifactDir, "reading-backup.json");
    await download.saveAs(exportPath);
    const backup = JSON.parse(fs.readFileSync(exportPath, "utf8"));
    assert.equal(backup.entries[pcode].favorite, true);
    await page.getByRole("link", { name: "我的收藏", exact: true }).click();
    await page.locator(".shelf-row").waitFor();
    await page.getByRole("button", { name: "移出列表", exact: true }).click();
    await page.locator(".shelf-empty").waitFor();
    await page.locator("#library-import").setInputFiles(exportPath);
    await page.locator(".shelf-row").waitFor();
    await page.locator("#library-import").setInputFiles({
      name: "bad.json",
      mimeType: "application/json",
      buffer: Buffer.from("{broken"),
    });
    await page.waitForFunction(() =>
      document
        .querySelector("#feature-status")
        .textContent.includes("不是有效的 JSON"),
    );
    assert.equal(await page.locator(".shelf-row").count(), 1);
    await page.goto(base + book);
    await page.locator("#reader-content table").first().waitFor();
    await page
      .getByRole("button", { name: "Aa 阅读设置", exact: true })
      .click();
    await page.locator("#reading-size").fill("20");
    await page.locator("#reading-spacing").fill("2.3");
    await page.locator("#reading-width").selectOption("wide");
    await page.getByRole("button", { name: "关闭阅读设置" }).click();
    await page.getByRole("button", { name: "专注阅读", exact: true }).click();
    assert.equal(await page.locator(".reader-aside").isVisible(), false);
    await page.reload();
    await page.locator("#reader-content table").first().waitFor();
    assert.equal(
      await page
        .locator("#reader-content")
        .evaluate((el) => getComputedStyle(el).fontSize),
      "20px",
    );
    await page.getByRole("button", { name: "退出专注", exact: true }).click();
    await page
      .getByRole("button", { name: "Aa 阅读设置", exact: true })
      .click();
    await page.getByRole("button", { name: "恢复默认", exact: true }).click();
    await page.keyboard.press("Escape");
    const panel = page.locator(".experiment-panel");
    assert.equal(await panel.locator(".chart-row").count(), 3);
    await panel.locator(".chart-metric").selectOption("mal");
    assert.match(
      await panel.locator(".comparison-chart").innerText(),
      /3.9–3.92/,
    );
    await panel
      .getByRole("checkbox", { name: "固定子词表", exact: true })
      .uncheck();
    assert.equal(await panel.locator(".chart-row").count(), 2);
    await panel.getByRole("button", { name: "手机 CPU", exact: true }).click();
    assert.match(await panel.locator(".comparison-chart").innerText(), /7.41/);
    await panel.locator(".chart-metric").selectOption("memory");
    assert.match(await panel.locator(".comparison-chart").innerText(), /3972/);
    assert.match(await panel.locator(".chart-note").innerText(), /0\/8/);
    assert.equal(await page.locator('[id="实验结果交互查看"]').count(), 1);
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({
      path: path.join(artifactDir, "experiment-desktop.png"),
      fullPage: false,
    });
    await page.goto(base + "/?view=series");
    await page.locator(".series-card").first().waitFor();
    assert.equal(await page.locator(".series-card").count(), 4);
    await page
      .getByRole("link", { name: "TVM：从原理到运行", exact: true })
      .click();
    await page.waitForFunction(
      () => document.querySelectorAll(".series-card").length === 1,
    );
    await page.locator(".series-card ol a").first().click();
    await page.locator("#series-neighbors a").first().waitFor();
    assert.match(
      await page.locator("#series-context").innerText(),
      /第 1 \/ 3/,
    );
    await page.locator("#series-neighbors a").click();
    await page.waitForFunction(() =>
      document
        .querySelector("#series-context")
        .textContent.includes("第 2 / 3"),
    );
    assert.ok((await page.locator("pre .token").count()) > 0);
    const fold = page
      .getByRole("button", { name: "展开完整代码", exact: true })
      .first();
    await fold.click();
    assert.equal(
      await page
        .getByRole("button", { name: "收起代码", exact: true })
        .first()
        .getAttribute("aria-expanded"),
      "true",
    );
    await page.goto(base + "/?doc=" + encodeURIComponent(qwen));
    await page.locator("#reader-content img").first().waitFor();
    await page.waitForFunction(() =>
      [...document.querySelectorAll("#reader-content img")].every(
        (img) => img.complete && img.naturalWidth > 0,
      ),
    );
    const imageCount = await page.locator("#reader-content img").count();
    await page.locator("#reader-content img").first().click();
    assert.equal(
      await page.locator("#image-dialog").evaluate((el) => el.open),
      true,
    );
    await page.getByRole("button", { name: "放大图片", exact: true }).click();
    assert.match(await page.locator("#image-counter").innerText(), /150%/);
    await page.getByRole("button", { name: "下一张图片", exact: true }).click();
    assert.ok(
      (await page.locator("#image-counter").innerText()).startsWith(
        "2 / " + imageCount,
      ),
    );
    await page.keyboard.press("Escape");
    assert.equal(
      await page.locator("#image-dialog").evaluate((el) => el.open),
      false,
    );
    await page.goto(base + "/?view=archive");
    await page.locator(".archive-month").first().waitFor();
    assert.equal(await page.locator(".archive-month li").count(), articleCount);
    const feed = await page.request.get(base + "/feed.xml");
    assert.equal(feed.status(), 200);
    assert.equal((await feed.text()).match(/<item>/g).length, articleCount);
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const route of [
        "/",
        book,
        "/?view=library",
        "/?view=series",
        "/?view=archive",
      ]) {
        await page.goto(base + route);
        await page.waitForFunction(() =>
          document.querySelector("#post-list .post-card"),
        );
        if (route === book) await page.locator(".experiment-panel").waitFor();
        if (route.includes("series"))
          await page.locator(".series-card").first().waitFor();
        assert.ok(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
          `overflow ${width} ${route}`,
        );
      }
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(base + "/?view=library");
    await page.locator(".shelf-row").first().waitFor();
    await page.screenshot({
      path: path.join(artifactDir, "library-mobile.png"),
    });
    await page.goto(base + "/?view=library&tab=history");
    await page.locator(".shelf-row").first().waitFor();
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "清空历史", exact: true }).click();
    await page.locator(".shelf-empty").waitFor();
    const cleared = await page.evaluate(
      (p) =>
        JSON.parse(localStorage.getItem("field-notes-library-v1")).entries[p],
      pcode,
    );
    assert.equal(cleared.visitedAt, 0);
    assert.equal(cleared.progress, 0);
    assert.equal(cleared.favorite, true);
    assert.equal(cleared.later, true);
    assert.deepEqual(errors, []);
    console.log(
      "PASS: full-text results/highlights, favorites, later list, history/resume, export/import, settings/focus persistence, series/previous/next, code highlight/fold, image viewer/zoom/navigation, archives/RSS, charts, 320–1440px layouts, no JS errors.",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
